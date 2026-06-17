from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import socket
import struct
import statistics
import threading
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from netio import safe_close_conn, send_packet


STOP_EVENT = threading.Event()

PREFERRED_MODEL_ORDER = {
    "best_baseline": 0,
    "best_kd": 1,
    "best_p2": 2,
    "best_p2_and_kd": 3,
    "best_80": 0,
    "best_320": 1,
    "best_640": 2,
    "best_s": 3,
    "best_m": 4,
}


def _sigint_handler(sig, frame):
    if not STOP_EVENT.is_set():
        print("\n[SEG-LAT] Ctrl+C received -> stopping...", flush=True)
    STOP_EVENT.set()


signal.signal(signal.SIGINT, _sigint_handler)


def decode_jpg(jpg: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG.")
    return img


def parse_control_packet(payload: bytes) -> Optional[Dict[str, Any]]:
    if not payload or payload[:1] not in (b"{", b"["):
        return None
    try:
        msg = json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    if isinstance(msg, dict) and msg.get("type") == "control":
        return msg
    return None


def recv_packet_with_limits(
    conn: socket.socket,
    stop_event: threading.Event,
    should_stop,
    idle_timeout: float,
) -> bytes:
    last_progress = time.time()

    def recv_exact(n: int) -> bytes:
        nonlocal last_progress
        buf = bytearray()
        while len(buf) < n:
            if stop_event.is_set() or should_stop():
                raise InterruptedError("Stopped by limit or user.")
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                if time.time() - last_progress >= idle_timeout:
                    raise TimeoutError(f"No socket data for {idle_timeout:.1f}s.")
                continue
            if not chunk:
                raise ConnectionError("Socket closed while receiving.")
            last_progress = time.time()
            buf.extend(chunk)
        return bytes(buf)

    header = recv_exact(4)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > 50_000_000:
        raise ValueError(f"Invalid packet length: {length}")
    return recv_exact(length)


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if not target.is_relative_to(dest_root):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(dest_dir)


def model_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem.lower()
    return PREFERRED_MODEL_ORDER.get(stem, 100), stem


def discover_model_paths(models_zip: Optional[Path], models_dir: Optional[Path], run_dir: Path) -> List[Path]:
    if models_dir is not None:
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
        paths = sorted(models_dir.rglob("*.pt"), key=model_sort_key)
    elif models_zip is not None:
        if not models_zip.exists():
            raise FileNotFoundError(f"Models zip does not exist: {models_zip}")
        extracted_dir = run_dir / "extracted_models"
        print(f"[SEG-LAT] Extracting models: {models_zip} -> {extracted_dir}", flush=True)
        safe_extract_zip(models_zip, extracted_dir)
        paths = sorted(extracted_dir.rglob("*.pt"), key=model_sort_key)
    else:
        raise ValueError("Pass either --models-zip or --models-dir.")

    if not paths:
        raise FileNotFoundError("No .pt segmentation models found.")
    return paths


def normalize_imgsz(raw: Any, fallback: int) -> int:
    if isinstance(raw, (list, tuple)) and raw:
        return int(raw[0])
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        match = re.search(r"\d+", raw)
        if match:
            return int(match.group(0))
    return int(fallback)


def fallback_imgsz_from_name(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return 640


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def make_run_dir(output_dir: Path) -> Tuple[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / base
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.name, candidate


class SegForwardModel:
    def __init__(self, path: Path, device: torch.device, use_half: bool):
        self.path = path
        self.name = path.stem
        self.device = device
        self.use_half = bool(use_half and device.type == "cuda")

        yolo = YOLO(str(path))
        self.task = str(getattr(yolo, "task", ""))
        self.names = getattr(yolo, "names", {})
        self.net = yolo.model.to(device).eval()

        if self.use_half:
            self.net.half()
            self.dtype = "float16"
        else:
            self.net.float()
            self.dtype = "float32"

        args = getattr(self.net, "args", {}) or {}
        self.imgsz = normalize_imgsz(args.get("imgsz"), fallback_imgsz_from_name(path))
        self.stride = self._format_stride(getattr(self.net, "stride", None))

        if self.task != "segment":
            print(f"[SEG-LAT] Warning: {self.name} task={self.task!r}, expected 'segment'.", flush=True)

    @staticmethod
    def _format_stride(stride: Any) -> str:
        try:
            if hasattr(stride, "detach"):
                return ",".join(str(int(x)) for x in stride.detach().cpu().flatten().tolist())
            if isinstance(stride, Iterable):
                return ",".join(str(int(x)) for x in stride)
        except Exception:
            pass
        return str(stride)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "task": self.task,
            "imgsz": self.imgsz,
            "stride": self.stride,
            "dtype": self.dtype,
            "names": self.names,
        }

    def prepare_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(frame_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tensor = torch.from_numpy(chw).to(self.device, non_blocking=True)
        tensor = tensor.unsqueeze(0)
        if self.use_half:
            tensor = tensor.half()
        else:
            tensor = tensor.float()
        tensor = tensor / 255.0
        return tensor

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def warmup(self, runs: int) -> None:
        if runs <= 0:
            return
        dtype = torch.float16 if self.use_half else torch.float32
        x = torch.zeros((1, 3, self.imgsz, self.imgsz), dtype=dtype, device=self.device)
        for _ in range(runs):
            _ = self.net(x)
        self._sync()

    @torch.inference_mode()
    def forward_latency_ms(self, tensor: torch.Tensor) -> float:
        self._sync()
        t0 = time.perf_counter()
        _ = self.net(tensor)
        self._sync()
        return (time.perf_counter() - t0) * 1000.0


class SegLatencyBenchmark:
    def __init__(self, model_paths: List[Path], device: torch.device, use_half: bool, warmup_runs: int, rotate_order: bool):
        self.device = device
        self.use_half = bool(use_half and device.type == "cuda")
        self.rotate_order = rotate_order
        self.models: List[SegForwardModel] = []

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        for path in model_paths:
            model = SegForwardModel(path, device=device, use_half=self.use_half)
            self.models.append(model)
            print(
                f"[SEG-LAT] Loaded {model.name}: task={model.task} imgsz={model.imgsz} "
                f"dtype={model.dtype} path={model.path}",
                flush=True,
            )

        for model in self.models:
            print(f"[SEG-LAT] Warmup {model.name}: {warmup_runs} forward runs", flush=True)
            model.warmup(warmup_runs)

    def metadata(self) -> List[Dict[str, Any]]:
        return [model.metadata() for model in self.models]

    def run_frame(self, frame_bgr: np.ndarray, frame_index: int) -> List[Dict[str, Any]]:
        prepared = [(model, model.prepare_tensor(frame_bgr)) for model in self.models]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        rows: List[Dict[str, Any]] = []
        model_count = len(prepared)
        order_offset = (frame_index - 1) % model_count if self.rotate_order and model_count > 0 else 0
        for order_index in range(model_count):
            model, tensor = prepared[(order_offset + order_index) % model_count]
            latency_ms = model.forward_latency_ms(tensor)
            rows.append(
                {
                    "model_name": model.name,
                    "model_path": str(model.path),
                    "imgsz": model.imgsz,
                    "device": str(self.device),
                    "dtype": model.dtype,
                    "latency_ms": latency_ms,
                    "order_index": order_index,
                    "order_offset": order_offset,
                }
            )
        return rows


class LatencyRecorder:
    def __init__(self, run_id: str, run_dir: Path, benchmark: SegLatencyBenchmark, args: argparse.Namespace):
        self.run_id = run_id
        self.run_dir = run_dir
        self.benchmark = benchmark
        self.args = args
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.finished_at: Optional[str] = None
        self.measurement_started_at: Optional[str] = None
        self.measurement_stopped_at: Optional[str] = None
        self.measurement_status = "idle"
        self.frame_count = 0
        self.values: Dict[str, List[float]] = defaultdict(list)
        self.order_indices: Dict[str, List[int]] = defaultdict(list)
        self._closed = False

        self.per_frame_csv = run_dir / "per_frame_latency.csv"
        self.summary_csv = run_dir / "summary_latency.csv"
        self.metadata_json = run_dir / "metadata.json"

        self._fh = self.per_frame_csv.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "run_id",
                "frame_index",
                "recv_wall_time",
                "image_w",
                "image_h",
                "jpg_bytes",
                "model_name",
                "model_path",
                "imgsz",
                "device",
                "dtype",
                "order_index",
                "order_offset",
                "latency_ms",
            ],
        )
        self._writer.writeheader()
        self.write_metadata()

    def begin_measurement(self) -> None:
        if self._closed:
            raise RuntimeError("Recorder is already closed.")
        self.measurement_started_at = datetime.now().isoformat(timespec="seconds")
        self.measurement_stopped_at = None
        self.measurement_status = "recording"
        self.frame_count = 0
        self.values.clear()
        self.order_indices.clear()

        self._fh.seek(0)
        self._fh.truncate(0)
        self._writer.writeheader()
        self._fh.flush()
        self.write_metadata()

    def stop_measurement(self) -> None:
        if self._closed:
            return
        self.measurement_stopped_at = datetime.now().isoformat(timespec="seconds")
        self.measurement_status = "stopped"
        self.write_summary()
        self.write_metadata()

    def write_frame(self, frame_index: int, frame_bgr: np.ndarray, jpg_bytes: int, rows: List[Dict[str, Any]]) -> None:
        h, w = frame_bgr.shape[:2]
        recv_wall_time = datetime.now().isoformat(timespec="milliseconds")
        for row in rows:
            latency = float(row["latency_ms"])
            model_name = str(row["model_name"])
            self.values[model_name].append(latency)
            self.order_indices[model_name].append(int(row.get("order_index", 0)))
            self._writer.writerow(
                {
                    "run_id": self.run_id,
                    "frame_index": frame_index,
                    "recv_wall_time": recv_wall_time,
                    "image_w": w,
                    "image_h": h,
                    "jpg_bytes": jpg_bytes,
                    "model_name": row["model_name"],
                    "model_path": row["model_path"],
                    "imgsz": row["imgsz"],
                    "device": row["device"],
                    "dtype": row["dtype"],
                    "order_index": int(row.get("order_index", 0)),
                    "order_offset": int(row.get("order_offset", 0)),
                    "latency_ms": f"{latency:.6f}",
                }
            )
        self.frame_count = max(self.frame_count, frame_index)
        self._fh.flush()

    def summary_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for model in self.benchmark.models:
            vals = self.values.get(model.name, [])
            orders = self.order_indices.get(model.name, [])
            if vals:
                avg_ms = statistics.fmean(vals)
                row = {
                    "model_name": model.name,
                    "model_path": str(model.path),
                    "imgsz": model.imgsz,
                    "device": str(self.benchmark.device),
                    "dtype": model.dtype,
                    "frames": len(vals),
                    "avg_ms": avg_ms,
                    "median_ms": statistics.median(vals),
                    "p90_ms": percentile(vals, 90),
                    "p95_ms": percentile(vals, 95),
                    "p99_ms": percentile(vals, 99),
                    "min_ms": min(vals),
                    "max_ms": max(vals),
                    "std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "fps_from_avg": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
                    "avg_order_index": statistics.fmean(orders) if orders else 0.0,
                    "first_position_count": sum(1 for idx in orders if idx == 0),
                }
            else:
                row = {
                    "model_name": model.name,
                    "model_path": str(model.path),
                    "imgsz": model.imgsz,
                    "device": str(self.benchmark.device),
                    "dtype": model.dtype,
                    "frames": 0,
                    "avg_ms": 0.0,
                    "median_ms": 0.0,
                    "p90_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "std_ms": 0.0,
                    "fps_from_avg": 0.0,
                    "avg_order_index": 0.0,
                    "first_position_count": 0,
                }
            rows.append(row)
        return rows

    def write_summary(self) -> None:
        fieldnames = [
            "model_name",
            "model_path",
            "imgsz",
            "device",
            "dtype",
            "frames",
            "avg_ms",
            "median_ms",
            "p90_ms",
            "p95_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
            "std_ms",
            "fps_from_avg",
            "avg_order_index",
            "first_position_count",
        ]
        with self.summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.summary_rows():
                formatted = dict(row)
                for key in fieldnames:
                    if key.endswith("_ms") or key in ("fps_from_avg", "avg_order_index"):
                        formatted[key] = f"{float(formatted[key]):.6f}"
                writer.writerow(formatted)

    def write_metadata(self) -> None:
        data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "measurement_started_at": self.measurement_started_at,
            "measurement_stopped_at": self.measurement_stopped_at,
            "measurement_status": self.measurement_status,
            "frame_count": self.frame_count,
            "measurement_scope": "PC CUDA/PyTorch model forward only: preprocess, decode, NMS, mask postprocess, tracking, and network time excluded",
            "rotate_order": self.benchmark.rotate_order,
            "args": vars(self.args),
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            },
            "models": self.benchmark.metadata(),
            "artifacts": {
                "per_frame_csv": str(self.per_frame_csv),
                "summary_csv": str(self.summary_csv),
                "metadata_json": str(self.metadata_json),
            },
        }
        self.metadata_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self._closed:
            return
        self.finished_at = datetime.now().isoformat(timespec="seconds")
        if self.measurement_status != "stopped":
            self.write_summary()
        self.write_metadata()
        self._fh.close()
        self._closed = True


class SegLatencyServer:
    def __init__(
        self,
        benchmark: SegLatencyBenchmark,
        recorder: LatencyRecorder,
        host: str,
        port: int,
        max_frames: int,
        duration_sec: float,
        client_idle_timeout: float,
        print_interval: float,
        send_replies: bool,
        exit_after_stop: bool,
    ):
        self.benchmark = benchmark
        self.recorder = recorder
        self.host = host
        self.port = int(port)
        self.max_frames = int(max_frames)
        self.duration_sec = float(duration_sec)
        self.client_idle_timeout = float(client_idle_timeout)
        self.print_interval = float(print_interval)
        self.send_replies = bool(send_replies)
        self.exit_after_stop = bool(exit_after_stop)
        self._stream_frames_seen = 0
        self._started_at = time.time()
        self._recording = False
        self._measurement_completed = False

    def _should_stop_for_limits(self) -> bool:
        if self.exit_after_stop and self._measurement_completed:
            return True
        if self.max_frames > 0 and self.recorder.frame_count >= self.max_frames:
            return True
        if self.duration_sec > 0 and time.time() - self._started_at >= self.duration_sec:
            return True
        return False

    @staticmethod
    def _format_frame_log(frame_index: int, rows: List[Dict[str, Any]]) -> str:
        parts = [f"{row['model_name']}={float(row['latency_ms']):.2f}ms" for row in rows]
        return f"[SEG-LAT] frame={frame_index} | " + " | ".join(parts)

    def _summary_text(self) -> str:
        rows = self.recorder.summary_rows()
        measured = sum(int(row["frames"]) for row in rows[:1])
        if measured <= 0:
            return "[SEG-LAT] stopped | frames=0"
        parts = [
            f"{row['model_name']} avg={float(row['avg_ms']):.2f}ms"
            for row in rows
        ]
        return f"[SEG-LAT] stopped | frames={measured} | " + " | ".join(parts)

    def _status_text(self) -> str:
        if self._recording:
            return f"[SEG-LAT] recording | frames={self.recorder.frame_count} | press A to stop"
        if self._measurement_completed:
            return self._summary_text()
        return "[SEG-LAT] idle | press A to start"

    def _build_status_response(self, pc_log: Optional[str] = None) -> Dict[str, Any]:
        return {
            "ts": time.time(),
            "img_w": 1,
            "img_h": 1,
            "tiles": [],
            "hand": [],
            "hand_stable": False,
            "advice": {
                "benefit": {"tile_id": -1, "tile": "", "source": "seg_latency", "reason": "latency measurement"},
                "safe": {"tile_id": -1, "tile": "", "source": "seg_latency", "reason": "latency measurement"},
            },
            "pc_log": pc_log or self._status_text(),
            "debug": {
                "mode": "segmentation_forward_latency",
                "forward_only": True,
                "recording": self._recording,
                "measurement_completed": self._measurement_completed,
                "frames": self.recorder.frame_count,
            },
        }

    def _build_frame_response(self, frame_bgr: np.ndarray, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        return {
            "ts": time.time(),
            "img_w": int(w),
            "img_h": int(h),
            "tiles": [],
            "hand": [],
            "hand_stable": False,
            "advice": {
                "benefit": {"tile_id": -1, "tile": "", "source": "seg_latency", "reason": "latency test"},
                "safe": {"tile_id": -1, "tile": "", "source": "seg_latency", "reason": "latency test"},
            },
            "pc_log": self._format_frame_log(self.recorder.frame_count, rows),
            "debug": {
                "mode": "segmentation_forward_latency",
                "forward_only": True,
                "recording": self._recording,
                "measurement_completed": self._measurement_completed,
                "frames": self.recorder.frame_count,
                "latency": [
                    {
                        "model": row["model_name"],
                        "imgsz": int(row["imgsz"]),
                        "latency_ms": float(row["latency_ms"]),
                        "order_index": int(row.get("order_index", 0)),
                        "order_offset": int(row.get("order_offset", 0)),
                    }
                    for row in rows
                ],
            },
        }

    def _send_response(self, conn: socket.socket, send_lock: threading.Lock, response: Dict[str, Any]) -> None:
        if not self.send_replies:
            return
        payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
        send_packet(conn, payload, send_lock, STOP_EVENT)

    def _start_measurement(self) -> str:
        if self._recording:
            return "[SEG-LAT] already recording | press A to stop"
        self.recorder.begin_measurement()
        self._recording = True
        self._measurement_completed = False
        print("[SEG-LAT] Measurement STARTED by Quest A button.", flush=True)
        return "[SEG-LAT] started | recording forward latency | press A to stop"

    def _stop_measurement(self) -> str:
        if not self._recording:
            if self._measurement_completed:
                return self._summary_text()
            return "[SEG-LAT] not recording | press A to start"

        self._recording = False
        self._measurement_completed = True
        self.recorder.stop_measurement()
        summary = self._summary_text()
        print("[SEG-LAT] Measurement STOPPED by Quest A button.", flush=True)
        print(summary, flush=True)
        print(f"[SEG-LAT] Per-frame CSV: {self.recorder.per_frame_csv}", flush=True)
        print(f"[SEG-LAT] Summary CSV: {self.recorder.summary_csv}", flush=True)
        return summary

    def _handle_control_packet(self, msg: Dict[str, Any], conn: socket.socket, send_lock: threading.Lock) -> None:
        command = str(msg.get("command") or "").strip().lower()

        if command in ("seg_latency_start", "latency_start", "measurement_start", "start"):
            pc_log = self._start_measurement()
        elif command in ("seg_latency_stop", "latency_stop", "measurement_stop", "stop"):
            pc_log = self._stop_measurement()
        elif command in ("seg_latency_toggle", "latency_toggle", "measurement_toggle", "toggle"):
            pc_log = self._stop_measurement() if self._recording else self._start_measurement()
        elif command in ("seg_latency_status", "latency_status", "measurement_status", "status"):
            pc_log = self._status_text()
        else:
            pc_log = f"[SEG-LAT] ignored control command: {command or 'empty'}"
            print(pc_log, flush=True)

        self._send_response(conn, send_lock, self._build_status_response(pc_log))

    def _process_client(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(1.0)
        send_lock = threading.Lock()
        last_print_time = 0.0

        while not STOP_EVENT.is_set() and not self._should_stop_for_limits():
            try:
                packet = recv_packet_with_limits(
                    conn,
                    STOP_EVENT,
                    self._should_stop_for_limits,
                    self.client_idle_timeout,
                )
            except TimeoutError:
                print(
                    f"[SEG-LAT] Closing idle client: no socket data for {self.client_idle_timeout:.1f}s",
                    flush=True,
                )
                break
            except InterruptedError:
                break

            control_msg = parse_control_packet(packet)
            if control_msg is not None:
                self._handle_control_packet(control_msg, conn, send_lock)
                continue

            self._stream_frames_seen += 1
            if not self._recording:
                now = time.time()
                if now - last_print_time >= self.print_interval:
                    last_print_time = now
                    print(self._status_text(), flush=True)
                self._send_response(conn, send_lock, self._build_status_response())
                continue

            frame_bgr = decode_jpg(packet)
            frame_index = self.recorder.frame_count + 1
            rows = self.benchmark.run_frame(frame_bgr, frame_index)
            self.recorder.write_frame(frame_index, frame_bgr, len(packet), rows)

            now = time.time()
            if now - last_print_time >= self.print_interval or self._should_stop_for_limits():
                last_print_time = now
                print(self._format_frame_log(frame_index, rows), flush=True)

            if self.max_frames > 0 and self.recorder.frame_count >= self.max_frames:
                self._stop_measurement()

            self._send_response(conn, send_lock, self._build_frame_response(frame_bgr, rows))

    def serve_forever(self) -> None:
        print(f"[SEG-LAT] Listening on {self.host}:{self.port}", flush=True)
        print("[SEG-LAT] Scope: model forward only. Decode/preprocess/postprocess/network excluded.", flush=True)
        print(f"[SEG-LAT] Results: {self.recorder.run_dir}", flush=True)

        server_conn: Optional[socket.socket] = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
                sock.listen(1)
                sock.settimeout(1.0)

                while not STOP_EVENT.is_set() and not self._should_stop_for_limits():
                    try:
                        conn, addr = sock.accept()
                    except socket.timeout:
                        continue
                    print(f"[SEG-LAT] Client connected: {addr}", flush=True)
                    server_conn = conn
                    try:
                        self._process_client(conn)
                    except (ConnectionError, OSError) as exc:
                        if not STOP_EVENT.is_set():
                            print(f"[SEG-LAT] Client disconnected / error: {exc}", flush=True)
                    finally:
                        safe_close_conn(conn)
                        server_conn = None

        finally:
            safe_close_conn(server_conn)
            self.recorder.close()
            print(f"[SEG-LAT] Stopped. Frames recorded: {self.recorder.frame_count}", flush=True)
            print(f"[SEG-LAT] Per-frame CSV: {self.recorder.per_frame_csv}", flush=True)
            print(f"[SEG-LAT] Summary CSV: {self.recorder.summary_csv}", flush=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but not available: {requested}")
    return device


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Quest stream segmentation forward-only latency benchmark.")
    ap.add_argument("--models-zip", default=None, help="Zip containing the five segmentation .pt models.")
    ap.add_argument("--models-dir", default=None, help="Directory containing segmentation .pt models.")
    ap.add_argument("--output-dir", default="seg_latency_results", help="Directory for CSV/metadata artifacts.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    ap.add_argument("--half", action="store_true", help="Use FP16 model/input on CUDA.")
    ap.add_argument("--warmup", type=int, default=5, help="Forward warmup runs per model before recording.")
    ap.add_argument("--rotate-order", action="store_true", help="Rotate the first model each frame to reduce first-forward idle bias.")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after this many measured Quest frames; 0 means A-button stop only.")
    ap.add_argument("--duration-sec", type=float, default=0.0, help="Stop after this many seconds; 0 disables.")
    ap.add_argument("--client-idle-timeout", type=float, default=30.0)
    ap.add_argument("--print-interval", type=float, default=1.0)
    ap.add_argument("--no-reply", action="store_true", help="Do not send JSON replies back to Quest.")
    ap.add_argument("--exit-after-stop", action="store_true", help="Exit the server after Quest stops a measurement.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    run_id, run_dir = make_run_dir(output_dir)

    models_zip = Path(args.models_zip) if args.models_zip else None
    models_dir = Path(args.models_dir) if args.models_dir else None
    model_paths = discover_model_paths(models_zip=models_zip, models_dir=models_dir, run_dir=run_dir)

    device = select_device(args.device)
    benchmark = SegLatencyBenchmark(
        model_paths=model_paths,
        device=device,
        use_half=args.half,
        warmup_runs=args.warmup,
        rotate_order=args.rotate_order,
    )
    recorder = LatencyRecorder(run_id=run_id, run_dir=run_dir, benchmark=benchmark, args=args)

    server = SegLatencyServer(
        benchmark=benchmark,
        recorder=recorder,
        host=args.host,
        port=args.port,
        max_frames=args.max_frames,
        duration_sec=args.duration_sec,
        client_idle_timeout=args.client_idle_timeout,
        print_interval=args.print_interval,
        send_replies=not args.no_reply,
        exit_after_stop=args.exit_after_stop,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
