from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import struct
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from netio import safe_close_conn, send_packet


STOP_EVENT = threading.Event()


def _sigint_handler(sig, frame):
    if not STOP_EVENT.is_set():
        print("\n[G-SAM2-LAT] Ctrl+C received -> stopping...", flush=True)
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


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ms_since(start: float, end: float) -> float:
    return (end - start) * 1000.0


class GroundedSam2Benchmark:
    def __init__(
        self,
        model_root: Path,
        device: torch.device,
        text_labels: List[str],
        box_threshold: float,
        text_threshold: float,
        amp_dtype: str,
    ):
        self.model_root = model_root
        self.device = device
        self.text_labels = [label.strip() for label in text_labels if label.strip()]
        if not self.text_labels:
            raise ValueError("At least one text label is required.")
        self.text_prompt = [[*self.text_labels]]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.amp_dtype_name = amp_dtype
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)

        self.sam_checkpoint = model_root / "checkpoints" / "sam2.1_hiera_large.pt"
        self.dino_dir = model_root / "grounding-dino-base"
        if not self.sam_checkpoint.exists():
            raise FileNotFoundError(f"Missing SAM2.1 checkpoint: {self.sam_checkpoint}")
        if not self.dino_dir.exists():
            raise FileNotFoundError(f"Missing GroundingDINO snapshot: {self.dino_dir}")

        print(f"[G-SAM2-LAT] Loading GroundingDINO-base from {self.dino_dir}", flush=True)
        self.dino_processor = AutoProcessor.from_pretrained(str(self.dino_dir), local_files_only=True)
        self.dino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(str(self.dino_dir), local_files_only=True)
            .to(self.device)
            .eval()
        )

        print(f"[G-SAM2-LAT] Loading SAM2.1-large from {self.sam_checkpoint}", flush=True)
        sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(self.sam_checkpoint), device=self.device)
        sam_model.eval()
        self.sam_predictor = SAM2ImagePredictor(sam_model)

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str):
        normalized = amp_dtype.lower()
        if normalized in ("none", "fp32", "float32"):
            return None
        if normalized in ("bf16", "bfloat16"):
            return torch.bfloat16
        if normalized in ("fp16", "float16", "half"):
            return torch.float16
        raise ValueError(f"Unsupported amp dtype: {amp_dtype}")

    def autocast_context(self):
        enabled = self.device.type == "cuda" and self.amp_dtype is not None
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=enabled)

    def warmup(self, image_bgr: np.ndarray, runs: int) -> None:
        if runs <= 0:
            return
        print(f"[G-SAM2-LAT] Warmup: {runs} Grounded-SAM2 runs", flush=True)
        for _ in range(runs):
            self.run_frame(image_bgr)

    def run_frame(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        with torch.inference_mode():
            t0 = time.perf_counter()
            dino_inputs = self.dino_processor(images=pil_image, text=self.text_prompt, return_tensors="pt").to(self.device)
            t1 = time.perf_counter()
            with self.autocast_context():
                sync_if_cuda(self.device)
                t2 = time.perf_counter()
                dino_outputs = self.dino_model(**dino_inputs)
                sync_if_cuda(self.device)
                t3 = time.perf_counter()

            results = self.dino_processor.post_process_grounded_object_detection(
                dino_outputs,
                dino_inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]
            t4 = time.perf_counter()

            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            labels = results.get("labels", [])
            if len(boxes):
                best_idx = int(torch.argmax(scores).detach().cpu())
                selected_box = boxes[best_idx].detach().cpu().numpy().astype(np.float32)
                selected_score = float(scores[best_idx].detach().cpu())
                selected_label = str(labels[best_idx])
            else:
                selected_box = np.array([0, 0, w, h], dtype=np.float32)
                selected_score = 0.0
                selected_label = "fallback_full_image"

            with self.autocast_context():
                sync_if_cuda(self.device)
                t5 = time.perf_counter()
                self.sam_predictor.set_image(image_rgb)
                sync_if_cuda(self.device)
                t6 = time.perf_counter()
                masks, mask_scores, _ = self.sam_predictor.predict(box=selected_box, multimask_output=False)
                sync_if_cuda(self.device)
                t7 = time.perf_counter()

        dino_preprocess_ms = ms_since(t0, t1)
        dino_forward_ms = ms_since(t2, t3)
        dino_postprocess_ms = ms_since(t3, t4)
        sam_set_image_ms = ms_since(t5, t6)
        sam_predict_ms = ms_since(t6, t7)
        forward_only_ms = dino_forward_ms + sam_set_image_ms + sam_predict_ms
        compute_total_ms = ms_since(t0, t7)

        return {
            "image_width": w,
            "image_height": h,
            "detections": int(len(boxes)),
            "selected_box_xyxy": json.dumps([float(v) for v in selected_box.tolist()]),
            "selected_score": selected_score,
            "selected_label": selected_label,
            "mask_shape": json.dumps(list(getattr(masks, "shape", []))),
            "mask_score": float(mask_scores[0]) if len(mask_scores) else 0.0,
            "dino_preprocess_ms": dino_preprocess_ms,
            "dino_forward_ms": dino_forward_ms,
            "dino_postprocess_ms": dino_postprocess_ms,
            "sam_set_image_ms": sam_set_image_ms,
            "sam_predict_ms": sam_predict_ms,
            "forward_only_ms": forward_only_ms,
            "compute_total_ms": compute_total_ms,
        }


class GroundedSam2Recorder:
    FIELDNAMES = [
        "frame_index",
        "timestamp_unix",
        "packet_bytes",
        "decode_ms",
        "image_width",
        "image_height",
        "detections",
        "selected_box_xyxy",
        "selected_score",
        "selected_label",
        "mask_shape",
        "mask_score",
        "dino_preprocess_ms",
        "dino_forward_ms",
        "dino_postprocess_ms",
        "sam_set_image_ms",
        "sam_predict_ms",
        "forward_only_ms",
        "compute_total_ms",
    ]

    SUMMARY_FIELDS = [
        "stage",
        "frames",
        "avg_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
    ]

    STAGES = [
        "decode_ms",
        "dino_preprocess_ms",
        "dino_forward_ms",
        "dino_postprocess_ms",
        "sam_set_image_ms",
        "sam_predict_ms",
        "forward_only_ms",
        "compute_total_ms",
    ]

    def __init__(self, run_dir: Path, benchmark: GroundedSam2Benchmark, args: argparse.Namespace):
        self.run_dir = run_dir
        self.benchmark = benchmark
        self.args = args
        self.per_frame_csv = run_dir / "per_frame_latency.csv"
        self.summary_csv = run_dir / "summary_latency.csv"
        self.metadata_json = run_dir / "metadata.json"
        self.rows: List[Dict[str, Any]] = []
        self.frame_count = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.per_frame_csv.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self.write_metadata(measurement_status="initialized")

    def write_metadata(self, measurement_status: str) -> None:
        metadata = {
            "measurement_status": measurement_status,
            "run_dir": str(self.run_dir),
            "scope": (
                "GroundingDINO-base + SAM2.1-large PC GPU inference latency from Quest JPEG stream. "
                "decode and preprocessing are reported separately; forward_only_ms sums dino_forward, "
                "sam_set_image, and sam_predict."
            ),
            "device": str(self.benchmark.device),
            "amp_dtype": self.benchmark.amp_dtype_name,
            "text_labels": self.benchmark.text_labels,
            "box_threshold": self.benchmark.box_threshold,
            "text_threshold": self.benchmark.text_threshold,
            "sam_checkpoint": str(self.benchmark.sam_checkpoint),
            "grounding_dino_dir": str(self.benchmark.dino_dir),
            "args": vars(self.args),
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    def write_frame(self, row: Dict[str, Any]) -> None:
        self.frame_count += 1
        row["frame_index"] = self.frame_count
        row["timestamp_unix"] = time.time()
        self.rows.append(row)
        self._writer.writerow({name: row.get(name, "") for name in self.FIELDNAMES})
        self._csv_file.flush()

    def summary_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for stage in self.STAGES:
            values = [float(row[stage]) for row in self.rows if stage in row]
            if not values:
                continue
            sorted_values = sorted(values)
            if len(sorted_values) == 1:
                p95 = sorted_values[0]
            else:
                p95 = statistics.quantiles(sorted_values, n=20, method="inclusive")[18]
            rows.append(
                {
                    "stage": stage,
                    "frames": len(values),
                    "avg_ms": statistics.fmean(values),
                    "median_ms": statistics.median(values),
                    "p95_ms": p95,
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
            )
        return rows

    def write_summary(self) -> None:
        rows = self.summary_rows()
        with self.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def close(self) -> None:
        self.write_summary()
        self.write_metadata(measurement_status="stopped")
        self._csv_file.close()


class GroundedSam2Server:
    def __init__(
        self,
        recorder: GroundedSam2Recorder,
        host: str,
        port: int,
        max_frames: int,
        duration_sec: float,
        client_idle_timeout: float,
        print_interval: int,
        exit_after_stop: bool,
    ):
        self.recorder = recorder
        self.host = host
        self.port = port
        self.max_frames = max_frames
        self.duration_sec = duration_sec
        self.client_idle_timeout = client_idle_timeout
        self.print_interval = max(1, print_interval)
        self.exit_after_stop = exit_after_stop
        self._recording = False
        self._measurement_start: Optional[float] = None

    def _should_stop_for_limits(self) -> bool:
        if self.max_frames > 0 and self.recorder.frame_count >= self.max_frames:
            return True
        if self.duration_sec > 0 and self._measurement_start is not None:
            return (time.time() - self._measurement_start) >= self.duration_sec
        return False

    def _status_payload(self, pc_log: str = "") -> bytes:
        payload = {
            "type": "seg_latency_status",
            "mode": "grounded_sam2_pc_stream",
            "recording": self._recording,
            "frames": self.recorder.frame_count,
            "pc_log": pc_log or self._format_summary_log(),
        }
        return json.dumps(payload).encode("utf-8")

    def _format_summary_log(self) -> str:
        if self._recording:
            return f"[G-SAM2-LAT] recording | frames={self.recorder.frame_count} | press A to stop"
        if not self.recorder.rows:
            return "[G-SAM2-LAT] idle | press A to start"
        summary = {row["stage"]: row["avg_ms"] for row in self.recorder.summary_rows()}
        return (
            f"[G-SAM2-LAT] stopped | frames={self.recorder.frame_count} | "
            f"dino={summary.get('dino_forward_ms', 0):.1f}ms | "
            f"sam_set_image={summary.get('sam_set_image_ms', 0):.1f}ms | "
            f"sam_predict={summary.get('sam_predict_ms', 0):.1f}ms | "
            f"forward_only={summary.get('forward_only_ms', 0):.1f}ms"
        )

    def _start_measurement(self) -> str:
        if self._recording:
            return "[G-SAM2-LAT] already recording | press A to stop"
        self._recording = True
        self._measurement_start = time.time()
        self.recorder.write_metadata(measurement_status="recording")
        print("[G-SAM2-LAT] Measurement STARTED by Quest A button.", flush=True)
        return "[G-SAM2-LAT] started | recording Grounded-SAM2 latency | press A to stop"

    def _stop_measurement(self) -> str:
        if not self._recording:
            return "[G-SAM2-LAT] not recording | press A to start"
        self._recording = False
        self.recorder.write_summary()
        self.recorder.write_metadata(measurement_status="stopped")
        print("[G-SAM2-LAT] Measurement STOPPED by Quest A button.", flush=True)
        print(self._format_summary_log(), flush=True)
        print(f"[G-SAM2-LAT] Per-frame CSV: {self.recorder.per_frame_csv}", flush=True)
        print(f"[G-SAM2-LAT] Summary CSV: {self.recorder.summary_csv}", flush=True)
        if self.exit_after_stop:
            STOP_EVENT.set()
        return self._format_summary_log()

    def _handle_control(self, msg: Dict[str, Any]) -> str:
        command = str(msg.get("command", "")).lower()
        if command == "seg_latency_start":
            return self._start_measurement()
        if command == "seg_latency_stop":
            return self._stop_measurement()
        if command == "seg_latency_toggle":
            return self._stop_measurement() if self._recording else self._start_measurement()
        return f"[G-SAM2-LAT] unknown control command: {command}"

    def _serve_client(self, conn: socket.socket, addr, send_lock: threading.Lock) -> None:
        conn.settimeout(0.5)
        send_packet(conn, self._status_payload(), send_lock, STOP_EVENT)
        while not STOP_EVENT.is_set() and not self._should_stop_for_limits():
            payload = recv_packet_with_limits(
                conn,
                STOP_EVENT,
                self._should_stop_for_limits,
                self.client_idle_timeout,
            )
            control = parse_control_packet(payload)
            if control is not None:
                pc_log = self._handle_control(control)
                send_packet(conn, self._status_payload(pc_log), send_lock, STOP_EVENT)
                continue

            if not self._recording:
                send_packet(conn, self._status_payload(), send_lock, STOP_EVENT)
                continue

            decode_start = time.perf_counter()
            frame_bgr = decode_jpg(payload)
            decode_ms = ms_since(decode_start, time.perf_counter())
            row = self.recorder.benchmark.run_frame(frame_bgr)
            row["packet_bytes"] = len(payload)
            row["decode_ms"] = decode_ms
            self.recorder.write_frame(row)

            if self.recorder.frame_count == 1 or self.recorder.frame_count % self.print_interval == 0:
                print(
                    "[G-SAM2-LAT] "
                    f"frame={self.recorder.frame_count} | "
                    f"dino={row['dino_forward_ms']:.1f}ms | "
                    f"sam_set={row['sam_set_image_ms']:.1f}ms | "
                    f"sam_predict={row['sam_predict_ms']:.1f}ms | "
                    f"forward={row['forward_only_ms']:.1f}ms | "
                    f"det={row['detections']}",
                    flush=True,
                )

            send_packet(conn, self._status_payload(), send_lock, STOP_EVENT)

        if self._recording and self._should_stop_for_limits():
            self._stop_measurement()

    def serve_forever(self) -> None:
        send_lock = threading.Lock()
        print(f"[G-SAM2-LAT] Listening on {self.host}:{self.port}", flush=True)
        print("[G-SAM2-LAT] Scope: PC GPU GroundingDINO + SAM2.1-large inference latency.", flush=True)
        print(f"[G-SAM2-LAT] Results: {self.recorder.run_dir}", flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)
            while not STOP_EVENT.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                print(f"[G-SAM2-LAT] Client connected: {addr}", flush=True)
                try:
                    self._serve_client(conn, addr, send_lock)
                except Exception as exc:
                    if not STOP_EVENT.is_set():
                        print(f"[G-SAM2-LAT] Client disconnected / error: {exc}", flush=True)
                finally:
                    safe_close_conn(conn)


def make_run_dir(output_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_grounded_sam2")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Quest -> PC GroundingDINO + SAM2.1-large latency server.")
    parser.add_argument("--output-dir", type=Path, default=Path("seg_latency_results"))
    parser.add_argument("--model-root", type=Path, default=Path("models/GroundedSAM2"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=0)
    parser.add_argument("--client-idle-timeout", type=float, default=90)
    parser.add_argument("--print-interval", type=int, default=1)
    parser.add_argument("--text-labels", default="mahjong tile,table")
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32", "none"])
    parser.add_argument("--exit-after-stop", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    model_root = args.model_root
    if not model_root.is_absolute():
        model_root = Path.cwd() / model_root
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    benchmark = GroundedSam2Benchmark(
        model_root=model_root,
        device=device,
        text_labels=args.text_labels.split(","),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        amp_dtype=args.amp_dtype,
    )

    if args.warmup > 0:
        sample_path = Path.cwd() / "sample.png"
        if sample_path.exists():
            sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        else:
            sample = np.zeros((720, 1280, 3), dtype=np.uint8)
        benchmark.warmup(sample, args.warmup)

    run_dir = make_run_dir(output_dir)
    recorder = GroundedSam2Recorder(run_dir=run_dir, benchmark=benchmark, args=args)
    server = GroundedSam2Server(
        recorder=recorder,
        host=args.host,
        port=args.port,
        max_frames=args.max_frames,
        duration_sec=args.duration_sec,
        client_idle_timeout=args.client_idle_timeout,
        print_interval=args.print_interval,
        exit_after_stop=args.exit_after_stop,
    )
    try:
        server.serve_forever()
    finally:
        recorder.close()
        print(f"[G-SAM2-LAT] Stopped. Frames recorded: {recorder.frame_count}", flush=True)
        print(f"[G-SAM2-LAT] Per-frame CSV: {recorder.per_frame_csv}", flush=True)
        print(f"[G-SAM2-LAT] Summary CSV: {recorder.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
