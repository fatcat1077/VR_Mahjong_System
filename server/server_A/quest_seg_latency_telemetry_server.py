from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import statistics
import struct
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        print("\n[QUEST-SEG-LAT] Ctrl+C received -> stopping...", flush=True)
    STOP_EVENT.set()


signal.signal(signal.SIGINT, _sigint_handler)


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
    base = datetime.now().strftime("%Y%m%d_%H%M%S_quest_local")
    candidate = output_dir / base
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.name, candidate


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
    if length <= 0 or length > 10_000_000:
        raise ValueError(f"Invalid packet length: {length}")
    return recv_exact(length)


def send_packet(conn: socket.socket, payload: bytes, lock: threading.Lock) -> None:
    packet = struct.pack(">I", len(payload)) + payload
    with lock:
        conn.sendall(packet)


def safe_close_conn(conn: Optional[socket.socket]) -> None:
    if conn is None:
        return
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


def packet_to_json(payload: bytes) -> Optional[Dict[str, Any]]:
    if not payload or payload[:1] not in (b"{", b"["):
        return None
    try:
        msg = json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    return msg if isinstance(msg, dict) else None


def model_sort_key(name: str) -> Tuple[int, str]:
    return PREFERRED_MODEL_ORDER.get(name, 100), name


class QuestSegLatencyRecorder:
    def __init__(self, run_id: str, run_dir: Path, args: argparse.Namespace):
        self.run_id = run_id
        self.run_dir = run_dir
        self.args = args
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.finished_at: Optional[str] = None
        self.measurement_started_at: Optional[str] = None
        self.measurement_stopped_at: Optional[str] = None
        self.measurement_status = "idle"
        self.frame_count = 0
        self.models: Dict[str, Dict[str, Any]] = {}
        self.control_metadata: Dict[str, Any] = {}
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
                "quest_realtime_sec",
                "image_w",
                "image_h",
                "order_offset",
                "rotate_order",
                "model_name",
                "resource",
                "imgsz",
                "parameters",
                "backend",
                "order_index",
                "latency_ms",
                "output0_shape",
                "output1_shape",
                "sync_method",
            ],
        )
        self._writer.writeheader()
        self.write_metadata()

    def begin_measurement(self, control_msg: Dict[str, Any]) -> None:
        self.measurement_started_at = datetime.now().isoformat(timespec="seconds")
        self.measurement_stopped_at = None
        self.measurement_status = "recording"
        self.frame_count = 0
        self.values.clear()
        self.order_indices.clear()
        self.control_metadata = dict(control_msg)
        self.models.clear()
        for model in control_msg.get("models") or []:
            name = str(model.get("model_name") or "")
            if name:
                self.models[name] = dict(model)

        self._fh.seek(0)
        self._fh.truncate(0)
        self._writer.writeheader()
        self._fh.flush()
        self.write_metadata()

    def stop_measurement(self) -> None:
        self.measurement_stopped_at = datetime.now().isoformat(timespec="seconds")
        self.measurement_status = "stopped"
        self.write_summary()
        self.write_metadata()

    def write_frame(self, msg: Dict[str, Any]) -> None:
        frame_index = int(msg.get("frame_index") or (self.frame_count + 1))
        records = msg.get("records") or []
        recv_wall_time = datetime.now().isoformat(timespec="milliseconds")
        sync_method = str(msg.get("sync_method") or "")

        for record in records:
            model_name = str(record.get("model_name") or "")
            latency = float(record.get("latency_ms") or 0.0)
            order_index = int(record.get("order_index") or 0)
            if model_name and model_name not in self.models:
                self.models[model_name] = {
                    "model_name": model_name,
                    "resource": record.get("resource") or "",
                    "imgsz": record.get("imgsz") or 0,
                    "parameters": record.get("parameters") or 0,
                    "backend": record.get("backend") or msg.get("backend") or "",
                }

            self.values[model_name].append(latency)
            self.order_indices[model_name].append(order_index)
            self._writer.writerow(
                {
                    "run_id": self.run_id,
                    "frame_index": frame_index,
                    "recv_wall_time": recv_wall_time,
                    "quest_realtime_sec": msg.get("quest_realtime_sec") or "",
                    "image_w": msg.get("image_w") or "",
                    "image_h": msg.get("image_h") or "",
                    "order_offset": msg.get("order_offset") or 0,
                    "rotate_order": msg.get("rotate_order"),
                    "model_name": model_name,
                    "resource": record.get("resource") or "",
                    "imgsz": record.get("imgsz") or "",
                    "parameters": record.get("parameters") or "",
                    "backend": record.get("backend") or msg.get("backend") or "",
                    "order_index": order_index,
                    "latency_ms": f"{latency:.6f}",
                    "output0_shape": record.get("output0_shape") or "",
                    "output1_shape": record.get("output1_shape") or "",
                    "sync_method": sync_method,
                }
            )

        self.frame_count = max(self.frame_count, frame_index)
        self._fh.flush()

    def summary_rows(self) -> List[Dict[str, Any]]:
        model_names = set(self.models) | set(self.values)
        rows: List[Dict[str, Any]] = []
        for model_name in sorted(model_names, key=model_sort_key):
            meta = self.models.get(model_name, {})
            vals = self.values.get(model_name, [])
            orders = self.order_indices.get(model_name, [])
            if vals:
                avg_ms = statistics.fmean(vals)
                row = {
                    "model_name": model_name,
                    "resource": meta.get("resource", ""),
                    "imgsz": meta.get("imgsz", ""),
                    "parameters": meta.get("parameters", ""),
                    "backend": meta.get("backend", self.control_metadata.get("backend", "")),
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
                    "model_name": model_name,
                    "resource": meta.get("resource", ""),
                    "imgsz": meta.get("imgsz", ""),
                    "parameters": meta.get("parameters", ""),
                    "backend": meta.get("backend", self.control_metadata.get("backend", "")),
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
            "resource",
            "imgsz",
            "parameters",
            "backend",
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
            "measurement_scope": "Quest local Sentis model forward only; texture-to-tensor/preprocess, postprocess, image/network transfer, and UI excluded",
            "control_metadata": self.control_metadata,
            "args": vars(self.args),
            "models": [self.models[name] for name in sorted(self.models, key=model_sort_key)],
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


class QuestSegLatencyTelemetryServer:
    def __init__(
        self,
        recorder: QuestSegLatencyRecorder,
        host: str,
        port: int,
        client_idle_timeout: float,
        print_interval: float,
        exit_after_stop: bool,
    ):
        self.recorder = recorder
        self.host = host
        self.port = int(port)
        self.client_idle_timeout = float(client_idle_timeout)
        self.print_interval = float(print_interval)
        self.exit_after_stop = bool(exit_after_stop)
        self._recording = False
        self._measurement_completed = False
        self._last_print_time = 0.0

    def _should_stop(self) -> bool:
        return self.exit_after_stop and self._measurement_completed

    def _status_text(self) -> str:
        if self._recording:
            return f"[QUEST-SEG-LAT] recording | frames={self.recorder.frame_count} | press A to stop"
        if self._measurement_completed:
            return self._summary_text()
        return "[QUEST-SEG-LAT] idle | press A to start"

    def _summary_text(self) -> str:
        rows = self.recorder.summary_rows()
        if not rows or max((int(row["frames"]) for row in rows), default=0) <= 0:
            return "[QUEST-SEG-LAT] stopped | frames=0"
        parts = [f"{row['model_name']} avg={float(row['avg_ms']):.2f}ms" for row in rows]
        return f"[QUEST-SEG-LAT] stopped | frames={self.recorder.frame_count} | " + " | ".join(parts)

    @staticmethod
    def _build_response(pc_log: str, recording: bool, frames: int) -> Dict[str, Any]:
        return {
            "ts": time.time(),
            "img_w": 1,
            "img_h": 1,
            "tiles": [],
            "hand": [],
            "hand_stable": False,
            "advice": {
                "benefit": {"tile_id": -1, "tile": "", "source": "quest_seg_latency", "reason": "telemetry"},
                "safe": {"tile_id": -1, "tile": "", "source": "quest_seg_latency", "reason": "telemetry"},
            },
            "pc_log": pc_log,
            "debug": {
                "mode": "quest_local_sentis_forward_latency",
                "recording": recording,
                "frames": frames,
            },
        }

    def _send_response(self, conn: socket.socket, send_lock: threading.Lock, pc_log: str) -> None:
        payload = json.dumps(
            self._build_response(pc_log, self._recording, self.recorder.frame_count),
            ensure_ascii=False,
        ).encode("utf-8")
        send_packet(conn, payload, send_lock)

    def _start_measurement(self, msg: Dict[str, Any]) -> str:
        if self._recording:
            return "[QUEST-SEG-LAT] already recording | press A to stop"
        self.recorder.begin_measurement(msg)
        self._recording = True
        self._measurement_completed = False
        print("[QUEST-SEG-LAT] Measurement STARTED by Quest A button.", flush=True)
        print(f"[QUEST-SEG-LAT] rotate_order={msg.get('rotate_order')} backend={msg.get('backend')}", flush=True)
        return "[QUEST-SEG-LAT] started | Quest local Sentis forward latency | press A to stop"

    def _stop_measurement(self) -> str:
        if not self._recording:
            if self._measurement_completed:
                return self._summary_text()
            return "[QUEST-SEG-LAT] not recording | press A to start"

        self._recording = False
        self._measurement_completed = True
        self.recorder.stop_measurement()
        summary = self._summary_text()
        print("[QUEST-SEG-LAT] Measurement STOPPED by Quest A button.", flush=True)
        print(summary, flush=True)
        print(f"[QUEST-SEG-LAT] Per-frame CSV: {self.recorder.per_frame_csv}", flush=True)
        print(f"[QUEST-SEG-LAT] Summary CSV: {self.recorder.summary_csv}", flush=True)
        return summary

    def _handle_control(self, msg: Dict[str, Any], conn: socket.socket, send_lock: threading.Lock) -> None:
        command = str(msg.get("command") or "").strip().lower()
        if command in ("seg_latency_start", "latency_start", "measurement_start", "start"):
            pc_log = self._start_measurement(msg)
        elif command in ("seg_latency_stop", "latency_stop", "measurement_stop", "stop"):
            pc_log = self._stop_measurement()
        elif command in ("seg_latency_status", "latency_status", "measurement_status", "status"):
            pc_log = self._status_text()
        else:
            pc_log = f"[QUEST-SEG-LAT] ignored control command: {command or 'empty'}"
            print(pc_log, flush=True)
        self._send_response(conn, send_lock, pc_log)

    def _handle_frame(self, msg: Dict[str, Any], conn: socket.socket, send_lock: threading.Lock) -> None:
        if not self._recording:
            self._send_response(conn, send_lock, self._status_text())
            return

        self.recorder.write_frame(msg)
        now = time.time()
        rows = msg.get("records") or []
        parts = [
            f"{row.get('model_name')}={float(row.get('latency_ms') or 0.0):.2f}ms@{row.get('order_index')}"
            for row in rows
        ]
        pc_log = f"[QUEST-SEG-LAT] frame={self.recorder.frame_count} | " + " | ".join(parts)
        if now - self._last_print_time >= self.print_interval:
            self._last_print_time = now
            print(pc_log, flush=True)
        self._send_response(conn, send_lock, pc_log)

    def _process_client(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(1.0)
        send_lock = threading.Lock()

        while not STOP_EVENT.is_set() and not self._should_stop():
            try:
                packet = recv_packet_with_limits(conn, STOP_EVENT, self._should_stop, self.client_idle_timeout)
            except TimeoutError:
                print(
                    f"[QUEST-SEG-LAT] Closing idle client: no socket data for {self.client_idle_timeout:.1f}s",
                    flush=True,
                )
                break
            except InterruptedError:
                break

            msg = packet_to_json(packet)
            if msg is None:
                print(f"[QUEST-SEG-LAT] Ignored non-JSON packet bytes={len(packet)}", flush=True)
                self._send_response(conn, send_lock, "[QUEST-SEG-LAT] ignored non-JSON packet")
                continue

            msg_type = str(msg.get("type") or "")
            if msg_type == "control":
                self._handle_control(msg, conn, send_lock)
            elif msg_type == "quest_local_seg_latency_frame":
                self._handle_frame(msg, conn, send_lock)
            else:
                pc_log = f"[QUEST-SEG-LAT] ignored packet type: {msg_type or 'empty'}"
                print(pc_log, flush=True)
                self._send_response(conn, send_lock, pc_log)

    def serve_forever(self) -> None:
        print(f"[QUEST-SEG-LAT] Listening on {self.host}:{self.port}", flush=True)
        print("[QUEST-SEG-LAT] Scope: Quest local Sentis forward telemetry only. PC does not run models.", flush=True)
        print(f"[QUEST-SEG-LAT] Results: {self.recorder.run_dir}", flush=True)

        server_conn: Optional[socket.socket] = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
                sock.listen(1)
                sock.settimeout(1.0)

                while not STOP_EVENT.is_set() and not self._should_stop():
                    try:
                        conn, addr = sock.accept()
                    except socket.timeout:
                        continue
                    print(f"[QUEST-SEG-LAT] Client connected: {addr}", flush=True)
                    server_conn = conn
                    try:
                        self._process_client(conn)
                    except (ConnectionError, OSError) as exc:
                        if not STOP_EVENT.is_set():
                            print(f"[QUEST-SEG-LAT] Client disconnected / error: {exc}", flush=True)
                    finally:
                        safe_close_conn(conn)
                        server_conn = None
        finally:
            safe_close_conn(server_conn)
            self.recorder.close()
            print(f"[QUEST-SEG-LAT] Stopped. Frames recorded: {self.recorder.frame_count}", flush=True)
            print(f"[QUEST-SEG-LAT] Per-frame CSV: {self.recorder.per_frame_csv}", flush=True)
            print(f"[QUEST-SEG-LAT] Summary CSV: {self.recorder.summary_csv}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Quest-local Sentis segmentation latency telemetry receiver.")
    ap.add_argument("--output-dir", default="seg_latency_results", help="Directory for CSV/metadata artifacts.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--client-idle-timeout", type=float, default=60.0)
    ap.add_argument("--print-interval", type=float, default=1.0)
    ap.add_argument("--exit-after-stop", action="store_true", help="Exit after Quest stops a measurement.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run_id, run_dir = make_run_dir(Path(args.output_dir))
    recorder = QuestSegLatencyRecorder(run_id=run_id, run_dir=run_dir, args=args)
    server = QuestSegLatencyTelemetryServer(
        recorder=recorder,
        host=args.host,
        port=args.port,
        client_idle_timeout=args.client_idle_timeout,
        print_interval=args.print_interval,
        exit_after_stop=args.exit_after_stop,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
