import argparse
import csv
import json
import socket
import statistics
import threading
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from netio import recv_packet, safe_close_conn, send_packet
from tracking import Tracker
from vision import VisionPipeline, decode_jpg
import mahjong


# ============================================================
# Global stop flag (Ctrl+C / q to quit)
# ============================================================
STOP_EVENT = threading.Event()


def _sigint_handler(sig, frame):
    # NOTE: signal handler runs in main thread
    if not STOP_EVENT.is_set():
        print("\n[PC] Ctrl+C received -> stopping...")
    STOP_EVENT.set()


signal.signal(signal.SIGINT, _sigint_handler)


def _percentile(values: List[float], p: float) -> float:
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


def _make_run_dir(output_dir: Path) -> tuple[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S_full_pipeline")
    candidate = output_dir / base
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.name, candidate


class FullPipelineLatencyRecorder:
    MS_FIELDS = [
        "pc_frame_age_ms",
        "decode_ms",
        "segmentation_ms",
        "det_postprocess_ms",
        "crop_ms",
        "classification_preprocess_ms",
        "classification_forward_ms",
        "classification_postprocess_ms",
        "classification_total_ms",
        "valid_filter_ms",
        "vision_total_ms",
        "hand_parse_ms",
        "tracking_ms",
        "tile_build_ms",
        "agent_ms",
        "ppo_live_update_ms",
        "ppo_action_mask_ms",
        "ppo_obs_build_ms",
        "ppo_predict_ms",
        "ppo_topk_ms",
        "ppo_total_ms",
        "json_serialize_ms",
        "send_ms",
        "total_pc_pipeline_ms",
    ]

    FIELDNAMES = [
        "run_id",
        "frame_index",
        "recv_wall_time",
        "image_w",
        "image_h",
        "jpg_bytes",
        "recv_packets",
        "pc_frame_age_ms",
        "raw_detection_count",
        "valid_detection_count",
        "classified_crop_count",
        "track_count",
        "hand_count",
        "live_hand_count",
        "table_count",
        "hand_id_count",
        "classification_backend",
        "batch_classification",
        "ppo_called",
        "ppo_synthetic_hand",
        "ppo_detected_hand_count",
        "ppo_synthetic_hand_count",
        "ppo_filled_tile_count",
        "ppo_filled_tiles",
        "ppo_decision_type",
        "ppo_source",
        "ppo_action",
        "ppo_tile",
        *MS_FIELDS[1:],
    ]

    def __init__(self, output_dir: Path, metadata: Dict[str, Any]):
        self.output_dir = output_dir
        self.metadata = dict(metadata)
        self._lock = threading.Lock()
        self.run_id = ""
        self.run_dir: Optional[Path] = None
        self.per_frame_csv: Optional[Path] = None
        self.summary_csv: Optional[Path] = None
        self.metadata_json: Optional[Path] = None
        self.measurement_started_at: Optional[str] = None
        self.measurement_stopped_at: Optional[str] = None
        self.measurement_status = "idle"
        self.control_start: Dict[str, Any] = {}
        self.control_stop: Dict[str, Any] = {}
        self.frame_count = 0
        self.ppo_called_frames = 0
        self.ppo_synthetic_frames = 0
        self._values: Dict[str, List[float]] = {key: [] for key in self.MS_FIELDS}
        self._classified_crop_counts: List[int] = []
        self._fh = None
        self._writer: Optional[csv.DictWriter] = None

    def is_recording(self) -> bool:
        with self._lock:
            return self.measurement_status == "recording" and self._writer is not None

    def begin(self, control_msg: Dict[str, Any]) -> Path:
        with self._lock:
            self._close_open_file_locked()
            self.run_id, self.run_dir = _make_run_dir(self.output_dir)
            self.per_frame_csv = self.run_dir / "per_frame_latency.csv"
            self.summary_csv = self.run_dir / "summary_latency.csv"
            self.metadata_json = self.run_dir / "metadata.json"
            self.measurement_started_at = datetime.now().isoformat(timespec="seconds")
            self.measurement_stopped_at = None
            self.measurement_status = "recording"
            self.control_start = dict(control_msg)
            self.control_stop = {}
            self.frame_count = 0
            self.ppo_called_frames = 0
            self.ppo_synthetic_frames = 0
            self._values = {key: [] for key in self.MS_FIELDS}
            self._classified_crop_counts = []

            self._fh = self.per_frame_csv.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()
            self._fh.flush()
            self._write_metadata_locked()
            return self.run_dir

    def stop(self, control_msg: Dict[str, Any]) -> Optional[Path]:
        with self._lock:
            if self.measurement_status != "recording":
                return self.run_dir
            self.measurement_stopped_at = datetime.now().isoformat(timespec="seconds")
            self.measurement_status = "stopped"
            self.control_stop = dict(control_msg)
            self._write_summary_locked()
            self._write_metadata_locked()
            self._close_open_file_locked()
            return self.run_dir

    def close(self) -> None:
        with self._lock:
            if self.measurement_status == "recording":
                self.measurement_stopped_at = datetime.now().isoformat(timespec="seconds")
                self.measurement_status = "closed"
                self._write_summary_locked()
                self._write_metadata_locked()
            self._close_open_file_locked()

    def record_frame(self, row: Dict[str, Any]) -> bool:
        with self._lock:
            if self.measurement_status != "recording" or self._writer is None or self._fh is None:
                return False

            self.frame_count += 1
            out = {key: "" for key in self.FIELDNAMES}
            out.update(row)
            out["run_id"] = self.run_id
            out["frame_index"] = self.frame_count
            out["recv_wall_time"] = datetime.now().isoformat(timespec="milliseconds")

            for key in self.MS_FIELDS:
                value = out.get(key)
                if value is None or value == "":
                    out[key] = ""
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    out[key] = ""
                    continue
                out[key] = f"{numeric:.6f}"
                self._values.setdefault(key, []).append(numeric)

            try:
                classified_count = int(out.get("classified_crop_count") or 0)
                self._classified_crop_counts.append(classified_count)
            except Exception:
                pass
            if bool(out.get("ppo_called", False)):
                self.ppo_called_frames += 1
            if bool(out.get("ppo_synthetic_hand", False)):
                self.ppo_synthetic_frames += 1

            self._writer.writerow(out)
            self._fh.flush()
            self._write_metadata_locked()
            return True

    def _summary_rows_locked(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for metric in self.MS_FIELDS:
            vals = self._values.get(metric, [])
            if vals:
                avg_ms = statistics.fmean(vals)
                rows.append(
                    {
                        "metric": metric,
                        "frames": len(vals),
                        "avg_ms": avg_ms,
                        "median_ms": statistics.median(vals),
                        "p90_ms": _percentile(vals, 90),
                        "p95_ms": _percentile(vals, 95),
                        "p99_ms": _percentile(vals, 99),
                        "min_ms": min(vals),
                        "max_ms": max(vals),
                        "std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                        "fps_from_avg": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
                    }
                )
            else:
                rows.append(
                    {
                        "metric": metric,
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
                    }
                )
        return rows

    def _write_summary_locked(self) -> None:
        if self.summary_csv is None:
            return
        fieldnames = [
            "metric",
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
        ]
        with self.summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._summary_rows_locked():
                formatted = dict(row)
                for key in fieldnames:
                    if key.endswith("_ms") or key == "fps_from_avg":
                        formatted[key] = f"{float(formatted[key]):.6f}"
                writer.writerow(formatted)

    def _write_metadata_locked(self) -> None:
        if self.metadata_json is None:
            return
        avg_classified = statistics.fmean(self._classified_crop_counts) if self._classified_crop_counts else 0.0
        data = {
            "run_id": self.run_id,
            "measurement_status": self.measurement_status,
            "measurement_started_at": self.measurement_started_at,
            "measurement_stopped_at": self.measurement_stopped_at,
            "frame_count": self.frame_count,
            "ppo_called_frames": self.ppo_called_frames,
            "ppo_skipped_frames": max(0, self.frame_count - self.ppo_called_frames),
            "ppo_synthetic_frames": self.ppo_synthetic_frames,
            "avg_classified_crops_per_frame": avg_classified,
            "measurement_scope": (
                "PC-side realtime pipeline after JPEG frame is available to the PC server: "
                "JPEG decode, segmentation predict, crop/mask processing, batch classification, "
                "tracking, Mahjong state/PPO, JSON serialization, and socket send. Quest capture, "
                "JPEG encode, and network transfer before PC receive are excluded."
            ),
            "control_start": self.control_start,
            "control_stop": self.control_stop,
            "server_metadata": self.metadata,
            "artifacts": {
                "per_frame_csv": str(self.per_frame_csv) if self.per_frame_csv else "",
                "summary_csv": str(self.summary_csv) if self.summary_csv else "",
                "metadata_json": str(self.metadata_json) if self.metadata_json else "",
            },
        }
        self.metadata_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _close_open_file_locked(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None


class InferServer:
    def __init__(
        self,
        yolo_path: str,
        cls_path: str,
        host: str,
        port: int,
        det_imgsz: int,
        det_conf: float,
        det_iou: float,
        cls_imgsz: int,
        crop_pad: float,
        track_iou: float,
        track_ttl: float,
        smooth_len: int,
        view: bool,
        device: Optional[str],
        cls_labels_path: Optional[str] = None,
        cls_nc: Optional[int] = None,
        cls_conf_threshold: float = 0.5,
        ppo_model_path: Optional[str] = None,
        ppo_device: Optional[str] = None,
        ppo_fill_invalid_hand: bool = True,
        ppo_fill_target_count: int = 14,
        client_idle_timeout: float = 5.0,
        debug_vision_dir: Optional[str] = None,
        debug_vision_interval: float = 1.0,
        latency_output_dir: Optional[str] = None,
        latency_exit_after_stop: bool = False,
        latency_stale_timeout: float = 5.0,
    ):
        self.host = host
        self.port = port

        self.view = view
        self.device = device
        self.client_idle_timeout = float(client_idle_timeout)
        self.latency_exit_after_stop = bool(latency_exit_after_stop)
        self.latency_stale_timeout = max(0.5, float(latency_stale_timeout))
        self.ppo_fill_invalid_hand = bool(ppo_fill_invalid_hand)
        self.ppo_fill_target_count = int(ppo_fill_target_count)

        # Models / pipeline
        self.vision = VisionPipeline(
            yolo_path=yolo_path,
            cls_path=cls_path,
            det_imgsz=det_imgsz,
            det_conf=det_conf,
            det_iou=det_iou,
            cls_imgsz=cls_imgsz,
            crop_pad=crop_pad,
            device=device,
            cls_labels_path=cls_labels_path,
            cls_nc=cls_nc,
            cls_conf_threshold=cls_conf_threshold,
            debug_dir=debug_vision_dir,
            debug_interval_sec=debug_vision_interval,
        )

        # Tracking
        self.tracker = Tracker(track_iou=track_iou, track_ttl=track_ttl, smooth_len=smooth_len)

        # Mahjong PPO model (optional)
        self.ppo_model = mahjong.load_ppo_model(ppo_model_path, ppo_device=ppo_device)
        self.game_tracker = mahjong.GameStateTracker()
        self.latency_recorder = FullPipelineLatencyRecorder(
            output_dir=Path(latency_output_dir or "seg_latency_results"),
            metadata={
                "mode": "pc_stream_full_pipeline_latency",
                "yolo_path": yolo_path,
                "cls_path": cls_path,
                "ppo_model_path": ppo_model_path or "",
                "ppo_device": ppo_device or "",
                "ppo_fill_invalid_hand_during_latency": self.ppo_fill_invalid_hand,
                "ppo_fill_target_count": self.ppo_fill_target_count,
                "device": device or "",
                "det_imgsz": det_imgsz,
                "det_conf": det_conf,
                "det_iou": det_iou,
                "cls_imgsz": cls_imgsz,
                "cls_conf_threshold": cls_conf_threshold,
                "crop_pad": crop_pad,
                "track_iou": track_iou,
                "track_ttl": track_ttl,
                "smooth_len": smooth_len,
                "latency_stale_timeout": self.latency_stale_timeout,
            },
        )

        # Shared (latest frame)
        self._latest_jpg: Optional[bytes] = None
        self._latest_ts = 0.0
        self._recv_packets = 0
        self._recv_bytes = 0
        self._receiver_error = ""
        self._stable_hand_active = False
        self._stable_hand_labels: List[str] = []
        self._stable_hand_tiles: List[int] = []
        self._stable_recommended_discard_id: Optional[int] = None
        self._latest_hand_labels: List[str] = []
        self._latest_hand_tiles: List[int] = []
        self._live_hand_candidate_signature: Optional[tuple[int, ...]] = None
        self._live_hand_candidate_frames = 0
        self._scene_reset_requested = False
        self._lock = threading.Lock()

    def _handle_control_packet(self, payload: bytes) -> bool:
        if not payload or payload[:1] not in (b"{", b"["):
            return False

        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception:
            return False

        if not isinstance(msg, dict) or msg.get("type") != "control":
            return False

        command = str(msg.get("command") or "")
        mode = str(msg.get("mode") or "")
        if command in ("scene_reset", "reset_scene", "reset"):
            with self._lock:
                self._scene_reset_requested = True
            print("[Control] scene reset requested", flush=True)
            return True
        if command in ("seg_latency_start", "pipeline_latency_start", "latency_start"):
            if mode == "quest_local_sentis_forward_latency":
                print(
                    "[Latency] Ignoring Quest-local latency start on PC full-pipeline server. "
                    "Launch Quest with seg_latency_mode=pc_stream.",
                    flush=True,
                )
                return True
            run_dir = self.latency_recorder.begin(msg)
            print(f"[Latency] START full pipeline recording -> {run_dir}", flush=True)
            return True
        if command in ("seg_latency_stop", "pipeline_latency_stop", "latency_stop"):
            if mode == "quest_local_sentis_forward_latency":
                print(
                    "[Latency] Ignoring Quest-local latency stop on PC full-pipeline server. "
                    "No PC frames were recorded.",
                    flush=True,
                )
                return True
            run_dir = self.latency_recorder.stop(msg)
            if run_dir is not None:
                print(f"[Latency] STOP full pipeline recording -> {run_dir}", flush=True)
                print(f"[Latency] summary -> {run_dir / 'summary_latency.csv'}", flush=True)
            if self.latency_exit_after_stop:
                STOP_EVENT.set()
            return True

        return True

    def _auto_stop_latency_recording(self, reason: str) -> None:
        if not self.latency_recorder.is_recording():
            return

        stop_msg = {
            "type": "control",
            "command": "client_disconnect_auto_stop",
            "mode": "pc_stream",
            "reason": reason,
        }
        run_dir = self.latency_recorder.stop(stop_msg)
        if run_dir is not None:
            print(f"[Latency] AUTO STOP full pipeline recording ({reason}) -> {run_dir}", flush=True)
            print(f"[Latency] summary -> {run_dir / 'summary_latency.csv'}", flush=True)

    def _set_latest_and_get_effective_hand(
        self,
        live_hand_labels: List[str],
        live_hand_tiles: List[int],
    ) -> tuple[List[str], List[int], bool, str]:
        with self._lock:
            self._latest_hand_labels = list(live_hand_labels)
            self._latest_hand_tiles = list(live_hand_tiles)
            if self._stable_hand_active:
                return list(self._stable_hand_labels), list(self._stable_hand_tiles), True, "auto"
            return list(live_hand_labels), list(live_hand_tiles), False, ""

    @staticmethod
    def _tile_count(tiles: List[int], tile_id: int) -> int:
        return sum(1 for tile in tiles if int(tile) == int(tile_id))

    @staticmethod
    def _canonical_labels_from_tiles(tiles: List[int]) -> List[str]:
        return mahjong.sorted_tile_labels_from_ids(tiles)

    def _clear_live_hand_candidate_locked(self) -> None:
        self._live_hand_candidate_signature = None
        self._live_hand_candidate_frames = 0

    def _remove_one_tile_from_stable_hand_locked(self, tile_id: int) -> tuple[List[str], List[int]]:
        removed = False
        new_tiles: List[int] = []
        for tile in self._stable_hand_tiles:
            if not removed and int(tile) == int(tile_id):
                removed = True
                continue
            new_tiles.append(int(tile))

        if not removed:
            return list(self._stable_hand_labels), list(self._stable_hand_tiles)

        return self._canonical_labels_from_tiles(new_tiles), new_tiles

    def _mark_hand_stable(self, labels: List[str], tiles: List[int], recommended_discard_id: Optional[int] = None) -> bool:
        if not labels or not tiles:
            return False
        with self._lock:
            if self._stable_hand_active:
                return False
            self._stable_hand_active = True
            self._stable_hand_labels = list(labels)
            self._stable_hand_tiles = list(tiles)
            self._stable_recommended_discard_id = recommended_discard_id
            self._clear_live_hand_candidate_locked()
        return True

    def _set_stable_recommended_discard(self, recommended_discard_id: Optional[int]) -> bool:
        if recommended_discard_id is None or not (0 <= int(recommended_discard_id) < 34):
            return False
        with self._lock:
            if not self._stable_hand_active:
                return False
            if self._tile_count(self._stable_hand_tiles, int(recommended_discard_id)) <= 0:
                return False
            self._stable_recommended_discard_id = int(recommended_discard_id)
        return True

    def _release_stable_hand(self) -> bool:
        with self._lock:
            if not self._stable_hand_active:
                return False
            self._stable_hand_active = False
            self._stable_hand_labels = []
            self._stable_hand_tiles = []
            self._stable_recommended_discard_id = None
            self._clear_live_hand_candidate_locked()
        return True

    def _try_complete_self_discard_from_live(
        self,
        live_hand_labels: List[str],
        live_hand_tiles: List[int],
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._stable_hand_active or self._stable_recommended_discard_id is None:
                return None

            discard_id = int(self._stable_recommended_discard_id)
            stable_tiles = list(self._stable_hand_tiles)
            stable_count = len(stable_tiles)
            live_tiles = [int(tile) for tile in live_hand_tiles if 0 <= int(tile) < 34]
            live_count = len(live_tiles)

            if stable_count <= 0 or stable_count % 3 != 2:
                return None
            if live_count != stable_count - 1 or live_count % 3 != 1:
                return None
            if self._tile_count(live_tiles, discard_id) >= self._tile_count(stable_tiles, discard_id):
                return None

            self._stable_hand_active = False
            self._stable_hand_labels = []
            self._stable_hand_tiles = []
            self._stable_recommended_discard_id = None
            self._clear_live_hand_candidate_locked()

        return {
            "type": "stable_hand_self_discard_detected",
            "tile_id": discard_id,
            "tile": mahjong.tile_id_to_label(discard_id),
            "stable_count_before": stable_count,
            "live_count_after": live_count,
            "hand_stable_released": True,
        }

    def _maybe_promote_live_draw_to_stable(
        self,
        live_hand_labels: List[str],
        live_hand_tiles: List[int],
    ) -> Optional[Dict[str, Any]]:
        live_tiles = [int(tile) for tile in live_hand_tiles if 0 <= int(tile) < 34]
        live_signature = tuple(sorted(live_tiles))

        with self._lock:
            if not self._stable_hand_active:
                self._clear_live_hand_candidate_locked()
                return None

            stable_count = len(self._stable_hand_tiles)
            live_count = len(live_tiles)
            if stable_count <= 0 or stable_count % 3 != 1 or live_count != stable_count + 1 or live_count % 3 != 2:
                self._clear_live_hand_candidate_locked()
                return None

            if live_signature == self._live_hand_candidate_signature:
                self._live_hand_candidate_frames += 1
            else:
                self._live_hand_candidate_signature = live_signature
                self._live_hand_candidate_frames = 1

            if self._live_hand_candidate_frames < mahjong.TURN_STABLE_FRAMES:
                return None

            self._stable_hand_labels = self._canonical_labels_from_tiles(live_tiles)
            self._stable_hand_tiles = live_tiles
            self._stable_recommended_discard_id = None
            promoted_frames = self._live_hand_candidate_frames
            self._clear_live_hand_candidate_locked()

        return {
            "type": "stable_hand_promoted_to_self_turn",
            "count": len(live_tiles),
            "stable_frames": promoted_frames,
        }

    def _consume_scene_reset_request(self) -> bool:
        with self._lock:
            if not self._scene_reset_requested:
                return False
            self._scene_reset_requested = False
            self._stable_hand_active = False
            self._stable_hand_labels = []
            self._stable_hand_tiles = []
            self._stable_recommended_discard_id = None
            self._latest_hand_labels = []
            self._latest_hand_tiles = []
            self._clear_live_hand_candidate_locked()

        self.tracker.reset()
        self.game_tracker.reset_runtime()
        return True

    def _receiver_loop(self, conn: socket.socket):
        try:
            while not STOP_EVENT.is_set():
                try:
                    packet = recv_packet(conn, STOP_EVENT)
                except socket.timeout:
                    continue
                except InterruptedError:
                    break

                if self._handle_control_packet(packet):
                    continue

                now = time.time()
                with self._lock:
                    self._latest_jpg = packet
                    self._latest_ts = now
                    self._recv_packets += 1
                    self._recv_bytes += len(packet)
                    self._receiver_error = ""
                if self._recv_packets <= 3:
                    print(f"[Receiver] packet #{self._recv_packets}: {len(packet)} bytes", flush=True)
        except Exception as e:
            with self._lock:
                self._receiver_error = str(e)
            if not STOP_EVENT.is_set():
                print(f"[Receiver] stopped: {e}")

    def _get_latest(self):
        with self._lock:
            return self._latest_jpg, self._latest_ts, self._recv_packets, self._recv_bytes, self._receiver_error

    @staticmethod
    def _turn_label(player: Any) -> str:
        try:
            player_i = int(player)
        except Exception:
            player_i = 0
        return {
            0: "自己",
            1: "下家",
            2: "對家",
            3: "上家",
        }.get(player_i % 4, str(player_i))

    def _last_discard_text(self, agent: Dict[str, Any]) -> str:
        tile = str(agent.get("last_discard_tile") or "")
        if not tile:
            return ""
        try:
            discarder = int(agent.get("last_discarder", -1))
        except Exception:
            discarder = -1
        if discarder < 0:
            return tile
        return f"{self._turn_label(discarder)} {tile}"

    @staticmethod
    def _agent_action_id(advice: Dict[str, Any], agent: Dict[str, Any]) -> int:
        benefit = advice.get("benefit", {}) if isinstance(advice, dict) else {}
        raw_action = benefit.get("tile_id", None)
        if raw_action is None:
            raw_action = agent.get("recommended_action", -1)
        try:
            return int(raw_action)
        except Exception:
            return -1

    def _format_pc_log(self, out: Dict[str, Any]) -> str:
        hand = out.get("hand", [])
        table = out.get("table", [])
        agent = out.get("agent", {})
        advice = out.get("advice", {})
        benefit = advice.get("benefit", {}) if isinstance(advice, dict) else {}

        hand_str = " ".join(str(x) for x in hand) if isinstance(hand, list) else str(hand or "")
        table_str = " ".join(str(x) for x in table) if isinstance(table, list) else str(table or "")
        suggested_tile = str(benefit.get("tile") or agent.get("recommended_tile") or "")
        suggested_action = self._agent_action_id(advice, agent)

        if suggested_action < 0 or not suggested_tile:
            suggested_tile = f"正在偵測{self._turn_label(agent.get('current_player', 0))}出牌"

        if suggested_action >= 0 and suggested_tile and agent.get("decision_type") == "discard":
            action_text = f"打 {suggested_tile}"
        else:
            action_text = suggested_tile
        lines = [
            f"手牌：{hand_str if hand_str else '辨識中'}",
            f"桌上牌：{table_str if table_str else '辨識中'}",
            f"建議動作：{action_text}",
            f"Turn: {self._turn_label(agent.get('current_player', 0))}",
        ]
        try:
            current_player = int(agent.get("current_player", 0)) % 4
        except Exception:
            current_player = 0
        last_discard_text = self._last_discard_text(agent)
        if last_discard_text and current_player != 0:
            lines.append(f"出牌：{last_discard_text}")
        if bool(out.get("hand_stable", False)):
            lines.append("Stable: hand")
        return "\n".join(lines)

    @staticmethod
    def _optional_ms(value: Any) -> Any:
        if value is None:
            return ""
        return value

    def _build_latency_row(
        self,
        out: Dict[str, Any],
        jpg_bytes: int,
        recv_packets: int,
        decode_ms: float,
        json_serialize_ms: float,
        send_ms: float,
        total_pc_pipeline_ms: float,
        pc_frame_age_ms: float,
    ) -> Dict[str, Any]:
        debug = out.get("debug", {}) if isinstance(out.get("debug"), dict) else {}
        agent = out.get("agent", {}) if isinstance(out.get("agent"), dict) else {}
        synthetic_filled_tiles = agent.get("synthetic_filled_tiles", [])
        if isinstance(synthetic_filled_tiles, (list, tuple)):
            synthetic_filled_tiles_str = " ".join(str(tile) for tile in synthetic_filled_tiles)
            synthetic_filled_tile_count = len(synthetic_filled_tiles)
        else:
            synthetic_filled_tiles_str = str(synthetic_filled_tiles or "")
            synthetic_filled_tile_count = 0
        return {
            "image_w": out.get("img_w", 0),
            "image_h": out.get("img_h", 0),
            "jpg_bytes": jpg_bytes,
            "recv_packets": recv_packets,
            "pc_frame_age_ms": pc_frame_age_ms,
            "raw_detection_count": debug.get("raw_detection_count", debug.get("det_count", 0)),
            "valid_detection_count": debug.get("valid_detection_count", debug.get("det_count", 0)),
            "classified_crop_count": debug.get("classified_crop_count", debug.get("det_count", 0)),
            "track_count": debug.get("track_count", 0),
            "hand_count": debug.get("hand_count", 0),
            "live_hand_count": debug.get("live_hand_count", 0),
            "table_count": debug.get("table_count", 0),
            "hand_id_count": debug.get("hand_id_count", 0),
            "classification_backend": debug.get("classification_backend", ""),
            "batch_classification": debug.get("batch_classification", True),
            "ppo_called": bool(debug.get("ppo_called", False)),
            "ppo_synthetic_hand": bool(agent.get("synthetic_hand_for_ppo", False)),
            "ppo_detected_hand_count": agent.get("detected_hand_count", debug.get("hand_id_count", 0)),
            "ppo_synthetic_hand_count": agent.get("hand_count", ""),
            "ppo_filled_tile_count": synthetic_filled_tile_count,
            "ppo_filled_tiles": synthetic_filled_tiles_str,
            "ppo_decision_type": debug.get("ppo_decision_type", agent.get("decision_type", "")),
            "ppo_source": debug.get("ppo_source", agent.get("source", "")),
            "ppo_action": agent.get("recommended_action", -1),
            "ppo_tile": agent.get("recommended_tile", ""),
            "decode_ms": decode_ms,
            "segmentation_ms": debug.get("segmentation_ms", ""),
            "det_postprocess_ms": debug.get("det_postprocess_ms", ""),
            "crop_ms": debug.get("crop_ms", ""),
            "classification_preprocess_ms": debug.get("classification_preprocess_ms", ""),
            "classification_forward_ms": debug.get("classification_forward_ms", ""),
            "classification_postprocess_ms": debug.get("classification_postprocess_ms", ""),
            "classification_total_ms": debug.get("classification_total_ms", ""),
            "valid_filter_ms": debug.get("valid_filter_ms", ""),
            "vision_total_ms": debug.get("vision_total_ms", debug.get("vision_ms", "")),
            "hand_parse_ms": debug.get("hand_parse_ms", ""),
            "tracking_ms": debug.get("tracking_ms", ""),
            "tile_build_ms": debug.get("tile_build_ms", ""),
            "agent_ms": debug.get("agent_ms", ""),
            "ppo_live_update_ms": debug.get("ppo_live_update_ms", ""),
            "ppo_action_mask_ms": debug.get("ppo_action_mask_ms", ""),
            "ppo_obs_build_ms": self._optional_ms(debug.get("ppo_obs_build_ms", "")),
            "ppo_predict_ms": self._optional_ms(debug.get("ppo_predict_ms", "")),
            "ppo_topk_ms": self._optional_ms(debug.get("ppo_topk_ms", "")),
            "ppo_total_ms": debug.get("ppo_total_ms", ""),
            "json_serialize_ms": json_serialize_ms,
            "send_ms": send_ms,
            "total_pc_pipeline_ms": total_pc_pipeline_ms,
        }

    def _infer_once(self, frame_bgr, collect_latency: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scene_reset = self._consume_scene_reset_request()
        h, w = frame_bgr.shape[:2]
        ts = time.time()

        vision_result = self.vision.det_and_cls(
            frame_bgr,
            stop_event=STOP_EVENT,
            return_timing=collect_latency,
            sync_timing=collect_latency,
        )
        if collect_latency:
            det_boxes, cls_names, cls_confs, area_types, (img_w, img_h), vision_timing = vision_result
        else:
            det_boxes, cls_names, cls_confs, area_types, (img_w, img_h) = vision_result
            vision_timing = {}
        t_vision = time.perf_counter()

        t_hand0 = time.perf_counter()
        detected_hand_labels = [
            name for name, area_type in zip(cls_names, area_types) if area_type == "hand" and name
        ]
        detected_table_labels = [
            name for name, area_type in zip(cls_names, area_types) if area_type == "table" and name
        ]
        live_hand_labels = mahjong.sort_tile_labels(detected_hand_labels)
        live_hand_tiles = mahjong.tile_labels_to_ids(live_hand_labels)
        hand_flow_events: List[Dict[str, Any]] = []
        self_discard_event = self._try_complete_self_discard_from_live(live_hand_labels, live_hand_tiles)
        if self_discard_event is not None:
            hand_flow_events.append(self_discard_event)
            hand_flow_events.append(
                self.game_tracker.complete_self_discard_from_hand(int(self_discard_event["tile_id"]))
            )
        promoted_event = self._maybe_promote_live_draw_to_stable(live_hand_labels, live_hand_tiles)
        if promoted_event is not None:
            hand_flow_events.append(promoted_event)
        effective_hand_labels, detected_hand_tiles, hand_stable, hand_stable_source = self._set_latest_and_get_effective_hand(
            live_hand_labels,
            live_hand_tiles,
        )
        t_hand1 = time.perf_counter()
        debug_info: Dict[str, Any] = {
            "det_count": len(det_boxes),
            "raw_detection_count": int(vision_timing.get("raw_detection_count", len(det_boxes))) if vision_timing else len(det_boxes),
            "valid_detection_count": len(det_boxes),
            "classified_crop_count": int(vision_timing.get("classified_crop_count", len(det_boxes))) if vision_timing else len(det_boxes),
            "hand_count": len(effective_hand_labels),
            "live_hand_count": len(live_hand_labels),
            "hand_stable": hand_stable,
            "hand_stable_source": hand_stable_source,
            "scene_reset": scene_reset,
            "table_count": len(detected_table_labels),
            "hand_id_count": len(detected_hand_tiles),
            "vision_ms": (t_vision - t0) * 1000.0,
            "hand_parse_ms": (t_hand1 - t_hand0) * 1000.0,
            "tracking_ms": 0.0,
            "tile_build_ms": 0.0,
            "track_count": 0,
            "agent_ms": 0.0,
            "total_ms": 0.0,
            "hand_events": hand_flow_events,
        }
        debug_info.update(vision_timing)

        if not det_boxes:
            empty_advice = {
                "benefit": {"tile_id": -1, "tile": "", "source": "", "reason": ""},
                "safe": {"tile_id": -1, "tile": "", "source": "", "reason": ""},
            }
            agent_timing: Dict[str, Any] = {}
            t_agent0 = time.perf_counter()
            agent_result = mahjong.maybe_correct_self_turn_and_suggest(
                self.game_tracker,
                detected_hand_tiles,
                table_observations=[],
                rl_model=self.ppo_model,
                timing=agent_timing if collect_latency else None,
                fill_invalid_hand_for_ppo=collect_latency and self.ppo_fill_invalid_hand,
                ppo_fill_target_count=self.ppo_fill_target_count,
            )
            t_agent1 = time.perf_counter()
            stable_table = self.game_tracker.stable_table_labels()
            debug_info["agent_ms"] = (t_agent1 - t_agent0) * 1000.0
            debug_info["total_ms"] = (t_agent1 - t0) * 1000.0
            debug_info["stable_table_count"] = len(stable_table)
            debug_info.update(agent_timing)
            return {
                "ts": ts,
                "img_w": w,
                "img_h": h,
                "tiles": [],
                "hand": effective_hand_labels,
                "table": stable_table,
                "detected_hand_tiles": detected_hand_tiles,
                "hand_stable": hand_stable,
                "hand_stable_source": hand_stable_source,
                "agent": agent_result,
                "advice": empty_advice,
                "debug": debug_info,
            }

        t_tracking0 = time.perf_counter()
        tracks = self.tracker.update(det_boxes, cls_names, cls_confs, area_types)
        t_tracking1 = time.perf_counter()
        debug_info["tracking_ms"] = (t_tracking1 - t_tracking0) * 1000.0
        debug_info["track_count"] = len(tracks)

        tiles: List[Dict[str, Any]] = []
        table_observations: List[Dict[str, Any]] = []
        t_tile0 = time.perf_counter()
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            stable_cls = tr.stable_cls()
            stable_conf = tr.stable_conf()
            stable_area_type = tr.stable_area_type()
            tiles.append(
                {
                    "id": tr.id,
                    "cls": stable_cls,
                    "conf": stable_conf,
                    "area": stable_area_type,
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(bw),
                    "h": float(bh),
                }
            )
            if stable_area_type == "table":
                tile_id = mahjong.tile_label_to_id(stable_cls)
                if tile_id is not None:
                    table_observations.append(
                        {
                            "track_id": int(tr.id),
                            "tile_id": int(tile_id),
                            "label": stable_cls,
                            "conf": float(stable_conf),
                            "cx": float(cx),
                            "cy": float(cy),
                        }
                    )

        tiles.sort(key=lambda t: t["cx"])
        debug_info["tile_build_ms"] = (time.perf_counter() - t_tile0) * 1000.0
        hand = effective_hand_labels
        live_table = mahjong.sort_tile_labels(detected_table_labels)
        agent_timing = {}
        t_agent0 = time.perf_counter()
        agent_result = mahjong.maybe_correct_self_turn_and_suggest(
            self.game_tracker,
            detected_hand_tiles,
            table_observations=table_observations,
            rl_model=self.ppo_model,
            timing=agent_timing if collect_latency else None,
            fill_invalid_hand_for_ppo=collect_latency and self.ppo_fill_invalid_hand,
            ppo_fill_target_count=self.ppo_fill_target_count,
        )
        t_agent1 = time.perf_counter()
        stable_table = self.game_tracker.stable_table_labels()
        table = stable_table if stable_table else live_table
        debug_info["agent_ms"] = (t_agent1 - t_agent0) * 1000.0
        debug_info["total_ms"] = (t_agent1 - t0) * 1000.0
        debug_info.update(agent_timing)
        debug_info["stable_table_count"] = agent_result.get("live", {}).get("stable_table_count", 0)
        debug_info["table_events"] = agent_result.get("live", {}).get("table_events", [])
        if agent_result.get("decision_type") == "discard":
            recommended_discard_id = int(agent_result.get("recommended_action", -1))
            if not (0 <= recommended_discard_id < 34):
                recommended_discard_id = -1
            if not hand_stable and self._mark_hand_stable(
                hand,
                detected_hand_tiles,
                recommended_discard_id if recommended_discard_id >= 0 else None,
            ):
                hand_stable = True
                hand_stable_source = "auto"
                debug_info["hand_stable"] = True
                debug_info["hand_stable_source"] = "auto"
            elif hand_stable and recommended_discard_id >= 0:
                self._set_stable_recommended_discard(recommended_discard_id)
        suggested_tile_id = int(agent_result.get("recommended_action", -1))
        suggested_tile = str(agent_result.get("recommended_tile", ""))
        safe_tile_id = mahjong.fallback_discard_tile(detected_hand_tiles)
        safe_tile = mahjong.tile_id_to_label(safe_tile_id) if safe_tile_id is not None else ""
        advice = {
            "benefit": {
                "tile_id": suggested_tile_id,
                "tile": suggested_tile,
                "source": agent_result.get("source", ""),
                "reason": agent_result.get("reason", ""),
            },
            "safe": {
                "tile_id": int(safe_tile_id) if safe_tile_id is not None else -1,
                "tile": safe_tile,
                "source": "fallback",
                "reason": "Legal fallback discard from detected hand",
            },
        }

        if self.view:
            vis = frame_bgr.copy()
            for t in tiles:
                x1 = int((t["cx"] - t["w"] / 2) * w)
                y1 = int((t["cy"] - t["h"] / 2) * h)
                x2 = int((t["cx"] + t["w"] / 2) * w)
                y2 = int((t["cy"] + t["h"] / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{t['cls']}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            overlay_lines = [
                "hand: " + (" ".join(hand) if hand else "detecting"),
                f"suggest: {agent_result.get('recommended_tile', '') or 'waiting'}",
            ]
            for idx, line in enumerate(overlay_lines):
                cv2.putText(
                    vis,
                    line,
                    (12, 28 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            cv2.imshow("PC Inference (press q to quit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                STOP_EVENT.set()
                raise InterruptedError("Quit by 'q'.")

        return {
            "ts": ts,
            "img_w": w,
            "img_h": h,
            "tiles": tiles,
            "hand": hand,
            "table": table,
            "detected_hand_tiles": detected_hand_tiles,
            "hand_stable": hand_stable,
            "hand_stable_source": hand_stable_source,
            "agent": agent_result,
            "advice": advice,
            "debug": debug_info,
        }

    def serve_forever(self, print_interval_sec: float = 10.0):
        print(f"[PC] Listening on {self.host}:{self.port}", flush=True)
        print(
            f"[PC] Inference runs on each new frame; terminal log at least every {print_interval_sec:.1f}s",
            flush=True,
        )

        server_conn: Optional[socket.socket] = None

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(1)
                s.settimeout(1.0)  # accept: avoid blocking forever so we can check STOP_EVENT

                while not STOP_EVENT.is_set():
                    try:
                        conn, addr = s.accept()
                    except socket.timeout:
                        continue

                    print(f"[PC] Client connected: {addr}", flush=True)
                    server_conn = conn

                    # conn: set timeout so recv can check STOP_EVENT
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.settimeout(1.0)

                    # reset state for each new client
                    self._latest_jpg = None
                    self._latest_ts = 0.0
                    self._recv_packets = 0
                    self._recv_bytes = 0
                    self._receiver_error = ""
                    self._stable_hand_active = False
                    self._stable_hand_labels = []
                    self._stable_hand_tiles = []
                    self._stable_recommended_discard_id = None
                    self._latest_hand_labels = []
                    self._latest_hand_tiles = []
                    self._live_hand_candidate_signature = None
                    self._live_hand_candidate_frames = 0
                    self._scene_reset_requested = False
                    self.tracker.reset()
                    self.game_tracker.reset_runtime()

                    # send lock for thread-safe full-duplex (send in main thread, recv in receiver thread)
                    send_lock = threading.Lock()

                    recv_th = threading.Thread(target=self._receiver_loop, args=(conn,), daemon=False)
                    recv_th.start()

                    last_status_print_time = 0.0
                    last_processed_frame_ts = -1.0
                    client_connected_time = time.time()
                    client_end_reason = "client_connection_closed"

                    try:
                        while not STOP_EVENT.is_set():
                            now = time.time()
                            jpg, frame_ts, recv_packets, recv_bytes, receiver_error = self._get_latest()

                            if receiver_error:
                                client_end_reason = "receiver_error_before_frames" if jpg is None else "receiver_error"
                                print(
                                    f"[PC] {time.strftime('%H:%M:%S')} | closing client after receiver error: "
                                    f"{receiver_error}",
                                    flush=True,
                                )
                                break

                            if jpg is None:
                                if now - last_status_print_time >= print_interval_sec:
                                    last_status_print_time = now
                                    err = f" | receiver_error={receiver_error}" if receiver_error else ""
                                    print(
                                        f"[PC] {time.strftime('%H:%M:%S')} | waiting for frames "
                                        f"(packets={recv_packets}, bytes={recv_bytes}){err}",
                                        flush=True,
                                    )
                                if now - client_connected_time >= self.client_idle_timeout:
                                    print(
                                        f"[PC] {time.strftime('%H:%M:%S')} | "
                                        f"closing idle client: no frame for {self.client_idle_timeout:.1f}s",
                                        flush=True,
                                    )
                                    client_end_reason = "idle_no_frames_timeout"
                                    break
                                time.sleep(0.01)
                                continue

                            if frame_ts == last_processed_frame_ts:
                                recording_wait = self.latency_recorder.is_recording()
                                stale_timeout = (
                                    self.latency_stale_timeout if recording_wait else self.client_idle_timeout
                                )
                                if now - frame_ts >= stale_timeout:
                                    print(
                                        f"[PC] {time.strftime('%H:%M:%S')} | "
                                        f"closing stale client: no new frame for {stale_timeout:.1f}s",
                                        flush=True,
                                    )
                                    client_end_reason = (
                                        "latency_stale_frame_timeout" if recording_wait else "stale_frame_timeout"
                                    )
                                    break
                                time.sleep(0.005)
                                continue

                            last_processed_frame_ts = frame_ts

                            try:
                                recording_this_frame = self.latency_recorder.is_recording()
                                t_pipeline0 = time.perf_counter()
                                t_decode0 = time.perf_counter()
                                frame = decode_jpg(jpg)
                                decode_ms = (time.perf_counter() - t_decode0) * 1000.0
                                out = self._infer_once(frame, collect_latency=recording_this_frame)
                                pc_frame_age_ms = (time.time() - frame_ts) * 1000.0
                                debug = out.setdefault("debug", {})
                                debug["pc_frame_age_ms"] = pc_frame_age_ms
                                debug["decode_ms"] = decode_ms
                                debug["latency_recording"] = recording_this_frame
                                debug["latency_run_id"] = self.latency_recorder.run_id if recording_this_frame else ""
                                debug["latency_frame_index"] = (
                                    self.latency_recorder.frame_count + 1 if recording_this_frame else 0
                                )
                                debug["pc_pipeline_before_send_ms"] = (time.perf_counter() - t_pipeline0) * 1000.0
                                pc_log = self._format_pc_log(out)
                                out["pc_log"] = pc_log

                                agent = out.get("agent", {})
                                should_print = (
                                    now - last_status_print_time >= print_interval_sec
                                    or bool(agent.get("corrected", False))
                                )
                                if should_print:
                                    last_status_print_time = now
                                    print(pc_log, flush=True)

                                # ---- send JSON back to Quest (length-prefixed, big-endian) ----
                                try:
                                    t_json0 = time.perf_counter()
                                    payload = json.dumps(out, ensure_ascii=False).encode("utf-8")
                                    json_serialize_ms = (time.perf_counter() - t_json0) * 1000.0
                                    t_send0 = time.perf_counter()
                                    send_packet(conn, payload, send_lock, STOP_EVENT)
                                    send_ms = (time.perf_counter() - t_send0) * 1000.0
                                    total_pc_pipeline_ms = (time.perf_counter() - t_pipeline0) * 1000.0
                                    if recording_this_frame:
                                        row = self._build_latency_row(
                                            out=out,
                                            jpg_bytes=len(jpg),
                                            recv_packets=recv_packets,
                                            decode_ms=decode_ms,
                                            json_serialize_ms=json_serialize_ms,
                                            send_ms=send_ms,
                                            total_pc_pipeline_ms=total_pc_pipeline_ms,
                                            pc_frame_age_ms=pc_frame_age_ms,
                                        )
                                        self.latency_recorder.record_frame(row)
                                except Exception as e:
                                    # if send fails, likely the Quest side closed the socket
                                    print(f"[PC] {time.strftime('%H:%M:%S')} | send failed: {e}", flush=True)
                                    client_end_reason = "send_failed"
                                    raise
                            except InterruptedError:
                                client_end_reason = "interrupted"
                                raise
                            except Exception as e:
                                print(f"[PC] {time.strftime('%H:%M:%S')} | infer failed: {e}", flush=True)

                    except InterruptedError:
                        pass
                    except (ConnectionError, OSError) as e:
                        client_end_reason = "connection_error"
                        if not STOP_EVENT.is_set():
                            print(f"[PC] Client disconnected / error: {e}", flush=True)
                    finally:
                        self._auto_stop_latency_recording(client_end_reason)
                        # Close client to unblock receiver
                        safe_close_conn(conn)
                        try:
                            recv_th.join(timeout=2.0)
                        except Exception:
                            pass
                        server_conn = None

        finally:
            safe_close_conn(server_conn)
            self.latency_recorder.close()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("[PC] Server stopped.", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True, help="path to YOLO detect/segment .pt")
    ap.add_argument("--cls", required=True, help="path to classification .pt/.pth (MobileNetV3 Small weights)")

    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)

    ap.add_argument("--det-imgsz", type=int, default=960)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.45)

    ap.add_argument("--cls-imgsz", type=int, default=96)
    ap.add_argument("--cls-conf", type=float, default=0.5, help="drop classify results below this confidence")
    ap.add_argument("--cls-labels", default=None, help="optional labels txt (one class name per line)")
    ap.add_argument("--cls-nc", type=int, default=None, help="optional override num_classes if inference fails")

    ap.add_argument("--ppo-model", default=None, help="path to SB3 PPO zip model (e.g., tw_mahjong_ppo_gpu.zip)")
    ap.add_argument("--ppo-device", default=None, help="SB3 device: cpu / cuda / cuda:0 (optional)")
    ap.add_argument(
        "--no-ppo-fill-invalid-hand",
        action="store_false",
        dest="ppo_fill_invalid_hand",
        default=True,
        help="disable latency-mode synthetic hand filling when the detected hand cannot enter PPO",
    )
    ap.add_argument(
        "--ppo-fill-target-count",
        type=int,
        default=14,
        help="synthetic hand size for latency-mode PPO probing; must be 3n+2, default 14",
    )

    ap.add_argument("--print-interval", type=float, default=10.0, help="seconds between terminal advice prints")
    ap.add_argument(
        "--client-idle-timeout",
        type=float,
        default=5.0,
        help="seconds to keep a connected Quest client that sends no frames before accepting a new one",
    )

    ap.add_argument("--crop-pad", type=float, default=0.08, help="expand crop padding ratio")
    ap.add_argument("--track-iou", type=float, default=0.30)
    ap.add_argument("--track-ttl", type=float, default=0.35, help="seconds to keep missing tracks")
    ap.add_argument("--smooth-len", type=int, default=5)

    ap.add_argument("--view", action="store_true", help="show debug window")
    ap.add_argument("--debug-vision", action="store_true", help="save Quest frame and classification crops periodically")
    ap.add_argument("--debug-dir", default="debug_runtime", help="directory for --debug-vision dumps")
    ap.add_argument("--debug-interval", type=float, default=1.0, help="seconds between vision debug dumps")
    ap.add_argument(
        "--latency-output-dir",
        default="seg_latency_results",
        help="directory for A-button full-pipeline latency CSV/metadata outputs",
    )
    ap.add_argument(
        "--latency-exit-after-stop",
        action="store_true",
        help="stop the PC server after receiving the A-button latency stop command",
    )
    ap.add_argument(
        "--latency-stale-timeout",
        type=float,
        default=5.0,
        help="seconds without new Quest frames while recording before auto-stopping the latency run",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="e.g. cuda:0 or cpu (for ultralytics). Torch classifier follows this if possible.",
    )

    return ap


def main():
    args = build_argparser().parse_args()

    srv = InferServer(
        yolo_path=args.yolo,
        cls_path=args.cls,
        host=args.host,
        port=args.port,
        det_imgsz=args.det_imgsz,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        cls_imgsz=args.cls_imgsz,
        crop_pad=args.crop_pad,
        track_iou=args.track_iou,
        track_ttl=args.track_ttl,
        smooth_len=args.smooth_len,
        view=args.view,
        device=args.device,
        cls_labels_path=args.cls_labels,
        cls_nc=args.cls_nc,
        cls_conf_threshold=args.cls_conf,
        ppo_model_path=args.ppo_model,
        ppo_device=args.ppo_device,
        ppo_fill_invalid_hand=args.ppo_fill_invalid_hand,
        ppo_fill_target_count=args.ppo_fill_target_count,
        client_idle_timeout=args.client_idle_timeout,
        debug_vision_dir=args.debug_dir if args.debug_vision else None,
        debug_vision_interval=args.debug_interval,
        latency_output_dir=args.latency_output_dir,
        latency_exit_after_stop=args.latency_exit_after_stop,
        latency_stale_timeout=args.latency_stale_timeout,
    )

    srv.serve_forever(print_interval_sec=args.print_interval)


if __name__ == "__main__":
    main()
