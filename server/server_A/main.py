import argparse
import json
import socket
import threading
import time
import signal
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
        client_idle_timeout: float = 5.0,
        debug_vision_dir: Optional[str] = None,
        debug_vision_interval: float = 1.0,
    ):
        self.host = host
        self.port = port

        self.view = view
        self.device = device
        self.client_idle_timeout = float(client_idle_timeout)

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
        if command in ("scene_reset", "reset_scene", "reset"):
            with self._lock:
                self._scene_reset_requested = True
            print("[Control] scene reset requested", flush=True)
            return True

        return True

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

    def _infer_once(self, frame_bgr) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scene_reset = self._consume_scene_reset_request()
        h, w = frame_bgr.shape[:2]
        ts = time.time()

        det_boxes, cls_names, cls_confs, area_types, (img_w, img_h) = self.vision.det_and_cls(
            frame_bgr,
            stop_event=STOP_EVENT,
        )
        t_vision = time.perf_counter()

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
        debug_info: Dict[str, Any] = {
            "det_count": len(det_boxes),
            "hand_count": len(effective_hand_labels),
            "live_hand_count": len(live_hand_labels),
            "hand_stable": hand_stable,
            "hand_stable_source": hand_stable_source,
            "scene_reset": scene_reset,
            "table_count": len(detected_table_labels),
            "hand_id_count": len(detected_hand_tiles),
            "vision_ms": (t_vision - t0) * 1000.0,
            "agent_ms": 0.0,
            "total_ms": 0.0,
            "hand_events": hand_flow_events,
        }

        if not det_boxes:
            empty_advice = {
                "benefit": {"tile_id": -1, "tile": "", "source": "", "reason": ""},
                "safe": {"tile_id": -1, "tile": "", "source": "", "reason": ""},
            }
            t_agent0 = time.perf_counter()
            agent_result = mahjong.maybe_correct_self_turn_and_suggest(
                self.game_tracker,
                detected_hand_tiles,
                table_observations=[],
                rl_model=self.ppo_model,
            )
            t_agent1 = time.perf_counter()
            stable_table = self.game_tracker.stable_table_labels()
            debug_info["agent_ms"] = (t_agent1 - t_agent0) * 1000.0
            debug_info["total_ms"] = (t_agent1 - t0) * 1000.0
            debug_info["stable_table_count"] = len(stable_table)
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

        tracks = self.tracker.update(det_boxes, cls_names, cls_confs, area_types)

        tiles: List[Dict[str, Any]] = []
        table_observations: List[Dict[str, Any]] = []
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
        hand = effective_hand_labels
        live_table = mahjong.sort_tile_labels(detected_table_labels)
        t_agent0 = time.perf_counter()
        agent_result = mahjong.maybe_correct_self_turn_and_suggest(
            self.game_tracker,
            detected_hand_tiles,
            table_observations=table_observations,
            rl_model=self.ppo_model,
        )
        t_agent1 = time.perf_counter()
        stable_table = self.game_tracker.stable_table_labels()
        table = stable_table if stable_table else live_table
        debug_info["agent_ms"] = (t_agent1 - t_agent0) * 1000.0
        debug_info["total_ms"] = (t_agent1 - t0) * 1000.0
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

                    try:
                        while not STOP_EVENT.is_set():
                            now = time.time()
                            jpg, frame_ts, recv_packets, recv_bytes, receiver_error = self._get_latest()

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
                                    break
                                time.sleep(0.01)
                                continue

                            if frame_ts == last_processed_frame_ts:
                                if now - frame_ts >= self.client_idle_timeout:
                                    print(
                                        f"[PC] {time.strftime('%H:%M:%S')} | "
                                        f"closing stale client: no new frame for {self.client_idle_timeout:.1f}s",
                                        flush=True,
                                    )
                                    break
                                time.sleep(0.005)
                                continue

                            last_processed_frame_ts = frame_ts

                            try:
                                frame = decode_jpg(jpg)
                                out = self._infer_once(frame)
                                out.setdefault("debug", {})["pc_frame_age_ms"] = (time.time() - frame_ts) * 1000.0
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
                                    payload = json.dumps(out, ensure_ascii=False).encode("utf-8")
                                    send_packet(conn, payload, send_lock, STOP_EVENT)
                                except Exception as e:
                                    # if send fails, likely the Quest side closed the socket
                                    print(f"[PC] {time.strftime('%H:%M:%S')} | send failed: {e}", flush=True)
                                    raise
                            except InterruptedError:
                                raise
                            except Exception as e:
                                print(f"[PC] {time.strftime('%H:%M:%S')} | infer failed: {e}", flush=True)

                    except InterruptedError:
                        pass
                    except (ConnectionError, OSError) as e:
                        if not STOP_EVENT.is_set():
                            print(f"[PC] Client disconnected / error: {e}", flush=True)
                    finally:
                        # Close client to unblock receiver
                        safe_close_conn(conn)
                        try:
                            recv_th.join(timeout=2.0)
                        except Exception:
                            pass
                        server_conn = None

        finally:
            safe_close_conn(server_conn)
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
        client_idle_timeout=args.client_idle_timeout,
        debug_vision_dir=args.debug_dir if args.debug_vision else None,
        debug_vision_interval=args.debug_interval,
    )

    srv.serve_forever(print_interval_sec=args.print_interval)


if __name__ == "__main__":
    main()
