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
        ppo_model_path: Optional[str] = None,
        ppo_device: Optional[str] = None,
    ):
        self.host = host
        self.port = port

        self.view = view
        self.device = device

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
        )

        # Tracking
        self.tracker = Tracker(track_iou=track_iou, track_ttl=track_ttl, smooth_len=smooth_len)

        # Mahjong PPO model (optional)
        self.ppo_model = mahjong.load_ppo_model(ppo_model_path, ppo_device=ppo_device)

        # Shared (latest frame)
        self._latest_jpg: Optional[bytes] = None
        self._latest_ts = 0.0
        self._lock = threading.Lock()

    def _receiver_loop(self, conn: socket.socket):
        try:
            while not STOP_EVENT.is_set():
                try:
                    jpg = recv_packet(conn, STOP_EVENT)
                except socket.timeout:
                    continue
                except InterruptedError:
                    break

                now = time.time()
                with self._lock:
                    self._latest_jpg = jpg
                    self._latest_ts = now
        except Exception as e:
            if not STOP_EVENT.is_set():
                print(f"[Receiver] stopped: {e}")

    def _get_latest(self):
        with self._lock:
            return self._latest_jpg, self._latest_ts

    def _infer_once(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        ts = time.time()

        det_boxes, cls_names, cls_confs, (img_w, img_h) = self.vision.det_and_cls(frame_bgr, stop_event=STOP_EVENT)

        if not det_boxes:
            empty_advice = {
                "benefit": {"tile_id": -1, "tile": "", "source": "", "reason": ""},
                "safe": {"tile_id": -1, "tile": "", "source": "", "reason": ""},
            }
            return {"ts": ts, "img_w": w, "img_h": h, "tiles": [], "hand": [], "advice": empty_advice}

        tracks = self.tracker.update(det_boxes, cls_names, cls_confs)

        tiles: List[Dict[str, Any]] = []
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            tiles.append(
                {
                    "id": tr.id,
                    "cls": tr.stable_cls(),
                    "conf": tr.stable_conf(),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(bw),
                    "h": float(bh),
                }
            )

        tiles.sort(key=lambda t: t["cx"])
        hand = [t["cls"] for t in tiles]

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
                    f"{t['cls']} {t['conf']:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("PC Inference (press q to quit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                STOP_EVENT.set()
                raise InterruptedError("Quit by 'q'.")

        advice = mahjong.make_advice(hand, ppo_model=self.ppo_model)
        return {"ts": ts, "img_w": w, "img_h": h, "tiles": tiles, "hand": hand, "advice": advice}

    def serve_forever(self, print_interval_sec: float = 10.0):
        print(f"[PC] Listening on {self.host}:{self.port}", flush=True)
        print(
            f"[PC] Advice output every {print_interval_sec:.1f}s + send JSON response back to Quest",
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
                    self.tracker.reset()

                    # send lock for thread-safe full-duplex (send in main thread, recv in receiver thread)
                    send_lock = threading.Lock()

                    recv_th = threading.Thread(target=self._receiver_loop, args=(conn,), daemon=False)
                    recv_th.start()

                    last_print_time = 0.0
                    last_printed_frame_ts = -1.0

                    try:
                        while not STOP_EVENT.is_set():
                            now = time.time()
                            if now - last_print_time < print_interval_sec:
                                time.sleep(0.01)
                                continue

                            jpg, frame_ts = self._get_latest()
                            last_print_time = now  # keep fixed cadence even without new frames

                            if jpg is None:
                                print(f"[PC] {time.strftime('%H:%M:%S')} | waiting for frames...", flush=True)
                                continue

                            if frame_ts == last_printed_frame_ts:
                                print(
                                    f"[PC] {time.strftime('%H:%M:%S')} | no new frame (last ts={frame_ts:.3f})",
                                    flush=True,
                                )
                                continue

                            last_printed_frame_ts = frame_ts

                            try:
                                frame = decode_jpg(jpg)
                                out = self._infer_once(frame)

                                hand = out.get("hand", [])
                                advice = out.get("advice", {})
                                benefit = advice.get("benefit", {})
                                safe = advice.get("safe", {})

                                benefit_tile = benefit.get("tile", "")
                                benefit_id = benefit.get("tile_id", -1)
                                benefit_src = benefit.get("source", "")

                                safe_tile = safe.get("tile", "")
                                safe_id = safe.get("tile_id", -1)
                                safe_src = safe.get("source", "")

                                hand_str = " ".join([str(x) for x in hand]) if isinstance(hand, list) else str(hand)

                                print(
                                    f"[PC] {time.strftime('%H:%M:%S')} | "
                                    f"benefit: {benefit_tile} (id={benefit_id}, {benefit_src}) | "
                                    f"safe: {safe_tile} (id={safe_id}, {safe_src})",
                                    flush=True,
                                )
                                print(f"      hand: {hand_str}", flush=True)

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
    ap.add_argument("--yolo", required=True, help="path to yolo detection .pt")
    ap.add_argument("--cls", required=True, help="path to classification .pt/.pth (MobileNetV3 Small weights)")

    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)

    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.45)

    ap.add_argument("--cls-imgsz", type=int, default=224)
    ap.add_argument("--cls-labels", default=None, help="optional labels txt (one class name per line)")
    ap.add_argument("--cls-nc", type=int, default=None, help="optional override num_classes if inference fails")

    ap.add_argument("--ppo-model", default=None, help="path to SB3 PPO zip model (e.g., tw_mahjong_ppo_gpu.zip)")
    ap.add_argument("--ppo-device", default=None, help="SB3 device: cpu / cuda / cuda:0 (optional)")

    ap.add_argument("--print-interval", type=float, default=10.0, help="seconds between terminal advice prints")

    ap.add_argument("--crop-pad", type=float, default=0.08, help="expand crop padding ratio")
    ap.add_argument("--track-iou", type=float, default=0.30)
    ap.add_argument("--track-ttl", type=float, default=0.35, help="seconds to keep missing tracks")
    ap.add_argument("--smooth-len", type=int, default=5)

    ap.add_argument("--view", action="store_true", help="show debug window")
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
        ppo_model_path=args.ppo_model,
        ppo_device=args.ppo_device,
    )

    srv.serve_forever(print_interval_sec=args.print_interval)


if __name__ == "__main__":
    main()
