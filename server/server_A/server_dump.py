import argparse
import os
import json
import socket
import struct
import threading
import time
import signal
from collections import deque, Counter
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from ultralytics import YOLO

import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

import re

# Optional: SB3 PPO Mahjong helper
try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:
    PPO = None  # type: ignore


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


# -----------------------------
# TCP helpers (length-prefixed)
# -----------------------------
def recv_exact(conn: socket.socket, n: int, stop_event: threading.Event) -> bytes:
    """
    Receive exactly n bytes.
    - conn must have a timeout set (so we can periodically check stop_event).
    - Will keep partial data on timeouts and continue reading.
    - Raises InterruptedError if stop_event is set.
    """
    buf = bytearray()
    while len(buf) < n:
        if stop_event.is_set():
            raise InterruptedError("Stopped by user.")
        try:
            chunk = conn.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        buf.extend(chunk)
    return bytes(buf)


def recv_packet(conn: socket.socket, stop_event: threading.Event) -> bytes:
    # 4 bytes length (big-endian) + payload
    header = recv_exact(conn, 4, stop_event)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > 50_000_000:
        raise ValueError(f"Invalid packet length: {length}")
    return recv_exact(conn, length, stop_event)


def send_packet(conn: socket.socket, payload: bytes, lock: threading.Lock, stop_event: threading.Event):
    if stop_event.is_set():
        return
    header = struct.pack(">I", len(payload))
    with lock:
        conn.sendall(header + payload)


def safe_close_conn(conn: Optional[socket.socket]):
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


# -----------------------------
# Simple IOU tracking + smoothing
# -----------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


class Track:
    def __init__(self, tid: int, bbox_xyxy, cls_name: str, conf: float, hist_len: int):
        self.id = tid
        self.bbox = bbox_xyxy
        self.cls_hist = deque([cls_name], maxlen=hist_len)
        self.conf_hist = deque([conf], maxlen=hist_len)
        self.last_seen = time.time()

    def update(self, bbox_xyxy, cls_name: str, conf: float):
        self.bbox = bbox_xyxy
        self.cls_hist.append(cls_name)
        self.conf_hist.append(conf)
        self.last_seen = time.time()

    def stable_cls(self) -> str:
        c = Counter(self.cls_hist)
        return c.most_common(1)[0][0] if c else ""

    def stable_conf(self) -> float:
        return float(np.mean(self.conf_hist)) if self.conf_hist else 0.0


# -----------------------------
# Torch MobileNetV3 Small classifier
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_label_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [ln for ln in lines if ln]
    return lines if lines else None


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    支援常見存法：
    - torch.save(model.state_dict(), path)  -> OrderedDict
    - torch.save({"state_dict": ...}, path) -> dict with key
    - torch.save({"model_state_dict": ...}, path)
    - torch.save(model, path)              -> nn.Module
    """
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in ckpt:
                obj = ckpt[key]
                if isinstance(obj, torch.nn.Module):
                    return obj.state_dict()
                if isinstance(obj, dict):
                    return obj
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore

    if hasattr(ckpt, "keys") and hasattr(ckpt, "items"):
        return dict(ckpt.items())  # type: ignore

    raise ValueError("Unsupported checkpoint format for classifier .pt/.pth")


def _cleanup_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        if k2.startswith("model."):
            k2 = k2[len("model.") :]
        new_state[k2] = v
    return new_state


def _infer_num_classes_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    w = state.get("classifier.3.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])

    candidates = []
    for k, v in state.items():
        if "classifier" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            candidates.append((k, v.shape[0]))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return int(candidates[0][1])

    return None


class TorchMBV3SmallClassifier:
    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        img_size: int = 224,
        labels: Optional[List[str]] = None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        override_num_classes: Optional[int] = None,
    ):
        self.device = device
        self.img_size = int(img_size)
        self.labels = labels
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        ckpt = torch.load(weights_path, map_location="cpu")
        state = _cleanup_state_keys(_extract_state_dict(ckpt))

        nc = override_num_classes or _infer_num_classes_from_state(state) or 35

        self.model = mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(in_features, nc)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print(f"[CLS][Torch] Warning: missing keys (first 10): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"[CLS][Torch] Warning: unexpected keys (first 10): {unexpected[:10]}")

        self.model.to(self.device).eval()
        self.num_classes = nc

        if self.labels is not None and len(self.labels) != self.num_classes:
            print(f"[CLS][Torch] Warning: labels count={len(self.labels)} != num_classes={self.num_classes}")

    @torch.no_grad()
    def predict_bgr(self, crop_bgr: np.ndarray):
        if crop_bgr is None or crop_bgr.size == 0:
            return "UNKNOWN", 0.0, -1

        img = cv2.resize(crop_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)[0]
        conf, cls_id = torch.max(prob, dim=0)

        cls_id_i = int(cls_id.item())
        conf_f = float(conf.item())

        if self.labels is not None and 0 <= cls_id_i < len(self.labels):
            name = self.labels[cls_id_i]
        else:
            name = str(cls_id_i)

        return name, conf_f, cls_id_i



# ============================================================
# Mahjong tile label <-> 34-id utilities (for PPO model)
# 34 IDs (common Riichi / Tenhou order):
#   0-8:  1m-9m
#   9-17: 1p-9p
#   18-26:1s-9s
#   27-33: East,South,West,North,White,Green,Red
# ============================================================
_CANONICAL_ID_TO_LABEL = (
    [f"{i}m" for i in range(1, 10)]
    + [f"{i}p" for i in range(1, 10)]
    + [f"{i}s" for i in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
)

_ZH_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}

_HONOR_MAP = {
    "E": 27, "S": 28, "W": 29, "N": 30,
    "P": 31, "F": 32, "C": 33,
    # common alternates
    "WHITE": 31, "WH": 31, "WD": 31, "BAI": 31,
    "GREEN": 32, "GD": 32, "FA": 32,
    "RED": 33, "RD": 33, "ZHONG": 33, "R": 33,
    # Chinese
    "東": 27, "南": 28, "西": 29, "北": 30,
    "白": 31, "發": 32, "发": 32, "中": 33,
}

_SUIT_ALIASES = {
    "m": ("m", "man", "wan", "萬", "万"),
    "p": ("p", "pin", "tong", "筒", "餅", "饼", "pinzu", "pzu"),
    "s": ("s", "sou", "suo", "索", "条", "條", "souzu", "szu"),
}

def tile_label_to_id(label: str) -> Optional[int]:
    '''
    Convert classifier label to tile id (0..33).
    Supports common formats:
      - '1m','9p','3s'
      - '1萬','九萬','3筒','7索','7條'
      - honors: 'E/S/W/N/P/F/C', or '東南西北白發中'
      - numeric string '0'..'33'
    '''
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None

    # numeric id
    if s.isdigit():
        v = int(s)
        return v if 0 <= v < 34 else None

    # direct honors
    up = s.upper().strip()
    if up in _HONOR_MAP:
        return _HONOR_MAP[up]
    if s in _HONOR_MAP:
        return _HONOR_MAP[s]

    s2 = s.replace(" ", "").replace("_", "")
    up2 = s2.upper()
    if up2 in _HONOR_MAP:
        return _HONOR_MAP[up2]
    if s2 in _HONOR_MAP:
        return _HONOR_MAP[s2]

    # suit like 1m / 2p / 3s
    m = re.match(r"^([1-9])([mpsMPS])$", s2)
    if m:
        num = int(m.group(1))
        suit = m.group(2).lower()
        base = {"m": 0, "p": 9, "s": 18}[suit]
        return base + (num - 1)

    # chinese number + suit: 一萬 / 3萬 / 九筒 / 7索 / 7條
    # extract number
    num = None
    if len(s2) >= 2 and s2[0] in _ZH_NUM:
        num = _ZH_NUM[s2[0]]
        rest = s2[1:]
    else:
        m2 = re.match(r"^([1-9])(.+)$", s2)
        if m2:
            num = int(m2.group(1))
            rest = m2.group(2)
        else:
            return None

    if num is None or not (1 <= num <= 9):
        return None

    # detect suit alias
    for suit, aliases in _SUIT_ALIASES.items():
        for a in aliases:
            if rest == a or rest.endswith(a):
                base = {"m": 0, "p": 9, "s": 18}[suit]
                return base + (num - 1)

    return None


def tile_id_to_label(tile_id: int) -> str:
    if 0 <= int(tile_id) < 34:
        return _CANONICAL_ID_TO_LABEL[int(tile_id)]
    return str(tile_id)


def choose_display_label(tile_id: int, original_labels: List[str]) -> str:
    '''
    Prefer a label already in the hand (so Quest displays consistent naming),
    otherwise fallback to canonical.
    '''
    for lb in original_labels:
        tid = tile_label_to_id(lb)
        if tid == tile_id:
            return lb
    return tile_id_to_label(tile_id)


def obs34_from_hand_labels(hand_labels: List[str]) -> np.ndarray:
    cnt = np.zeros((34,), dtype=np.int8)
    for lb in hand_labels:
        tid = tile_label_to_id(lb)
        if tid is None:
            continue
        cnt[tid] += 1
    return cnt


def heuristic_safe_discard_id(cnt: np.ndarray) -> Optional[int]:
    '''
    "Most safe" heuristic without table context:
      1) honors (27..33) singletons, then honors overall
      2) isolated terminals (1/9), then terminals overall
      3) isolated simples
    Returns tile_id.
    '''
    if cnt is None or cnt.shape[0] != 34:
        return None

    def in_same_suit(a: int, b: int) -> bool:
        if a >= 27 or b >= 27:
            return False
        return (a // 9) == (b // 9)

    def connectedness(t: int) -> int:
        # higher = more useful to keep
        c = int(cnt[t])
        if t >= 27:
            return c * 3
        for d in (-2, -1, 1, 2):
            u = t + d
            if 0 <= u < 27 and in_same_suit(t, u):
                c += int(cnt[u])
        return c

    def is_terminal(t: int) -> bool:
        if t >= 27:
            return False
        n = (t % 9) + 1
        return n == 1 or n == 9

    candidates = [i for i in range(34) if cnt[i] > 0]
    if not candidates:
        return None

    def safety_rank(t: int) -> int:
        if t >= 27:
            return 0
        if is_terminal(t):
            return 1
        return 2

    def singleton_bonus(t: int) -> int:
        return 0 if cnt[t] == 1 else 1

    candidates.sort(key=lambda t: (safety_rank(t), singleton_bonus(t), connectedness(t), -int(cnt[t]), t))
    return candidates[0]


def heuristic_benefit_fallback_id(cnt: np.ndarray) -> Optional[int]:
    '''
    When PPO is unavailable/illegal action: discard something that harms hand structure least.
    (Currently same as safe heuristic.)
    '''
    return heuristic_safe_discard_id(cnt)


# -----------------------------
# Inference server
# -----------------------------
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
        dump_dir: Optional[str] = None,
        dump_every: int = 0,
        dump_on_empty: bool = False,
    ):
        self.host = host
        self.port = port

        # Dump (debug) options
        self.dump_dir = dump_dir
        self.dump_every = int(dump_every) if dump_every else 0
        self.dump_on_empty = bool(dump_on_empty)
        self._recv_count = 0
        if self.dump_dir:
            import os
            os.makedirs(self.dump_dir, exist_ok=True)


        # device pick for torch classifier
        if device is None:
            self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            if device.startswith("cuda") and torch.cuda.is_available():
                self.torch_device = torch.device(device)
            else:
                self.torch_device = torch.device("cpu")

        # Models
        self.det_model = YOLO(yolo_path)
        try:
            print(f"[DET] Loaded model task={getattr(self.det_model, 'task', None)} | path={yolo_path}")
            if getattr(self.det_model, 'task', None) != 'detect':
                print("[DET] WARNING: --yolo model task is not 'detect'. If you passed a classify model here, no boxes will ever be produced.")
        except Exception:
            pass

        self.cls_backend = "torch"
        self.cls_model = None
        self.torch_cls: Optional[TorchMBV3SmallClassifier] = None

        labels = _load_label_list(cls_labels_path)

        # Try Ultralytics classify model first; fallback to Torch MBV3 Small
        try:
            _tmp = YOLO(cls_path)
            if getattr(_tmp, "task", None) == "classify":
                self.cls_backend = "ultralytics"
                self.cls_model = _tmp
                print("[CLS] Using Ultralytics classify model.")
            else:
                raise RuntimeError(f"Ultralytics model task={getattr(_tmp, 'task', None)} (not classify)")
        except Exception as e:
            print(f"[CLS] Ultralytics load failed or not classify -> use Torch MobileNetV3 Small. Reason: {e}")
            self.torch_cls = TorchMBV3SmallClassifier(
                weights_path=cls_path,
                device=self.torch_device,
                img_size=cls_imgsz,
                labels=labels,
                override_num_classes=cls_nc,
            )

        # Mahjong PPO model (optional)
        self.ppo_model_path = ppo_model_path
        self.ppo_device = ppo_device
        self.ppo_model = None
        if self.ppo_model_path:
            if PPO is None:
                print("[PPO] stable-baselines3 not installed. Install with: pip install stable-baselines3")
            else:
                try:
                    dev = self.ppo_device or ("cuda" if torch.cuda.is_available() else "cpu")
                    self.ppo_model = PPO.load(self.ppo_model_path, device=dev)
                    print(f"[PPO] Loaded PPO model: {self.ppo_model_path} (device={dev})")
                except Exception as e:
                    print(f"[PPO] Failed to load model '{self.ppo_model_path}': {e}")
                    self.ppo_model = None

        self.det_imgsz = det_imgsz
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.cls_imgsz = cls_imgsz

        self.crop_pad = crop_pad
        self.track_iou = track_iou
        self.track_ttl = track_ttl
        self.smooth_len = smooth_len

        self.view = view
        self.device = device  # for ultralytics.predict

        # Shared (latest frame)
        self._latest_jpg = None
        self._latest_ts = 0.0
        self._lock = threading.Lock()

        # For sending back
        self._send_lock = threading.Lock()

        # Tracking
        self._tracks: dict[int, Track] = {}
        self._next_tid = 0

        self._warned_no_probs = False

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

                # Optional: dump raw incoming frames for debugging
                if self.dump_dir and self.dump_every > 0:
                    self._recv_count += 1
                    if (self._recv_count % self.dump_every) == 0:
                        try:
                            fn = f"in_{self._recv_count:06d}_{int(now*1000)}.jpg"
                            path = os.path.join(self.dump_dir, fn)
                            with open(path, "wb") as f:
                                f.write(jpg)
                        except Exception:
                            pass

        except Exception as e:
            if not STOP_EVENT.is_set():
                print(f"[Receiver] stopped: {e}")

    def _get_latest(self):
        with self._lock:
            return self._latest_jpg, self._latest_ts

    def _decode_jpg(self, jpg: bytes):
        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("Failed to decode JPEG.")
        return img

    def _expand_box(self, box, w, h):
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        px = bw * self.crop_pad
        py = bh * self.crop_pad
        x1 = max(0, int(x1 - px))
        y1 = max(0, int(y1 - py))
        x2 = min(w - 1, int(x2 + px))
        y2 = min(h - 1, int(y2 + py))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _match_tracks(self, det_boxes_xyxy, cls_names, cls_confs):
        used_tracks = set()
        assigned = [-1] * len(det_boxes_xyxy)
        track_items = list(self._tracks.items())

        for i, box in enumerate(det_boxes_xyxy):
            best = (-1, 0.0)
            for tid, tr in track_items:
                if tid in used_tracks:
                    continue
                score = iou_xyxy(box, tr.bbox)
                if score > best[1]:
                    best = (tid, score)
            if best[0] != -1 and best[1] >= self.track_iou:
                assigned[i] = best[0]
                used_tracks.add(best[0])

        now = time.time()
        for i, box in enumerate(det_boxes_xyxy):
            name = cls_names[i]
            conf = float(cls_confs[i])
            tid = assigned[i]
            if tid == -1:
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = Track(tid, box, name, conf, self.smooth_len)
            else:
                self._tracks[tid].update(box, name, conf)

        to_del = []
        for tid, tr in self._tracks.items():
            if (now - tr.last_seen) > self.track_ttl:
                to_del.append(tid)
        for tid in to_del:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _classify_crop(self, crop_bgr: np.ndarray):
        if self.cls_backend == "ultralytics":
            cls_res = self.cls_model.predict(
                crop_bgr,
                imgsz=self.cls_imgsz,
                device=self.device,
                verbose=False,
            )[0]
            if cls_res.probs is None:
                if not self._warned_no_probs:
                    print("[CLS][Ultralytics] Warning: cls_res.probs is None (weights may not be a classify model).")
                    self._warned_no_probs = True
                return "UNKNOWN", 0.0
            top1 = int(cls_res.probs.top1)
            cconf = float(cls_res.probs.top1conf)
            cname = cls_res.names.get(top1, str(top1))
            return cname, cconf

        assert self.torch_cls is not None
        cname, cconf, _ = self.torch_cls.predict_bgr(crop_bgr)
        return cname, cconf

    # -----------------------------
    # Mahjong advice (benefit via PPO, safe via heuristic)
    # -----------------------------
    def _make_advice(self, hand_labels: List[str]) -> Dict[str, Any]:
        cnt = obs34_from_hand_labels(hand_labels)

        # benefit: PPO
        benefit_id: Optional[int] = None
        benefit_src = "ppo"
        benefit_reason = "PPO policy (deterministic)"
        if self.ppo_model is not None:
            try:
                obs = cnt.astype(np.float32)
                action, _ = self.ppo_model.predict(obs, deterministic=True)
                benefit_id = int(action)
                if benefit_id < 0 or benefit_id >= 34 or cnt[benefit_id] <= 0:
                    benefit_reason = "PPO action illegal for current hand; fallback heuristic"
                    benefit_src = "fallback"
                    benefit_id = heuristic_benefit_fallback_id(cnt)
            except Exception as e:
                benefit_reason = f"PPO predict failed; fallback heuristic ({e})"
                benefit_src = "fallback"
                benefit_id = heuristic_benefit_fallback_id(cnt)
        else:
            benefit_src = "fallback"
            benefit_reason = "PPO model not loaded; fallback heuristic"
            benefit_id = heuristic_benefit_fallback_id(cnt)

        # safe: heuristic (no table context)
        safe_id = heuristic_safe_discard_id(cnt)
        safe_src = "heuristic"
        safe_reason = "No table context; heuristic prefers honors/terminals & least-connected tile"

        benefit_tile = choose_display_label(benefit_id, hand_labels) if benefit_id is not None else ""
        safe_tile = choose_display_label(safe_id, hand_labels) if safe_id is not None else ""

        return {
            "benefit": {
                "tile_id": int(benefit_id) if benefit_id is not None else -1,
                "tile": benefit_tile,
                "source": benefit_src,
                "reason": benefit_reason,
            },
            "safe": {
                "tile_id": int(safe_id) if safe_id is not None else -1,
                "tile": safe_tile,
                "source": safe_src,
                "reason": safe_reason,
            },
        }

    def _infer_once(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        ts = time.time()

        det_res = self.det_model.predict(
            frame_bgr,
            imgsz=self.det_imgsz,
            conf=self.det_conf,
            iou=self.det_iou,
            device=self.device,
            verbose=False,
        )[0]

        if det_res.boxes is None or len(det_res.boxes) == 0:
            # Optional: dump frames that produce no detections
            if self.dump_dir and self.dump_on_empty:
                try:
                    fn = f"empty_{int(ts*1000)}.jpg"
                    path = os.path.join(self.dump_dir, fn)
                    cv2.imwrite(path, frame_bgr)
                except Exception:
                    pass
            return {"ts": ts, "img_w": w, "img_h": h, "tiles": [], "hand": [], "advice": {"benefit": {"tile_id": -1, "tile": "", "source": "", "reason": ""}, "safe": {"tile_id": -1, "tile": "", "source": "", "reason": ""}}}

        boxes = det_res.boxes.xyxy.cpu().numpy()

        cls_names = []
        cls_confs = []
        crops_xyxy = []

        for b in boxes:
            if STOP_EVENT.is_set():
                raise InterruptedError("Stopped by user.")

            eb = self._expand_box(b, w, h)
            if eb is None:
                crops_xyxy.append(None)
                cls_names.append("")
                cls_confs.append(0.0)
                continue

            crops_xyxy.append(eb)
            x1, y1, x2, y2 = eb
            crop = frame_bgr[y1:y2, x1:x2]

            cname, cconf = self._classify_crop(crop)
            cls_names.append(cname)
            cls_confs.append(float(cconf))

        valid_boxes = []
        valid_names = []
        valid_confs = []
        for eb, name, conf in zip(crops_xyxy, cls_names, cls_confs):
            if eb is None:
                continue
            valid_boxes.append(eb)
            valid_names.append(name)
            valid_confs.append(conf)

        tracks = self._match_tracks(valid_boxes, valid_names, valid_confs)

        tiles = []
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
                    f'{t["cls"]} {t["conf"]:.2f}',
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

        advice = self._make_advice(hand)

        return {"ts": ts, "img_w": w, "img_h": h, "tiles": tiles, "hand": hand, "advice": advice}

    def serve_forever(self, print_interval_sec: float = 10.0):
        print(f"[PC] Listening on {self.host}:{self.port}", flush=True)
        print(f"[PC] Will output advice every {print_interval_sec:.1f}s and send JSON back to Quest.", flush=True)
        server_conn: Optional[socket.socket] = None

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(1)
                s.settimeout(1.0)  # accept 不要永久卡死，才能定期檢查 STOP_EVENT

                while not STOP_EVENT.is_set():
                    try:
                        conn, addr = s.accept()
                    except socket.timeout:
                        continue

                    print(f"[PC] Client connected: {addr}", flush=True)
                    server_conn = conn

                    # send lock for length-prefixed JSON responses back to Quest
                    send_lock = threading.Lock()

                    # conn: set timeout so recv can check STOP_EVENT
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.settimeout(1.0)

                    # reset state for each new client
                    self._latest_jpg = None
                    self._latest_ts = 0.0
                    self._tracks.clear()
                    self._next_tid = 0

                    recv_th = threading.Thread(target=self._receiver_loop, args=(conn,), daemon=False)
                    recv_th.start()

                    send_lock = threading.Lock()

                    last_print_time = 0.0
                    last_printed_frame_ts = -1.0

                    try:
                        while not STOP_EVENT.is_set():
                            now = time.time()
                            if now - last_print_time < print_interval_sec:
                                time.sleep(0.01)
                                continue

                            jpg, frame_ts = self._get_latest()
                            last_print_time = now  # 即便沒有新 frame，也要維持固定輸出節奏

                            if jpg is None:
                                print(f"[PC] {time.strftime('%H:%M:%S')} | waiting for frames...", flush=True)
                                continue

                            if frame_ts == last_printed_frame_ts:
                                # 沒有新 frame，就重複輸出上一個建議也行；這裡選擇提示
                                print(f"[PC] {time.strftime('%H:%M:%S')} | no new frame (last ts={frame_ts:.3f})", flush=True)
                                continue

                            last_printed_frame_ts = frame_ts

                            try:
                                frame = self._decode_jpg(jpg)
                                out = self._infer_once(frame)  # includes tiles/hand/advice
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

                                # compact hand summary
                                if isinstance(hand, list):
                                    hand_str = " ".join([str(x) for x in hand])
                                else:
                                    hand_str = str(hand)

                                print(
                                    f"[PC] {time.strftime('%H:%M:%S')} | "
                                    f"benefit: {benefit_tile} (id={benefit_id}, {benefit_src}) | "
                                    f"safe: {safe_tile} (id={safe_id}, {safe_src})"
                                , flush=True)
                                print(f"      hand: {hand_str}", flush=True)

                                # Send back to Quest (length-prefixed JSON)
                                pc_log = (
                                    f"[PC] {time.strftime('%H:%M:%S')} | "
                                    f"benefit: {benefit_tile} (id={benefit_id}, {benefit_src}) | "
                                    f"safe: {safe_tile} (id={safe_id}, {safe_src})\n"
                                    f"      hand: {hand_str}"
                                )
                                out["pc_log"] = pc_log
                                payload = json.dumps(out, ensure_ascii=False).encode("utf-8")
                                # Optional: dump json output for debugging
                                if self.dump_dir:
                                    try:
                                        fn = f"out_{int(out.get('ts', time.time())*1000)}.json"
                                        path = os.path.join(self.dump_dir, fn)
                                        with open(path, "wb") as f:
                                            f.write(payload)
                                    except Exception:
                                        pass
                                send_packet(conn, payload, send_lock, STOP_EVENT)
                            except InterruptedError:
                                raise
                            except Exception as e:
                                print(f"[PC] {time.strftime('%H:%M:%S')} | infer failed: {e}", flush=True)

                    except InterruptedError:
                        # STOP_EVENT already set
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
            # Ensure everything is closed
            safe_close_conn(server_conn)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("[PC] Server stopped.", flush=True)

        return


def main():
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

    ap.add_argument("--dump-dir", default=None, help="optional: directory to dump received frames and json (for debugging)")
    ap.add_argument("--dump-every", type=int, default=0, help="dump every N received frames (0=disable). If set, saves incoming jpg as dump_dir/in_XXXX.jpg")
    ap.add_argument("--dump-on-empty", action="store_true", help="if set, also dump frames when no tiles detected (even if dump-every=0)")


    ap.add_argument("--crop-pad", type=float, default=0.08, help="expand crop padding ratio")
    ap.add_argument("--track-iou", type=float, default=0.30)
    ap.add_argument("--track-ttl", type=float, default=0.35, help="seconds to keep missing tracks")
    ap.add_argument("--smooth-len", type=int, default=5)

    ap.add_argument("--view", action="store_true", help="show debug window")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (for ultralytics). Torch classifier follows this if possible.")

    args = ap.parse_args()

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
        dump_dir=args.dump_dir,
        dump_every=args.dump_every,
        dump_on_empty=args.dump_on_empty,
    )

    srv.serve_forever(print_interval_sec=args.print_interval)


if __name__ == "__main__":
    main()
