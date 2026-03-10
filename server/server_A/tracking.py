import time
from collections import Counter, deque
from typing import Dict, List, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection-over-Union for boxes in (x1, y1, x2, y2)."""
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
    def __init__(self, tid: int, bbox_xyxy: Box, cls_name: str, conf: float, hist_len: int):
        self.id = tid
        self.bbox: Box = bbox_xyxy
        self.cls_hist = deque([cls_name], maxlen=hist_len)
        self.conf_hist = deque([conf], maxlen=hist_len)
        self.last_seen = time.time()

    def update(self, bbox_xyxy: Box, cls_name: str, conf: float) -> None:
        self.bbox = bbox_xyxy
        self.cls_hist.append(cls_name)
        self.conf_hist.append(conf)
        self.last_seen = time.time()

    def stable_cls(self) -> str:
        c = Counter(self.cls_hist)
        return c.most_common(1)[0][0] if c else ""

    def stable_conf(self) -> float:
        return float(np.mean(self.conf_hist)) if self.conf_hist else 0.0


class Tracker:
    """Very lightweight IOU-based tracker with short-horizon smoothing."""

    def __init__(self, track_iou: float = 0.30, track_ttl: float = 0.35, smooth_len: int = 5):
        self.track_iou = float(track_iou)
        self.track_ttl = float(track_ttl)
        self.smooth_len = int(smooth_len)
        self._tracks: Dict[int, Track] = {}
        self._next_tid = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_tid = 0

    def update(self, det_boxes_xyxy: List[Box], cls_names: List[str], cls_confs: List[float]) -> List[Track]:
        used_tracks = set()
        assigned = [-1] * len(det_boxes_xyxy)
        track_items = list(self._tracks.items())

        for i, box in enumerate(det_boxes_xyxy):
            best_tid = -1
            best_score = 0.0
            for tid, tr in track_items:
                if tid in used_tracks:
                    continue
                score = iou_xyxy(box, tr.bbox)
                if score > best_score:
                    best_tid = tid
                    best_score = score
            if best_tid != -1 and best_score >= self.track_iou:
                assigned[i] = best_tid
                used_tracks.add(best_tid)

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
