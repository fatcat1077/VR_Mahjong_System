from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Box = Tuple[int, int, int, int]


def decode_jpg(jpg: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR image (OpenCV)."""
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG.")
    return img


def _load_label_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [ln for ln in lines if ln]
    return lines if lines else None


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Support common checkpoint formats."""
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
    new_state: Dict[str, torch.Tensor] = {}
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
    """Torch MobileNetV3 Small classifier used as fallback."""

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


class VisionPipeline:
    """Detection + classification wrapper."""

    def __init__(
        self,
        yolo_path: str,
        cls_path: str,
        det_imgsz: int,
        det_conf: float,
        det_iou: float,
        cls_imgsz: int,
        crop_pad: float,
        device: Optional[str] = None,
        cls_labels_path: Optional[str] = None,
        cls_nc: Optional[int] = None,
    ):
        self.det_imgsz = int(det_imgsz)
        self.det_conf = float(det_conf)
        self.det_iou = float(det_iou)
        self.cls_imgsz = int(cls_imgsz)
        self.crop_pad = float(crop_pad)
        self.device = device  # for ultralytics.predict

        # torch device for fallback classifier
        if device is None:
            self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            if device.startswith("cuda") and torch.cuda.is_available():
                self.torch_device = torch.device(device)
            else:
                self.torch_device = torch.device("cpu")

        self.det_model = YOLO(yolo_path)

        self.cls_backend = "torch"
        self.cls_model = None
        self.torch_cls: Optional[TorchMBV3SmallClassifier] = None

        labels = _load_label_list(cls_labels_path)

        # Try Ultralytics classify model first; fallback to Torch MBV3 Small
        try:
            tmp = YOLO(cls_path)
            if getattr(tmp, "task", None) == "classify":
                self.cls_backend = "ultralytics"
                self.cls_model = tmp
                print("[CLS] Using Ultralytics classify model.")
            else:
                raise RuntimeError(f"Ultralytics model task={getattr(tmp, 'task', None)} (not classify)")
        except Exception as e:
            print(f"[CLS] Ultralytics load failed or not classify -> use Torch MobileNetV3 Small. Reason: {e}")
            self.torch_cls = TorchMBV3SmallClassifier(
                weights_path=cls_path,
                device=self.torch_device,
                img_size=self.cls_imgsz,
                labels=labels,
                override_num_classes=cls_nc,
            )

        self._warned_no_probs = False

    def expand_box(self, box, w: int, h: int) -> Optional[Box]:
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

    def classify_crop(self, crop_bgr: np.ndarray):
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

    def det_and_cls(self, frame_bgr: np.ndarray, stop_event=None):
        """Run detection, then crop+classification.

        Returns: (valid_boxes_xyxy, valid_names, valid_confs)
        where boxes are expanded crop boxes.
        """
        h, w = frame_bgr.shape[:2]

        det_res = self.det_model.predict(
            frame_bgr,
            imgsz=self.det_imgsz,
            conf=self.det_conf,
            iou=self.det_iou,
            device=self.device,
            verbose=False,
        )[0]

        if det_res.boxes is None or len(det_res.boxes) == 0:
            return [], [], [], (w, h)

        boxes = det_res.boxes.xyxy.cpu().numpy()

        cls_names: List[str] = []
        cls_confs: List[float] = []
        crops_xyxy: List[Optional[Box]] = []

        for b in boxes:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Stopped by user.")

            eb = self.expand_box(b, w, h)
            if eb is None:
                crops_xyxy.append(None)
                cls_names.append("")
                cls_confs.append(0.0)
                continue

            crops_xyxy.append(eb)
            x1, y1, x2, y2 = eb
            crop = frame_bgr[y1:y2, x1:x2]

            cname, cconf = self.classify_crop(crop)
            cls_names.append(cname)
            cls_confs.append(float(cconf))

        valid_boxes: List[Box] = []
        valid_names: List[str] = []
        valid_confs: List[float] = []
        for eb, name, conf in zip(crops_xyxy, cls_names, cls_confs):
            if eb is None:
                continue
            valid_boxes.append(eb)
            valid_names.append(name)
            valid_confs.append(conf)

        return valid_boxes, valid_names, valid_confs, (w, h)
