import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_A_DIR = SCRIPT_DIR / "server_A"
if str(SERVER_A_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_A_DIR))

from vision import VisionPipeline  # noqa: E402


DEFAULT_MODEL_ROOT = Path(r"D:\Download\final_models\final_models")
DEFAULT_YOLO_NAME = "best_segmentation.pt"
DEFAULT_CLS_NAME = "best_classification.pt"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox_to_norm(box, img_w: int, img_h: int) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return {
        "cx": ((x1 + x2) / 2.0) / img_w,
        "cy": ((y1 + y2) / 2.0) / img_h,
        "w": (x2 - x1) / img_w,
        "h": (y2 - y1) / img_h,
    }


def build_prediction(
    sample_id: str,
    frame_bgr,
    existing: Dict[str, Any],
    boxes,
    names,
    confs,
    areas,
) -> Dict[str, Any]:
    img_h, img_w = frame_bgr.shape[:2]
    tiles: List[Dict[str, Any]] = []
    for idx, (box, name, conf, area) in enumerate(zip(boxes, names, confs, areas)):
        x1, y1, x2, y2 = [int(v) for v in box]
        tiles.append(
            {
                "index": idx,
                "track_id": idx,
                "predicted_label": str(name or ""),
                "confidence": float(conf or 0.0),
                "area": str(area or ""),
                "bbox_norm": bbox_to_norm((x1, y1, x2, y2), img_w, img_h),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )

    captured_at = existing.get("captured_at_epoch") or time.time()
    request = dict(existing.get("request") or {})
    request.setdefault("source", "relabel")
    request.setdefault("command", "relabel_segmentation_samples")
    prediction = {
        "schema_version": 2,
        "sample_id": sample_id,
        "captured_at_epoch": captured_at,
        "captured_at_local": existing.get("captured_at_local") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "relabelled_at_epoch": time.time(),
        "relabelled_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request": request,
        "image_file": "original.jpg",
        "preview_file": "preview.jpg",
        "image": {
            "width": img_w,
            "height": img_h,
        },
        "conditions": existing.get("conditions")
        or {
            "distance": "",
            "angle": "",
            "lighting": "",
        },
        "tiles": tiles,
        "hand": existing.get("hand", []),
        "table": existing.get("table", []),
        "debug": {
            **(existing.get("debug", {}) if isinstance(existing.get("debug"), dict) else {}),
            "classification_enabled": True,
            "relabelled_from_original": True,
        },
    }
    return prediction


def write_preview(frame_bgr, prediction: Dict[str, Any], preview_path: Path) -> None:
    vis = frame_bgr.copy()
    colors = {
        "hand": (0, 210, 120),
        "table": (255, 170, 40),
        "": (80, 220, 255),
    }
    for tile in prediction.get("tiles", []):
        bbox = tile.get("bbox_xyxy", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        area = str(tile.get("area", "") or "")
        color = colors.get(area, colors[""])
        text = str(tile.get("index", 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            text,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(preview_path), vis)


def iter_sample_dirs(samples_dir: Path):
    return sorted(
        [
            path
            for path in samples_dir.iterdir()
            if path.is_dir() and (path / "original.jpg").is_file()
        ],
        key=lambda p: p.name,
    )


def relabel_sample(sample_dir: Path, pipeline: VisionPipeline, backup: bool) -> int:
    image_path = sample_dir / "original.jpg"
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    existing = read_json(sample_dir / "prediction.json") or {}
    boxes, names, confs, areas, _ = pipeline.det_and_cls(frame)
    prediction = build_prediction(sample_dir.name, frame, existing, boxes, names, confs, areas)

    prediction_path = sample_dir / "prediction.json"
    if backup and prediction_path.exists():
        backup_path = sample_dir / "prediction.before_relabel.json"
        if not backup_path.exists():
            shutil.copy2(prediction_path, backup_path)

    write_json(prediction_path, prediction)
    write_preview(frame, prediction, sample_dir / "preview.jpg")
    return len(prediction.get("tiles", []))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run saved original.jpg samples through segmentation + classification."
    )
    parser.add_argument("--samples-dir", default=str(SCRIPT_DIR / "segmentation_samples"))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--cls", default=None)
    parser.add_argument("--det-imgsz", type=int, default=960)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--cls-imgsz", type=int, default=128)
    parser.add_argument("--cls-conf", type=float, default=0.5)
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N samples")
    parser.add_argument("--no-backup", action="store_true", help="do not create prediction.before_relabel.json")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    model_root = Path(args.model_root)
    yolo = Path(args.yolo or model_root / DEFAULT_YOLO_NAME)
    cls = Path(args.cls or model_root / DEFAULT_CLS_NAME)
    samples_dir = Path(args.samples_dir)

    if not yolo.is_file():
        raise FileNotFoundError(f"missing segmentation model: {yolo}")
    if not cls.is_file():
        raise FileNotFoundError(f"missing classification model: {cls}")
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"missing samples dir: {samples_dir}")

    pipeline = VisionPipeline(
        yolo_path=str(yolo),
        cls_path=str(cls),
        det_imgsz=args.det_imgsz,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        cls_imgsz=args.cls_imgsz,
        crop_pad=args.crop_pad,
        device=args.device,
        cls_conf_threshold=args.cls_conf,
    )

    sample_dirs = iter_sample_dirs(samples_dir)
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]

    print(f"[Relabel] samples={len(sample_dirs)} dir={samples_dir}", flush=True)
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        count = relabel_sample(sample_dir, pipeline, backup=not args.no_backup)
        print(f"[Relabel] {idx}/{len(sample_dirs)} {sample_dir.name}: {count} labels", flush=True)


if __name__ == "__main__":
    main()
