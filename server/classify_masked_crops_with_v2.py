from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# =========================
# 你要改的地方
# =========================
CLS_MODEL_PATH = r"models/classify_v2.pt"
CROP_DIR = r"debug_crops/masked_crop"
OUT_TXT = r"debug_crops/classify_v2_masked_crop_results.txt"
CLS_IMG_SIZE = 96
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# 如果你想強制用自己的類別順序，可以保留這份 CLASS_NAMES。
# 但 YOLOv8 classify 通常建議優先看 model.names。
CLASS_NAMES = [
    '1m', '1p', '1s',
    '2m', '2p', '2s',
    '3m', '3p', '3s',
    '4m', '4p', '4s',
    '5m', '5p', '5s',
    '6m', '6p', '6s',
    '7m', '7p', '7s',
    '8m', '8p', '8s',
    '9m', '9p', '9s',
    'east', 'flower', 'green', 'north', 'red', 'south', 'west', 'white'
]


def get_label(result, pred_idx: int):
    """優先用 YOLO 模型內建 names，避免手寫 CLASS_NAMES 順序跟訓練順序不一致。"""
    if hasattr(result, "names") and result.names is not None:
        if isinstance(result.names, dict):
            return result.names.get(pred_idx, str(pred_idx))
        if isinstance(result.names, list) and 0 <= pred_idx < len(result.names):
            return result.names[pred_idx]

    if 0 <= pred_idx < len(CLASS_NAMES):
        return CLASS_NAMES[pred_idx]

    return str(pred_idx)


def classify_one(model: YOLO, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    results = model.predict(
        source=img,
        imgsz=CLS_IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    probs = result.probs
    if probs is None:
        return {
            "file": img_path.name,
            "label": "unknown",
            "score": 0.0,
            "top5": []
        }

    pred_idx = int(probs.top1)
    score = float(probs.top1conf)
    label = get_label(result, pred_idx)

    top5 = []
    if probs.top5 is not None:
        for idx, conf in zip(probs.top5, probs.top5conf):
            idx = int(idx)
            conf = float(conf)
            top5.append((get_label(result, idx), conf))

    return {
        "file": img_path.name,
        "label": label,
        "score": score,
        "top5": top5
    }


def main():
    model_path = Path(CLS_MODEL_PATH)
    crop_dir = Path(CROP_DIR)
    out_txt = Path(OUT_TXT)

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 classify_v2 模型：{model_path.resolve()}")

    if not crop_dir.exists():
        raise FileNotFoundError(f"找不到 masked crop 資料夾：{crop_dir.resolve()}")

    image_paths = sorted([
        p for p in crop_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    if not image_paths:
        print(f"{crop_dir} 裡面沒有圖片")
        return

    model = YOLO(str(model_path))

    print("\n========== classify_v2 對 masked_crop 的預測結果 ==========")
    print(f"模型：{model_path}")
    print(f"資料夾：{crop_dir}\n")

    lines = []
    lines.append("========== classify_v2 對 masked_crop 的預測結果 ==========")
    lines.append(f"模型：{model_path}")
    lines.append(f"資料夾：{crop_dir}")
    lines.append("")

    for img_path in image_paths:
        item = classify_one(model, img_path)
        if item is None:
            continue

        line = f'{item["file"]}: {item["label"]}({item["score"]:.2f})'
        print(line)
        lines.append(line)

        if item["top5"]:
            top5_text = ", ".join([f"{label}({conf:.2f})" for label, conf in item["top5"]])
            top5_line = f"    top5: {top5_text}"
            print(top5_line)
            lines.append(top5_line)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n結果已輸出：{out_txt}")


if __name__ == "__main__":
    main()
