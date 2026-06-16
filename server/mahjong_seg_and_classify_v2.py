from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# =========================
# 你要改的地方
# =========================
SEG_MODEL_PATH = r"models\segmentation.pt"

# 改成你訓練好的 YOLOv8 classification 模型
# 常見位置例如：runs\classify\train\weights\best.pt
CLS_MODEL_PATH = r"models\classify_v2.pt"

IMAGE_PATH = "sample.png"
LABELS_PATH = "mahjong_labels.txt"

# 是否輸出 crop 圖片
SAVE_DEBUG_CROPS = True
DEBUG_CROP_DIR = "debug_crops"

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

CONF_THRES = 0.25
IMG_SIZE = 640
CLS_IMG_SIZE = 224
DEVICE = 0 if torch.cuda.is_available() else "cpu"
NUM_CLASSES = len(CLASS_NAMES)


# =========================
# 工具函式
# =========================
def polygon_center(poly: np.ndarray):
    return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))


def safe_filename(text: str):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def prepare_debug_dirs():
    """
    建立 debug_crops 資料夾，並清掉舊 crop，避免新舊結果混在一起。
    """
    root = Path(DEBUG_CROP_DIR)
    masked_dir = root / "masked_crop"
    bbox_dir = root / "bbox_crop"
    mask_dir = root / "mask"

    for d in [masked_dir, bbox_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for old_file in d.glob("*.png"):
            old_file.unlink()

    return root, masked_dir, bbox_dir, mask_dir


def crop_from_polygon(image: np.ndarray, polygon: np.ndarray):
    """
    回傳三種資訊：
    1. masked_crop：使用 segmentation mask 後的 crop，mask 外會變黑
    2. bbox_crop：單純用 bounding box 裁切，保留原本背景
    3. crop_mask：該 crop 區域的 mask 圖
    """
    h, w = image.shape[:2]
    polygon = polygon.astype(np.int32)

    if len(polygon) < 3:
        return None, None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    x, y, bw, bh = cv2.boundingRect(polygon)

    # 防止座標超出圖片範圍
    x = max(0, x)
    y = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    if bw <= 2 or bh <= 2:
        return None, None

    masked = cv2.bitwise_and(image, image, mask=mask)

    masked_crop = masked[y:y + bh, x:x + bw]
    bbox_crop = image[y:y + bh, x:x + bw]
    crop_mask = mask[y:y + bh, x:x + bw]

    info = {
        "x": x,
        "y": y,
        "w": bw,
        "h": bh,
        "crop_mask": crop_mask,
        "bbox_crop": bbox_crop,
    }

    # 注意：目前分類仍沿用 masked_crop，跟你原本版本一樣
    return masked_crop, info


def save_debug_crop_files(i: int, label: str, score: float, crop: np.ndarray, info: dict, dirs):
    """
    把每個 segmentation 切出來的 crop 存成圖片，方便你檢查。
    """
    if not SAVE_DEBUG_CROPS:
        return

    _, masked_dir, bbox_dir, mask_dir = dirs

    x = info["x"]
    y = info["y"]
    bw = info["w"]
    bh = info["h"]
    crop_mask = info["crop_mask"]
    bbox_crop = info["bbox_crop"]

    label_safe = safe_filename(label)
    filename_base = f"{i:03d}_{label_safe}_{score:.2f}_x{x}_y{y}_w{bw}_h{bh}"

    cv2.imwrite(str(masked_dir / f"{filename_base}_masked.png"), crop)
    cv2.imwrite(str(bbox_dir / f"{filename_base}_bbox.png"), bbox_crop)
    cv2.imwrite(str(mask_dir / f"{filename_base}_mask.png"), crop_mask)


def load_yolov8_classifier(model_path: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 YOLOv8 classify 模型: {model_path}")

    model = YOLO(str(model_path))
    return model


def classify_crop_yolov8(model: YOLO, crop_bgr: np.ndarray):
    """
    使用 YOLOv8 classification 模型分類單張裁切後的麻將牌。
    crop_bgr 是 OpenCV 讀進來的 BGR 圖片，Ultralytics 可以直接吃 numpy image。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    results = model.predict(
        source=crop_bgr,
        imgsz=CLS_IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    probs = result.probs

    if probs is None:
        return "unknown", 0.0

    pred_idx = int(probs.top1)
    score = float(probs.top1conf)

    # 優先使用你自己定義的 CLASS_NAMES，確保輸出順序和你的 35 類一致
    # 如果 pred_idx 超出範圍，就退回 YOLO 模型內建 names
    if 0 <= pred_idx < len(CLASS_NAMES):
        label = CLASS_NAMES[pred_idx]
    else:
        label = result.names.get(pred_idx, str(pred_idx))

    return label, score


def split_hand_and_table(detections, image_h):
    """
    簡單啟發式：
    y 越大越靠近畫面底部，通常是手牌
    """
    if not detections:
        return [], []

    centers_y = [d["cy"] for d in detections]
    median_y = np.median(centers_y)

    hand_tiles = []
    table_tiles = []

    for d in detections:
        if d["cy"] > median_y:
            hand_tiles.append(d)
        else:
            table_tiles.append(d)

    hand_tiles.sort(key=lambda d: d["cx"])
    table_tiles.sort(key=lambda d: (d["cy"], d["cx"]))

    return hand_tiles, table_tiles


# =========================
# 主流程
# =========================
def main():
    if len(CLASS_NAMES) != 35:
        raise ValueError(f"目前 CLASS_NAMES 數量是 {len(CLASS_NAMES)}，但 classify 模型需要 35 類。")

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"找不到圖片: {IMAGE_PATH}")

    h, w = image.shape[:2]

    debug_dirs = prepare_debug_dirs() if SAVE_DEBUG_CROPS else None

    seg_model = YOLO(SEG_MODEL_PATH)
    cls_model = load_yolov8_classifier(CLS_MODEL_PATH)

    results = seg_model.predict(
        source=IMAGE_PATH,
        conf=CONF_THRES,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    if result.masks is None or result.masks.xy is None or len(result.masks.xy) == 0:
        print("沒有偵測到任何 segmentation 物件")
        return

    detections = []
    polygons = [np.array(poly, dtype=np.float32) for poly in result.masks.xy]

    for i, poly in enumerate(polygons):
        crop, info = crop_from_polygon(image, poly)
        if crop is None:
            continue

        label, score = classify_crop_yolov8(cls_model, crop)
        cx, cy = polygon_center(poly)

        if SAVE_DEBUG_CROPS:
            save_debug_crop_files(i, label, score, crop, info, debug_dirs)

        detections.append({
            "id": i,
            "label": label,
            "score": score,
            "cx": cx,
            "cy": cy,
            "crop": crop,
            "poly": poly,
            "bbox_crop": info["bbox_crop"],
            "crop_mask": info["crop_mask"],
            "x": info["x"],
            "y": info["y"],
            "w": info["w"],
            "h": info["h"],
        })

    hand_tiles, table_tiles = split_hand_and_table(detections, h)

    print("\n========== 結果 ==========")
    print("手上的牌：")
    print([f'{d["label"]}({d["score"]:.2f})' for d in hand_tiles])

    print("\n桌上的牌：")
    print([f'{d["label"]}({d["score"]:.2f})' for d in table_tiles])

    if SAVE_DEBUG_CROPS:
        print(f"\n已輸出 crop 檢查資料夾：{DEBUG_CROP_DIR}")
        print(f"masked crop：{DEBUG_CROP_DIR}/masked_crop")
        print(f"bbox crop：{DEBUG_CROP_DIR}/bbox_crop")
        print(f"mask 圖：{DEBUG_CROP_DIR}/mask")

    vis = image.copy()
    for d in detections:
        poly = d["poly"].astype(np.int32)
        color = (0, 255, 0) if d in hand_tiles else (255, 0, 0)
        cv2.polylines(vis, [poly], True, color, 2)
        cv2.putText(
            vis,
            d["label"],
            (int(d["cx"]), int(d["cy"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    out_path = "mahjong_result_preview.png"
    cv2.imwrite(out_path, vis)
    print(f"\n預覽圖已輸出：{out_path}")


if __name__ == "__main__":
    main()
