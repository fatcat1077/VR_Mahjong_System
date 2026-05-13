from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# =========================
# 路徑設定
# =========================

SEG_MODEL_PATH = r"models/segmentation.pt"
CLS_MODEL_PATH = r"models/classify_v2.pt"

IMAGE_PATH = r"sample.png"

# 只輸出一張到目前資料夾
OUTPUT_IMAGE_PATH = Path("output.png")

# =========================
# 基本設定
# =========================

SEG_IMGSZ = 640
CLS_IMGSZ = 96
CONF_THRES = 0.25

# 是否使用 mask crop 丟進 classification
USE_MASKED_CROP = True

# =========================
# 重要：修正手牌 / 桌上牌搞混
# =========================
# 如果你的 segmentation 模型 names 正確，例如：
# {0: 'hand_tile', 1: 'table_tile'}
# 程式會自動判斷。
#
# 如果 names 不正確，或還是搞混，
# 就把 FORCE_CLASS_ID_MAPPING 改成 True，並手動設定下面兩個 ID。

FORCE_CLASS_ID_MAPPING = True

# 從你目前的結果看起來，大機率應該是這樣：
HAND_TILE_CLASS_ID = 0
TABLE_TILE_CLASS_ID = 1

# 如果還是反了，就改回：
# HAND_TILE_CLASS_ID = 1
# TABLE_TILE_CLASS_ID = 0

# =========================
# 載入模型
# =========================

seg_model = YOLO(SEG_MODEL_PATH)
cls_model = YOLO(CLS_MODEL_PATH)

# =========================
# 讀取原始照片
# =========================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"讀不到圖片：{IMAGE_PATH}")

original = image.copy()
draw_img = image.copy()

h, w = original.shape[:2]

# =========================
# 根據 segmentation 類別 ID 判斷是手牌還是桌上牌
# =========================

def get_tile_area_type(seg_cls_id):
    """
    回傳：
    'hand'  = 手牌
    'table' = 桌上牌
    None    = 其他類別，不處理
    """

    # 方式一：強制使用你手動設定的 class ID
    if FORCE_CLASS_ID_MAPPING:
        if seg_cls_id == HAND_TILE_CLASS_ID:
            return "hand"
        elif seg_cls_id == TABLE_TILE_CLASS_ID:
            return "table"
        else:
            return None

    # 方式二：自動讀 seg_model.names
    class_name = str(seg_model.names[seg_cls_id]).lower()

    if "hand" in class_name:
        return "hand"
    elif "table" in class_name:
        return "table"
    else:
        return None

# =========================
# 呼叫 segmentation
# =========================

seg_results = seg_model.predict(
    source=original,
    imgsz=SEG_IMGSZ,
    conf=CONF_THRES,
    save=False,
    verbose=False
)

seg_result = seg_results[0]

if seg_result.boxes is None or len(seg_result.boxes) == 0:
    print("沒有偵測到任何牌")
    cv2.imwrite(str(OUTPUT_IMAGE_PATH), draw_img)
    exit()

boxes = seg_result.boxes.xyxy.cpu().numpy()
classes = seg_result.boxes.cls.cpu().numpy().astype(int)
confs = seg_result.boxes.conf.cpu().numpy()

masks = None
if seg_result.masks is not None:
    masks = seg_result.masks.data.cpu().numpy()

# =========================
# 儲存結果
# =========================

hand_results = []
table_results = []

# =========================
# 處理每一張牌
# =========================

for i, box in enumerate(boxes):
    seg_cls_id = int(classes[i])

    tile_area_type = get_tile_area_type(seg_cls_id)

    # 不是 hand/table 就跳過
    if tile_area_type is None:
        continue

    x1, y1, x2, y2 = box.astype(int)

    # 防止超出圖片邊界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        continue

    crop = original[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # =========================
    # 使用 segmentation mask 裁切
    # =========================

    if USE_MASKED_CROP and masks is not None:
        mask = masks[i]

        # mask resize 回原圖大小
        mask = cv2.resize(mask, (w, h))

        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        masked_image = cv2.bitwise_and(
            original,
            original,
            mask=mask_binary
        )

        crop = masked_image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

    # =========================
    # classification：resize 成 96 x 96
    # =========================

    crop_96 = cv2.resize(crop, (CLS_IMGSZ, CLS_IMGSZ))

    cls_results = cls_model.predict(
        source=crop_96,
        imgsz=CLS_IMGSZ,
        save=False,
        verbose=False
    )

    cls_result = cls_results[0]

    pred_id = int(cls_result.probs.top1)
    pred_conf = float(cls_result.probs.top1conf)
    pred_name = cls_model.names[pred_id]

    result_item = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "pred_name": pred_name,
        "pred_conf": pred_conf,
        "seg_conf": float(confs[i]),
        "seg_cls_id": seg_cls_id
    }

    if tile_area_type == "hand":
        hand_results.append(result_item)
    elif tile_area_type == "table":
        table_results.append(result_item)

# =========================
# 排序
# =========================
# 手牌：由左到右
# 桌上牌：由上到下、由左到右

hand_results.sort(key=lambda item: item["x1"])
table_results.sort(key=lambda item: (item["y1"], item["x1"]))

# =========================
# 清楚版畫框與文字
# =========================

def draw_prediction(img, item, box_color):
    x1 = item["x1"]
    y1 = item["y1"]
    x2 = item["x2"]
    y2 = item["y2"]

    pred_name = str(item["pred_name"])

    # 圖片上只顯示牌名，避免太擠
    label = pred_name

    # 畫框
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        box_color,
        3
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    padding = 6

    text_size, _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    text_w, text_h = text_size

    img_h, img_w = img.shape[:2]

    # 預設文字放在框上方
    bg_x1 = x1
    bg_y1 = y1 - text_h - padding * 2
    bg_x2 = x1 + text_w + padding * 2
    bg_y2 = y1

    text_x = x1 + padding
    text_y = y1 - padding

    # 如果超出上方，就改放到框內
    if bg_y1 < 0:
        bg_y1 = y1
        bg_y2 = y1 + text_h + padding * 2
        text_y = y1 + text_h + padding

    # 如果超出右邊，就往左移
    if bg_x2 > img_w:
        bg_x2 = img_w
        bg_x1 = max(0, bg_x2 - text_w - padding * 2)
        text_x = bg_x1 + padding

    # 白底
    cv2.rectangle(
        img,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        (255, 255, 255),
        -1
    )

    # 黑框
    cv2.rectangle(
        img,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        (0, 0, 0),
        1
    )

    # 黑字
    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

# =========================
# 畫出手牌與桌上牌
# =========================

# 手牌：綠色框
for item in hand_results:
    draw_prediction(
        draw_img,
        item,
        box_color=(0, 255, 0)
    )

# 桌上牌：藍色框
for item in table_results:
    draw_prediction(
        draw_img,
        item,
        box_color=(255, 0, 0)
    )

# =========================
# 只輸出一張 output.png
# =========================

cv2.imwrite(str(OUTPUT_IMAGE_PATH), draw_img)

# =========================
# Terminal 輸出結果
# =========================

print("手牌預測結果：")
for idx, item in enumerate(hand_results, start=1):
    print(f"hand_tile_{idx}: {item['pred_name']}  confidence={item['pred_conf']:.2f}")

print()

print("桌上牌預測結果：")
for idx, item in enumerate(table_results, start=1):
    print(f"table_tile_{idx}: {item['pred_name']}  confidence={item['pred_conf']:.2f}")

print()
print(f"預測照片已輸出：{OUTPUT_IMAGE_PATH}")