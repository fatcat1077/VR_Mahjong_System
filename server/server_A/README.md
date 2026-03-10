# server_A (方案 A 拆分版)

這是由原本的 `server.py` 拆分成 5 個模組：

- `main.py`：程式入口、argparse、TCP server loop、整合推論 + 追蹤 + 麻將建議
- `netio.py`：length-prefixed TCP 封包收發工具
- `vision.py`：YOLO 偵測 + 分類器（Ultralytics classify 或 Torch MobileNetV3 fallback）
- `tracking.py`：IOU-based tracking + short-horizon smoothing
- `mahjong.py`：牌種 label/id 轉換、obs34、heuristic、（可選）SB3 PPO 載入與建議

## 執行方式

在此資料夾內執行：

```bash
python main.py --yolo /path/to/det.pt --cls /path/to/cls.pt --host 0.0.0.0 --port 5000
```

其他參數與原本 `server.py` 相同。
