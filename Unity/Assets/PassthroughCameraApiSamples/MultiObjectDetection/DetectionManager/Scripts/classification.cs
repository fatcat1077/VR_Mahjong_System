// MahjongClassifierRunManager.cs
// 簡單封裝一個「牌面分類模型」，外部只要呼叫 RunInference + GetLastResult 即可。

using System;
using Meta.XR.Samples;
using UnityEngine;
using Unity.Sentis;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("MahjongClassifierRunManager")]
    public class MahjongClassifierRunManager : MonoBehaviour
    {
        [Header("Classifier Model Config")]
        [Tooltip("你的分類模型（.onnx 轉成的 ModelAsset 或 .sentis）")]
        [SerializeField] private ModelAsset classifierModel;

        [Tooltip("一行一個 label，例如：1m,2m,3m,白,發,中...")]
        [SerializeField] private TextAsset labelFile;

        [Tooltip("分類模型的輸入影像大小（假設是 NCHW: 1x3xHxW，這裡的 H=W=inputSize）")]
        [SerializeField] private int inputSize = 64;

        [Tooltip("要跑在 CPU 還是 GPUCompute")]
        [SerializeField] private BackendType backend = BackendType.CPU;

        [Header("Debug")]
        [Tooltip("是否在每次推論時輸出 debug log")]
        [SerializeField] private bool debugLog = true;

        private Worker _worker;              // Sentis 2.x：IWorker -> Worker
        private string[] _labels;
        private string _lastResult = "UNKNOWN";

        public bool IsModelLoaded { get; private set; } = false;

        // -------------------
        //  Unity lifecycle
        // -------------------
        private void Awake()
        {
            if (classifierModel == null)
            {
                Debug.LogError("[MahjongClassifier] classifierModel is not assigned.");
                return;
            }

            // 1. 載入模型
            var runtimeModel = ModelLoader.Load(classifierModel);

            // 2. 建立 Worker（取代舊版的 WorkerFactory.CreateWorker）
            _worker = new Worker(runtimeModel, backend);

            // 3. 讀取 label
            if (labelFile != null)
            {
                _labels = labelFile.text.Split('\n');
            }
            else
            {
                _labels = Array.Empty<string>();
                Debug.LogWarning("[MahjongClassifier] labelFile is not assigned, results will just be indices.");
            }

            IsModelLoaded = true;

            if (debugLog)
            {
                Debug.Log("[MahjongClassifier] Model loaded and worker created.");
            }
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
            _worker = null;
        }

        // -------------------
        //  Public API
        // -------------------

        /// <summary>
        /// 對一張牌的貼圖做分類推論。
        /// </summary>
        public void RunInference(Texture2D tileImage)
        {
            if (!IsModelLoaded || _worker == null)
            {
                if (debugLog)
                    Debug.LogWarning("[MahjongClassifier] Model not loaded yet.");
                _lastResult = "UNKNOWN";
                return;
            }

            if (tileImage == null)
            {
                if (debugLog)
                    Debug.LogWarning("[MahjongClassifier] tileImage is null.");
                _lastResult = "UNKNOWN";
                return;
            }

            try
            {
                // 1. 把 Texture2D 轉成 tensor（CPU 上的 NCHW float tensor）
                using var inputTensor = Preprocess(tileImage);

                if (debugLog)
                {
                    Debug.Log($"[MahjongClassifier] RunInference on tile {tileImage.width}x{tileImage.height}, " +
                              $"tensor shape = {inputTensor.shape}");
                }

                // 2. 跑模型：Sentis 2.x 用 Schedule() 取代 Execute()
                _worker.Schedule(inputTensor);

                // 3. 取出輸出（假設輸出是 1D logits / probabilities）
                var outputTensor = _worker.PeekOutput() as Tensor<float>;
                if (outputTensor == null)
                {
                    if (debugLog)
                        Debug.LogWarning("[MahjongClassifier] Output tensor is null.");
                    _lastResult = "UNKNOWN";
                    return;
                }

                var data = outputTensor.DownloadToArray();
                int bestIndex = ArgMax(data);

                if (_labels != null && bestIndex >= 0 && bestIndex < _labels.Length)
                {
                    _lastResult = _labels[bestIndex].Trim();
                }
                else
                {
                    _lastResult = bestIndex.ToString();
                }

                // ★ Debug：每次推論後印出結果
                if (debugLog)
                {
                    Debug.Log($"[MahjongClassifier] Predicted: {_lastResult} (index {bestIndex})");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[MahjongClassifier] RunInference error: {e}");
                _lastResult = "UNKNOWN";
            }
        }

        /// <summary>
        /// 回傳上一張圖片的分類結果（牌名）。
        /// </summary>
        public string GetLastResult()
        {
            return _lastResult;
        }

        // -------------------
        //  Helper functions
        // -------------------

        /// <summary>
        /// 把輸入貼圖縮放成 inputSize×inputSize，再轉成 NCHW tensor（1x3xH xW）。
        /// 完全使用 CPU，不用 Graphics.ConvertTexture，避免 Quest 上的相容性問題。
        /// </summary>
        private Tensor<float> Preprocess(Texture2D tex)
        {
            // 來源大小
            int srcW = tex.width;
            int srcH = tex.height;

            // 讀出原圖像素（我們在 DetectionManager 自己 new 出來的 Texture2D 是可讀取的）
            Color[] srcPixels = tex.GetPixels();

            int dstW = inputSize;
            int dstH = inputSize;
            int c = 3;

            // 目標 tensor
            var shape = new TensorShape(1, c, dstH, dstW);
            var tensor = new Tensor<float>(shape, clearOnInit: false);

            // 最近鄰縮放：把 dst 每個像素對應回 src 的一個像素
            float xRatio = (float)srcW / dstW;
            float yRatio = (float)srcH / dstH;

            for (int y = 0; y < dstH; y++)
            {
                int sy = Mathf.Min((int)(y * yRatio), srcH - 1);
                for (int x = 0; x < dstW; x++)
                {
                    int sx = Mathf.Min((int)(x * xRatio), srcW - 1);

                    int srcIndex = sy * srcW + sx;
                    Color p = srcPixels[srcIndex];

                    tensor[0, 0, y, x] = p.r;   // 如有需要可在這裡做 normalize
                    tensor[0, 1, y, x] = p.g;
                    tensor[0, 2, y, x] = p.b;
                }
            }

            return tensor;
        }

        private int ArgMax(float[] data)
        {
            if (data == null || data.Length == 0)
                return -1;

            int best = 0;
            float bestVal = data[0];

            for (int i = 1; i < data.Length; i++)
            {
                if (data[i] > bestVal)
                {
                    bestVal = data[i];
                    best = i;
                }
            }
            return best;
        }
    }
}
