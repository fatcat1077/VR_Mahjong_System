using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Meta.XR.Samples;
using System.IO;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    // 資料集自動儲存器：每隔一段時間，把當前畫面 + BoxDrawn 存起來
    public class DatasetRecorder : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private WebCamTextureManager webcamManager;      // 連到場景裡的 WebCamTextureManager
        [SerializeField] private SentisInferenceUiManager uiInference;    // 連到 SentisInferenceUiManager（有 BoxDrawn）

        [Header("Settings")]
        [SerializeField] private string folderName = "MahjongDataset";
        [SerializeField] private float saveInterval = 3.0f;              // 每 3秒存一次

        private float _saveTimer = 0f;
        private string _rootPath;
        private int _index = 0;

        private void Start()
        {
            // 存在 Quest 的 persistentDataPath 底下
            _rootPath = Path.Combine(Application.persistentDataPath, folderName);
            if (!Directory.Exists(_rootPath))
            {
                Directory.CreateDirectory(_rootPath);
            }

            Debug.Log("[DatasetRecorder] Save Path: " + _rootPath);
        }

        private void Update()
        {
            _saveTimer += Time.deltaTime;

            if (_saveTimer >= saveInterval)
            {
                _saveTimer = 0f;
                CaptureCurrentFrame();
            }
        }

        /// <summary>
        /// 把當下這一幀的畫面 + BoxDrawn 存成一組檔案
        /// </summary>
        public void CaptureCurrentFrame()
        {
            if (webcamManager == null || webcamManager.WebCamTexture == null)
            {
                Debug.LogWarning("[DatasetRecorder] No WebCamTexture.");
                return;
            }

            if (uiInference == null || uiInference.BoxDrawn == null)
            {
                Debug.LogWarning("[DatasetRecorder] uiInference or BoxDrawn is null.");
                return;
            }

            var camTex = webcamManager.WebCamTexture;
            int w = camTex.width;
            int h = camTex.height;

            // 1. 把 WebCamTexture 轉成 Texture2D
            Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.SetPixels(camTex.GetPixels());
            tex.Apply();

            // 2. 存 PNG
            string idx = _index.ToString("D5");   // 00000, 00001, ...
            string imgPath = Path.Combine(_rootPath, $"img_{idx}.png");
            byte[] pngBytes = tex.EncodeToPNG();
            File.WriteAllBytes(imgPath, pngBytes);
            Debug.Log("[DatasetRecorder] Saved Image: " + imgPath);

            Destroy(tex);

            // 3. 存標註（簡單 txt 格式）
            string annoPath = Path.Combine(_rootPath, $"img_{idx}.txt");
            SaveAnnotation(annoPath, uiInference.BoxDrawn);
            Debug.Log("[DatasetRecorder] Saved Annotation: " + annoPath);

            _index++;
        }

        private void SaveAnnotation(
            string filePath,
            List<SentisInferenceUiManager.BoundingBox> boxes)
        {
            using (var writer = new StreamWriter(filePath))
            {
                foreach (var b in boxes)
                {
                    // 先存 className + 目前 UI 上的 box 資訊
                    writer.WriteLine($"{b.ClassName} {b.CenterX} {b.CenterY} {b.Width} {b.Height}");
                }
            }
        }
    }
}

