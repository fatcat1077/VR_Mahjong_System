// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections.Generic;
using Meta.XR.Samples;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using PassthroughCameraSamples.StartScene;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceUiManager : MonoBehaviour
    {
        [Header("Placement configureation")]
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;
        private PassthroughCameraEye CameraEye => m_webCamTextureManager.Eye;

        [Header("UI display references")]
        [SerializeField] private SentisObjectDetectedUiManager m_detectionCanvas;
        [SerializeField] private RawImage m_displayImage;
        [SerializeField] private Sprite m_boxTexture;
        [SerializeField] private Color m_boxColor;
        [SerializeField] private Font m_font;
        [SerializeField] private Color m_fontColor;
        [SerializeField] private int m_fontSize = 80;

        [Header("Optional: show hand text (leave null if not used)")]
        [SerializeField] private Text m_handText;

        [Header("Mahjong Advice UI (optional)")]
        [SerializeField] private Text m_adviceTitleText;
        [SerializeField] private Text m_adviceTileText;
        [SerializeField] private Text m_adviceReasonText;
        [SerializeField] private Button m_benefitButton;
        [SerializeField] private Button m_safeButton;

        [Header("Mahjong Prompt Panel (DebugUIBuilder, optional)")]
        [SerializeField] private bool m_useDebugPromptPanel = true;
        [Tooltip("避免 StartScene 也顯示一次：若勾選，只有在指定場景名稱時才會建立 DebugUIBuilder 的大框框。")]
        [SerializeField] private bool m_onlyBuildPromptInSpecificScene = true;
        [SerializeField] private string m_promptAllowedSceneName = "MultiObjectDetection";
        [SerializeField] private int m_promptPane = DebugUIBuilder.DEBUG_PANE_CENTER;
        [SerializeField] private string m_promptTitle = "麻將小幫手";
        [SerializeField] private string m_promptSubtitle = "請選擇你要使用的模式：";
        [SerializeField] private string m_offenseButtonLabel = "切換進攻模式";
        [SerializeField] private string m_defenseButtonLabel = "切換防守模式";
        [Tooltip("回覆框高度 = 按鈕高度 * 倍率（建議 4~7）")]
        [SerializeField] private float m_promptBoxHeightMultiplier = 6f;
        [Tooltip("回覆框內距 (px)")]
        [SerializeField] private float m_promptPadding = 12f;
        [Tooltip("若要強制更大字可填，例如 40；0 = 自動用按鈕字體大小")]
        [SerializeField] private int m_promptFontSizeOverride = 0;
        [TextArea(2, 6)]
        [SerializeField] private string m_promptDefaultText = "回覆會顯示在這裡。\n\n請按A按鈕開始辨識。";
        [Tooltip("先不要特別畫框：true=不畫 bounding boxes，只更新提示區塊文字")]
        [SerializeField] private bool m_disableBoxDrawing = true;
        [Tooltip("若 m_disableBoxDrawing=true，是否自動把偵測 canvas 隱藏")]
        [SerializeField] private bool m_hideDetectionCanvasWhenPromptOnly = false;


        [Space(10)]
        public UnityEvent<int> OnObjectsDetected;

        public List<BoundingBox> BoxDrawn = new();

        private string[] m_labels;
        private List<GameObject> m_boxPool = new();
        private Transform m_displayLocation;

        // DebugUIBuilder prompt panel runtime refs
        private bool _promptBuilt = false;
        private Text _promptStatusText;
        private Text _promptAnswerText;
        private bool _isConnected = false;
        private string[] _latestHand;


        // =========================
        // Remote JSON tile struct
        // (field names must match python json)
        // =========================
        [Serializable]
        public struct RemoteTile
        {
            public int id;
            public string cls;
            public float conf;
            public float cx;   // 0..1
            public float cy;   // 0..1 (top=0, bottom=1)
            public float w;    // 0..1
            public float h;    // 0..1
        }

        // =========================
        // Remote JSON advice structs
        // (field names must match python json)
        // =========================
        [Serializable]
        public class AdviceItem
        {
            public int tile_id;
            public string tile;
            public string source;
            public string reason;
        }

        [Serializable]
        public class Advice
        {
            public AdviceItem benefit;
            public AdviceItem safe;
        }

        private Advice _latestAdvice;
        private enum AdviceMode { Benefit, Safe }
        private AdviceMode _mode = AdviceMode.Benefit;

        //bounding box data
        public struct BoundingBox
        {
            public float CenterX;
            public float CenterY;
            public float Width;
            public float Height;
            public string Label;
            public Vector3? WorldPos;
            public string ClassName;
        }

        #region Unity Functions
        private void Start()
        {
            // UI boxes are anchored under the RawImage transform. If you only use the prompt panel, RawImage can be null.
            m_displayLocation = (m_displayImage != null) ? m_displayImage.transform : transform;

            // Hook scene buttons (optional)
            if (m_benefitButton != null) m_benefitButton.onClick.AddListener(OnClickBenefit);
            if (m_safeButton != null) m_safeButton.onClick.AddListener(OnClickSafe);

            // Build prompt panel (if the scene has DebugUIBuilder)
            // NOTE: 你說「StartScene 不要再顯示一次」，所以這裡加上「只在 MultiObjectDetection 場景建立」的保護。
            TryBuildPromptPanel();

            // Default status + hint
            SetStatusText("狀態：待命");
            SetPromptText(m_promptDefaultText);

            // default view
            UpdateAdviceUI();

            // Optional: if we're prompt-only, hide canvas to mimic the StartScene panel look
            if (m_hideDetectionCanvasWhenPromptOnly && m_detectionCanvas != null && m_disableBoxDrawing)
            {
                m_detectionCanvas.SetCanvasVisible(false);
            }
        }
        #endregion

        #region Detection Functions
        public void OnObjectDetectionError()
        {
            // Clear current boxes
            ClearAnnotations();

            // clear hand text
            if (m_handText != null) m_handText.text = "";

            // clear advice
            _latestAdvice = null;
            if (m_adviceTitleText != null) m_adviceTitleText.text = "";
            if (m_adviceTileText != null) m_adviceTileText.text = "";
            if (m_adviceReasonText != null) m_adviceReasonText.text = "";

            SetStatusText("狀態：待命");
            SetPromptText(m_promptDefaultText);

            // Set object found to 0
            OnObjectsDetected?.Invoke(0);
        }

        // 讓外部（例如 DetectionManager）在更新 BoxDrawn 之後，重新把文字套回 UI
        public void RefreshBoxLabels(bool useClassNameOnly = true)
        {
            // 依序走過每個框，更新對應 panel 上的 Text
            for (int i = 0; i < BoxDrawn.Count && i < m_boxPool.Count; i++)
            {
                var panel = m_boxPool[i];
                if (panel == null) continue;

                var text = panel.GetComponentInChildren<Text>();
                if (text == null) continue;

                var box = BoxDrawn[i];

                // 如果要只顯示分類器結果，就用 ClassName；
                // 否則就用 Label
                if (useClassNameOnly && !string.IsNullOrEmpty(box.ClassName))
                {
                    text.text = box.ClassName;
                }
                else
                {
                    text.text = box.Label;
                }
            }
        }
        #endregion

        #region BoundingBoxes functions
        public void SetLabels(TextAsset labelsAsset)
        {
            //Parse neural net labels
            m_labels = labelsAsset != null ? labelsAsset.text.Split('\n') : null;
        }

        public void SetDetectionCapture(Texture image)
        {
            if (m_displayImage != null) m_displayImage.texture = image;
            if (m_detectionCanvas != null) m_detectionCanvas.CapturePosition();
        }

        // 原本 Sentis 用的（保留）
        public void DrawUIBoxes(Tensor<float> output, Tensor<int> labelIDs, float imageWidth, float imageHeight)
        {
            if (m_detectionCanvas != null) m_detectionCanvas.UpdatePosition();
            ClearAnnotations();

            var displayWidth = m_displayImage.rectTransform.rect.width;
            var displayHeight = m_displayImage.rectTransform.rect.height;

            var scaleX = displayWidth / imageWidth;
            var scaleY = displayHeight / imageHeight;

            var halfWidth = displayWidth / 2;
            var halfHeight = displayHeight / 2;

            var boxesFound = output.shape[0];
            if (boxesFound <= 0)
            {
                OnObjectsDetected?.Invoke(0);
                return;
            }
            var maxBoxes = Mathf.Min(boxesFound, 200);

            OnObjectsDetected?.Invoke(maxBoxes);

            var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(CameraEye);
            var camRes = intrinsics.Resolution;

            for (var n = 0; n < maxBoxes; n++)
            {
                var centerX = output[n, 0] * scaleX - halfWidth;
                var centerY = output[n, 1] * scaleY - halfHeight;
                var perX = (centerX + halfWidth) / displayWidth;
                var perY = (centerY + halfHeight) / displayHeight;

                var classname = (m_labels != null && n < labelIDs.shape[0])
                    ? m_labels[labelIDs[n]].Replace(" ", "_")
                    : "unknown";

                var centerPixel = new Vector2Int(
                    Mathf.RoundToInt(perX * camRes.x),
                    Mathf.RoundToInt((1.0f - perY) * camRes.y)
                );
                var ray = PassthroughCameraUtils.ScreenPointToRayInWorld(CameraEye, centerPixel);
                var worldPos = m_environmentRaycast.PlaceGameObjectByScreenPos(ray);

                var box = new BoundingBox
                {
                    CenterX = centerX,
                    CenterY = centerY,
                    ClassName = classname,
                    Width = output[n, 2] * scaleX,
                    Height = output[n, 3] * scaleY,
                    Label = $"Id: {n} Class: {classname}",
                    WorldPos = worldPos,
                };

                BoxDrawn.Add(box);
                DrawBox(box, n);
            }
        }

        // ✅ 新增：PC 回傳 JSON（normalized cx/cy/w/h）
        // - 若 m_disableBoxDrawing=true：不畫框，只更新 hand 文字 + 讓提示區塊可以顯示 server 回覆
        public void DrawRemoteBoxes(RemoteTile[] tiles, int imgW, int imgH, string[] hand = null)
        {
            // Always remember hand for the prompt panel
            _latestHand = hand;

            // prompt-only mode
            if (m_disableBoxDrawing)
            {
                ClearAnnotations();
                OnObjectsDetected?.Invoke(tiles != null ? tiles.Length : 0);

                if (m_hideDetectionCanvasWhenPromptOnly && m_detectionCanvas != null)
                {
                    m_detectionCanvas.SetCanvasVisible(false);
                }

                if (m_handText != null)
                {
                    m_handText.text = (hand != null && hand.Length > 0) ? string.Join(" ", hand) : "";
                }

                return;
            }

            // normal: draw boxes
            if (m_detectionCanvas != null)
            {
                m_detectionCanvas.SetCanvasVisible(true);
            if (m_detectionCanvas != null) m_detectionCanvas.UpdatePosition();
            }
            ClearAnnotations();

            if (tiles == null || tiles.Length == 0)
            {
                OnObjectsDetected?.Invoke(0);
                if (m_handText != null) m_handText.text = (hand != null) ? string.Join(" ", hand) : "";
                return;
            }

            var displayWidth = m_displayImage.rectTransform.rect.width;
            var displayHeight = m_displayImage.rectTransform.rect.height;
            var halfWidth = displayWidth / 2f;
            var halfHeight = displayHeight / 2f;

            var maxBoxes = Mathf.Min(tiles.Length, 200);
            OnObjectsDetected?.Invoke(maxBoxes);

            var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(CameraEye);
            var camRes = intrinsics.Resolution;

            for (int n = 0; n < maxBoxes; n++)
            {
                var t = tiles[n];

                float perX = Mathf.Clamp01(t.cx);
                float perY = Mathf.Clamp01(t.cy);
                float bw = Mathf.Clamp01(t.w);
                float bh = Mathf.Clamp01(t.h);

                var centerX = perX * displayWidth - halfWidth;
                var centerY = perY * displayHeight - halfHeight;

                var width = bw * displayWidth;
                var height = bh * displayHeight;

                var classname = string.IsNullOrEmpty(t.cls) ? "UNKNOWN" : t.cls.Replace(" ", "_");

                // 注意：沿用原本邏輯，perY 需要做 1 - perY 才符合 ScreenPointToRay 的座標系
                var centerPixel = new Vector2Int(
                    Mathf.RoundToInt(perX * camRes.x),
                    Mathf.RoundToInt((1.0f - perY) * camRes.y)
                );
                var ray = PassthroughCameraUtils.ScreenPointToRayInWorld(CameraEye, centerPixel);
                var worldPos = (m_environmentRaycast != null)
                    ? m_environmentRaycast.PlaceGameObjectByScreenPos(ray)
                    : null;

                var box = new BoundingBox
                {
                    CenterX = centerX,
                    CenterY = centerY,
                    ClassName = classname,
                    Width = width,
                    Height = height,
                    Label = $"{classname} {t.conf:0.00}",
                    WorldPos = worldPos,
                };

                BoxDrawn.Add(box);
                DrawBox(box, n);
            }

            if (m_handText != null)
            {
                m_handText.text = (hand != null && hand.Length > 0) ? string.Join(" ", hand) : "";
            }
        }

        // =========================
        // =========================
        // Mahjong advice UI
        // =========================
        public void SetAdvice(Advice advice)
        {
            _latestAdvice = advice;
            UpdateAdviceUI();
        }

        public void OnClickBenefit()
        {
            _mode = AdviceMode.Benefit;
            UpdateAdviceUI();
        }

        public void OnClickSafe()
        {
            _mode = AdviceMode.Safe;
            UpdateAdviceUI();
        }

        private void UpdateAdviceUI()
        {
            if (m_adviceTitleText == null && m_adviceTileText == null && m_adviceReasonText == null)
                return;

            if (_latestAdvice == null)
            {
                if (m_adviceTitleText != null) m_adviceTitleText.text = "";
                if (m_adviceTileText != null) m_adviceTileText.text = "";
                if (m_adviceReasonText != null) m_adviceReasonText.text = "";
                return;
            }

            AdviceItem item = null;
            string title = "";

            if (_mode == AdviceMode.Benefit)
            {
                title = "效益最大（PPO）";
                item = _latestAdvice.benefit;
            }
            else
            {
                title = "最安全（啟發式）";
                item = _latestAdvice.safe;
            }

            if (m_adviceTitleText != null) m_adviceTitleText.text = title;

            if (item == null)
            {
                if (m_adviceTileText != null) m_adviceTileText.text = "建議出牌：";
                if (m_adviceReasonText != null) m_adviceReasonText.text = "";
                return;
            }

            if (m_adviceTileText != null)
            {
                var t = string.IsNullOrEmpty(item.tile) ? item.tile_id.ToString() : item.tile;
                m_adviceTileText.text = $"建議出牌：{t}";
            }

            if (m_adviceReasonText != null)
            {
                string src = string.IsNullOrEmpty(item.source) ? "" : $"[{item.source}] ";
                m_adviceReasonText.text = src + (item.reason ?? "");
            }

            UpdatePromptAnswerTextFromAdvice();
        }



        // =========================
        // DebugUIBuilder prompt panel (StartScene style)
        // =========================
        public void SetConnectionState(bool connected)
        {
            _isConnected = connected;
            SetStatusText(connected ? "狀態：連線中" : "狀態：未連線");
        }

        public void SetStatusText(string status)
        {
            if (_promptStatusText != null)
            {
                _promptStatusText.text = status;
            }
        }

        public void SetPromptText(string text)
        {
            if (_promptAnswerText != null)
            {
                _promptAnswerText.text = text ?? "";
            }
        }

        private void TryBuildPromptPanel()
        {
            if (_promptBuilt) return;
            if (!m_useDebugPromptPanel) return;

            if (m_onlyBuildPromptInSpecificScene)
            {
                var sceneName = SceneManager.GetActiveScene().name;
                if (!string.Equals(sceneName, m_promptAllowedSceneName, StringComparison.Ordinal))
                {
                    // 不在指定場景就不要建立（避免 StartScene 重複顯示）
                    return;
                }
            }

            var ui = DebugUIBuilder.Instance;
            if (ui == null)
            {   
                Debug.LogError("DebugUIBuilder.Instance is NULL");
                return; // scene doesn't have DebugUIBuilder

            } 

            ui.AddLabel(m_promptTitle, m_promptPane);
            ui.AddDivider(m_promptPane);
            ui.AddLabel(m_promptSubtitle, m_promptPane);

            // Buttons: map to existing advice modes
            var btnOffRt = ui.AddButton(m_offenseButtonLabel, OnClickBenefit, -1, m_promptPane);
            ui.AddButton(m_defenseButtonLabel, OnClickSafe, -1, m_promptPane);

            int buttonFontSize = GetButtonFontSize(btnOffRt);
            float buttonW = btnOffRt.rect.width;
            float buttonH = btnOffRt.rect.height;

            ui.AddDivider(m_promptPane);

            // Status line
            var statusRt = ui.AddLabel("狀態：待命", m_promptPane);
            _promptStatusText = statusRt.GetComponent<Text>();

            // Big prompt box
            CreateBigPromptBox(ui, m_promptPane, buttonW, buttonH, buttonFontSize);
            SetPromptText(m_promptDefaultText);

            ui.Show();
            _promptBuilt = true;
        }

        private int GetButtonFontSize(RectTransform buttonRT)
        {
            if (buttonRT == null) return 36;
            var t = buttonRT.GetComponentInChildren<Text>(true);
            if (t != null && t.fontSize > 0) return t.fontSize;
            return 36;
        }

        private void CreateBigPromptBox(DebugUIBuilder ui, int pane, float buttonW, float buttonH, int buttonFontSize)
        {
            var containerRT = ui.AddLabel("", pane);

            var oldText = containerRT.GetComponent<Text>();
            if (oldText != null) oldText.enabled = false;

            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, buttonW);
            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, buttonH * m_promptBoxHeightMultiplier);

            // Background
            var bgGO = new GameObject("PromptBG", typeof(RectTransform), typeof(Image));
            bgGO.transform.SetParent(containerRT, false);
            bgGO.transform.SetAsFirstSibling();

            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            var bgImg = bgGO.GetComponent<Image>();
            bgImg.color = new Color(0f, 0f, 0f, 0.65f);

            // Text
            var textGO = new GameObject("PromptText", typeof(RectTransform), typeof(Text));
            textGO.transform.SetParent(containerRT, false);

            var textRT = textGO.GetComponent<RectTransform>();
            textRT.anchorMin = Vector2.zero;
            textRT.anchorMax = Vector2.one;
            textRT.offsetMin = new Vector2(m_promptPadding, m_promptPadding);
            textRT.offsetMax = new Vector2(-m_promptPadding, -m_promptPadding);

            _promptAnswerText = textGO.GetComponent<Text>();
            _promptAnswerText.text = "";
            _promptAnswerText.alignment = TextAnchor.UpperLeft;
            _promptAnswerText.horizontalOverflow = HorizontalWrapMode.Wrap;
            _promptAnswerText.verticalOverflow = VerticalWrapMode.Overflow;
            _promptAnswerText.supportRichText = true;
            _promptAnswerText.color = Color.white;

            int targetSize = (m_promptFontSizeOverride > 0) ? m_promptFontSizeOverride : buttonFontSize;
            _promptAnswerText.fontSize = targetSize;

            if (oldText != null && oldText.font != null)
            {
                _promptAnswerText.font = oldText.font;
                _promptAnswerText.fontStyle = oldText.fontStyle;
                _promptAnswerText.lineSpacing = oldText.lineSpacing;
            }
            else
            {
                _promptAnswerText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
        }

        private void UpdatePromptAnswerTextFromAdvice()
        {
            if (!_promptBuilt || _promptAnswerText == null) return;

            if (_latestAdvice == null)
            {
                // Keep default hint if no advice yet
                if (string.IsNullOrEmpty(_promptAnswerText.text))
                {
                    SetPromptText(m_promptDefaultText);
                }
                return;
            }

            AdviceItem item = (_mode == AdviceMode.Benefit) ? _latestAdvice.benefit : _latestAdvice.safe;
            string modeTitle = (_mode == AdviceMode.Benefit)
                ? "進攻模式（效益最大 / PPO）"
                : "防守模式（最安全 / 啟發式）";

            string tile = "";
            string reason = "";

            if (item != null)
            {
                tile = string.IsNullOrEmpty(item.tile) ? item.tile_id.ToString() : item.tile;
                string src = string.IsNullOrEmpty(item.source) ? "" : $"[{item.source}] ";
                reason = src + (item.reason ?? "");
            }

            string handLine = "";
            if (_latestHand != null && _latestHand.Length > 0)
            {
                handLine = string.Join(" ", _latestHand);
            }
            else if (m_handText != null)
            {
                handLine = m_handText.text;
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"模式：{modeTitle}");
            sb.AppendLine();
            sb.AppendLine("（建議打這張牌）");
            sb.AppendLine(string.IsNullOrEmpty(tile) ? "(無)" : tile);
            sb.AppendLine();
            sb.AppendLine("（理由）");
            sb.AppendLine(string.IsNullOrEmpty(reason) ? "(無)" : reason);

            if (!string.IsNullOrEmpty(handLine))
            {
                sb.AppendLine();
                sb.AppendLine("（目前手牌）");
                sb.AppendLine(handLine);
            }

            sb.AppendLine();
            sb.AppendLine("請按A按鈕開始辨識。");

            SetPromptText(sb.ToString());
        }
        private void ClearAnnotations()
        {
            foreach (var box in m_boxPool)
            {
                box?.SetActive(false);
            }
            BoxDrawn.Clear();
        }

        private void DrawBox(BoundingBox box, int id)
        {
            GameObject panel;
            if (id < m_boxPool.Count)
            {
                panel = m_boxPool[id];
                if (panel == null) panel = CreateNewBox(m_boxColor);
                else panel.SetActive(true);
            }
            else
            {
                panel = CreateNewBox(m_boxColor);
            }

            panel.transform.localPosition = new Vector3(
                box.CenterX,
                -box.CenterY,
                box.WorldPos.HasValue ? box.WorldPos.Value.z : 0.0f
            );

            panel.transform.rotation = Quaternion.LookRotation(panel.transform.position - m_detectionCanvas.GetCapturedCameraPosition());

            var rt = panel.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(box.Width, box.Height);

            var label = panel.GetComponentInChildren<Text>();
            label.text = box.Label;
            label.fontSize = m_fontSize; // ✅ 不再硬改 12，尊重 Inspector 設定
        }

        private GameObject CreateNewBox(Color color)
        {
            var panel = new GameObject("ObjectBox");
            _ = panel.AddComponent<CanvasRenderer>();
            var img = panel.AddComponent<Image>();
            img.color = color;
            img.sprite = m_boxTexture;
            img.type = Image.Type.Sliced;
            img.fillCenter = false;
            panel.transform.SetParent(m_displayLocation, false);

            var text = new GameObject("ObjectLabel");
            _ = text.AddComponent<CanvasRenderer>();
            text.transform.SetParent(panel.transform, false);
            var txt = text.AddComponent<Text>();
            txt.font = m_font;
            txt.color = m_fontColor;
            txt.fontSize = m_fontSize;
            txt.horizontalOverflow = HorizontalWrapMode.Overflow;

            var rt2 = text.GetComponent<RectTransform>();
            rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
            rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
            rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
            rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
            rt2.anchorMin = new Vector2(0, 0);
            rt2.anchorMax = new Vector2(1, 1);

            m_boxPool.Add(panel);
            return panel;
        }
        #endregion
    }
}
