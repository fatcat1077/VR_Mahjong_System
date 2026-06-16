// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections;
using System.Text;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using PassthroughCameraSamples.StartScene;
using Meta.XR.Samples;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class DetectionUiMenuManager : MonoBehaviour
    {
        [Header("Ui buttons")]
        [SerializeField] private OVRInput.RawButton m_actionButton = OVRInput.RawButton.A;
        [SerializeField] private OVRInput.RawButton m_recenterPromptButton = OVRInput.RawButton.B;
        [SerializeField] private OVRInput.Button m_recenterPromptVirtualButton = OVRInput.Button.Two;
        [SerializeField] private KeyCode m_recenterPromptKey = KeyCode.B;

        [Header("Ui elements ref.")]
        [SerializeField] private GameObject m_loadingPanel;
        [SerializeField] private GameObject m_initialPanel;
        [SerializeField] private GameObject m_noPermissionPanel;
        [SerializeField] private Text m_labelInfromation;
        [SerializeField] private AudioSource m_buttonSound;

        [Header("Prompt Display")]
        [Tooltip("If true, connection/advice/pc log will be printed into m_labelInfromation.")]
        [SerializeField] private bool m_useLabelForPrompt = true;
        [Tooltip("Temporary: keep a second copy in the bottom panel while validating the side prompt.")]
        [SerializeField] private bool m_showBottomPromptCopy = true;

        [Header("DebugUIBuilder Prompt Box (optional)")]
        [Tooltip("If true, also show a StartScene-style big prompt box using DebugUIBuilder.")]
        [SerializeField] private bool m_useDebugPromptBox = true;
        [SerializeField] private int m_promptPane = DebugUIBuilder.DEBUG_PANE_LEFT;
        [SerializeField] private string m_promptTitle = "麻將小幫手";
        [SerializeField] private float m_promptWidth = 760f;
        [SerializeField] private float m_promptHeight = 420f;
        [SerializeField] private float m_promptPadding = 24f;
        [SerializeField] private int m_promptFontSize = 30;


        public bool IsInputActive { get; set; } = false;

        public UnityEvent<bool> OnPause = new UnityEvent<bool>();

        private bool m_initialMenu;

        // start menu
        private int m_objectsDetected = 0;
        private int m_objectsIdentified = 0;

        // pause menu
        public bool IsPaused { get; private set; } = true;

        // --- Prompt state (PC -> Quest) ---
        private bool _connected;
        private string _connectionStatus = "DISCONNECTED";
        private string _errorDetail = "";
        private string _pcLog = "";
        private string _streamDebug = "";
        private string[] _lastHand;
        private SentisInferenceUiManager.Advice _lastAdvice;

        // --- DebugUIBuilder prompt runtime refs ---
        private bool _promptBuilt = false;
        private Text _promptStatusText;
        private Text _promptAnswerText;


        #region Unity Functions
        private IEnumerator Start()
        {
            if (m_initialPanel != null) m_initialPanel.SetActive(false);
            if (m_noPermissionPanel != null) m_noPermissionPanel.SetActive(false);
            if (m_loadingPanel != null) m_loadingPanel.SetActive(true);

            // Wait until Sentis model is loaded
            var sentisInference = FindFirstObjectByType<SentisInferenceRunManager>();
            if (sentisInference != null)
            {
                while (!sentisInference.IsModelLoaded)
                    yield return null;
            }

            if (m_loadingPanel != null) m_loadingPanel.SetActive(false);

            while (!PassthroughCameraPermissions.HasCameraPermission.HasValue)
            {
                yield return null;
            }
            if (PassthroughCameraPermissions.HasCameraPermission == false)
            {
                OnNoPermissionMenu();
            }

            TryBuildPromptPanel();

            UpdateLabelInformation();
        }

        private void Update()
        {
            if (OVRInput.GetUp(m_recenterPromptButton) ||
                OVRInput.GetUp(m_recenterPromptVirtualButton) ||
                Input.GetKeyUp(m_recenterPromptKey))
            {
                RecenterPromptPanel();
            }

            if (!IsInputActive)
                return;

            if (m_initialMenu)
            {
                InitialMenuUpdate();
            }
        }
        #endregion

        public void RecenterPromptPanel()
        {
            TryBuildPromptPanel();

            var ui = DebugUIBuilder.Instance;
            if (ui == null || !_promptBuilt)
                return;

            ui.ShowAtCurrentHeadPose();
        }

        #region Ui state: No permissions Menu
        private void OnNoPermissionMenu()
        {
            m_initialMenu = false;
            IsPaused = true;
            if (m_initialPanel != null) m_initialPanel.SetActive(false);
            if (m_noPermissionPanel != null) m_noPermissionPanel.SetActive(true);
        }
        #endregion

        #region Ui state: Initial Menu
        public void OnInitialMenu(bool hasScenePermission)
        {
            // Check if we have the Scene data permission
            if (hasScenePermission)
            {
                m_initialMenu = true;
                IsPaused = true;
                if (m_initialPanel != null) m_initialPanel.SetActive(true);
                if (m_noPermissionPanel != null) m_noPermissionPanel.SetActive(false);
            }
            else
            {
                OnNoPermissionMenu();
            }
        }

        private void InitialMenuUpdate()
        {
            if (OVRInput.GetUp(m_actionButton) || Input.GetKey(KeyCode.Return))
            {
                m_buttonSound?.Play();
                OnPauseMenu(false);
            }
        }

        private void OnPauseMenu(bool visible)
        {
            m_initialMenu = false;
            IsPaused = visible;

            if (m_initialPanel != null) m_initialPanel.SetActive(false);
            if (m_noPermissionPanel != null) m_noPermissionPanel.SetActive(false);

            OnPause?.Invoke(visible);
        }
        #endregion

        #region Public API: Prompt panel update
        /// <summary>
        /// Update connection state shown in the bottom label (or your prompt panel).
        /// Keep parameter name 'errorDetail' so callers using named arguments compile.
        /// </summary>
        public void SetConnectionState(bool connected, string statusText = null, string errorDetail = null)
        {
            _connected = connected;
            _connectionStatus = string.IsNullOrEmpty(statusText)
                ? (connected ? "CONNECTED" : "DISCONNECTED")
                : statusText;
            _errorDetail = errorDetail ?? "";
            TryBuildPromptPanel();

            UpdateLabelInformation();
        }

        /// <summary>
        /// Display raw pc log (recommended for your current phase).
        /// </summary>
        public void SetPcLog(string pcLog)
        {
            _pcLog = pcLog ?? "";
            TryBuildPromptPanel();

            UpdateLabelInformation();
        }

        /// <summary>
        /// Display local Quest-side stream counters even before the PC responds.
        /// </summary>
        public void SetStreamDebug(string streamDebug)
        {
            _streamDebug = streamDebug ?? "";
            TryBuildPromptPanel();

            UpdateLabelInformation();
        }

        /// <summary>
        /// Compatibility: some scripts still call SetMahjongAdvice(). We keep it.
        /// </summary>
        public void SetMahjongAdvice(SentisInferenceUiManager.Advice advice, string[] hand, double ts = 0)
        {
            _lastAdvice = advice;
            _lastHand = hand;
            // If pc_log isn't provided, we will build a short text from advice + hand.
            if (string.IsNullOrEmpty(_pcLog))
                TryBuildPromptPanel();

            UpdateLabelInformation();
        }
        #endregion

        #region Ui state: detection information / label
        private void UpdateLabelInformation()
        {
            // Build prompt text once, then route to:
            //  - existing label (m_labelInfromation) if enabled
            //  - DebugUIBuilder big prompt box if enabled/built

            // Ensure the label can show multi-line content
            if (m_labelInfromation != null)
            {
                try
                {
                    m_labelInfromation.horizontalOverflow = HorizontalWrapMode.Wrap;
                    m_labelInfromation.verticalOverflow = VerticalWrapMode.Overflow;
                }
                catch { }
            }

            var sb = new StringBuilder(512);

            // PC log block (preferred). Keep the Quest overlay focused on the
            // current hand, locked table, latest discard, turn, and recommendation.
            if (!string.IsNullOrEmpty(_pcLog))
            {
                sb.AppendLine(_pcLog.TrimEnd());
            }
            else
            {
                // Fallback: build from advice/hand if available
                if (_lastAdvice != null || (_lastHand != null && _lastHand.Length > 0))
                {
                    var handStr = (_lastHand != null) ? string.Join(" ", _lastHand) : "";
                    var benefitTile = _lastAdvice?.benefit?.tile ?? "";
                    var safeTile = _lastAdvice?.safe?.tile ?? "";
                    var suggestedTile = !string.IsNullOrEmpty(benefitTile) ? benefitTile : safeTile;

                    sb.AppendLine($"手牌：{(!string.IsNullOrEmpty(handStr) ? handStr : "辨識中")}");
                    sb.AppendLine($"建議動作：{(!string.IsNullOrEmpty(suggestedTile) ? "打 " + suggestedTile : "正在偵測出牌")}");
                }
                else
                {
                    sb.AppendLine("手牌：辨識中");
                    sb.AppendLine("建議動作：等待 PC 回傳");
                }
            }

            var text = sb.ToString();
            var displayText = BuildPromptDisplayText(text);

            // 1) Existing bottom label. Default is off so the Quest view has one menu only.
            if ((m_useLabelForPrompt || m_showBottomPromptCopy) && m_labelInfromation != null)
            {
                m_labelInfromation.supportRichText = true;
                m_labelInfromation.text = displayText;
            }
            else if (m_labelInfromation != null)
            {
                m_labelInfromation.text = "";
            }

            // 2) DebugUIBuilder big prompt box
            if (_promptBuilt && _promptAnswerText != null)
            {
                _promptAnswerText.text = displayText;
            }
        }

        private string BuildPromptDisplayText(string rawText)
        {
            var hand = ExtractLineValue(rawText, "手牌");
            var table = ExtractLineValue(rawText, "桌上牌");
            var action = ExtractLineValue(rawText, "建議動作");
            var discard = ExtractLineValue(rawText, "出牌");
            var turn = ExtractLineValue(rawText, "Turn");
            var stableState = ExtractLineValue(rawText, "Stable");

            if (string.IsNullOrEmpty(hand)) hand = "辨識中";
            if (string.IsNullOrEmpty(table)) table = "辨識中";
            if (string.IsNullOrEmpty(action)) action = "等待 PC 回傳";
            if (!string.IsNullOrEmpty(stableState))
                hand = $"{hand} ({stableState})";

            var sb = new StringBuilder(512);
            AppendPromptSection(sb, "手牌", hand, "#9AE6B4", 34);
            sb.AppendLine();
            AppendPromptSection(sb, "桌上牌", table, "#93C5FD", 30);
            if (!string.IsNullOrEmpty(discard))
            {
                sb.AppendLine();
                AppendPromptSection(sb, "出牌", discard, "#FCA5A5", 32);
            }
            if (!string.IsNullOrEmpty(turn))
            {
                sb.AppendLine();
                AppendPromptSection(sb, "Turn", turn, "#C4B5FD", 30);
            }
            sb.AppendLine();
            AppendPromptSection(sb, "建議動作", action, "#FDE68A", 36);
            return sb.ToString().TrimEnd();
        }

        private static string ExtractLineValue(string rawText, string label)
        {
            if (string.IsNullOrEmpty(rawText) || string.IsNullOrEmpty(label))
                return "";

            var lines = rawText.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (!trimmed.StartsWith(label, StringComparison.Ordinal))
                    continue;

                var idx = trimmed.IndexOf('：');
                if (idx < 0) idx = trimmed.IndexOf(':');
                if (idx >= 0 && idx + 1 < trimmed.Length)
                    return trimmed.Substring(idx + 1).Trim();
            }
            return "";
        }

        private static void AppendPromptSection(StringBuilder sb, string label, string value, string color, int valueSize)
        {
            sb.Append("<size=24><color=").Append(color).Append("><b>");
            sb.Append(EscapeRichText(label));
            sb.AppendLine("</b></color></size>");
            sb.Append("<size=").Append(valueSize).Append("><b>");
            sb.Append(EscapeRichText(value));
            sb.AppendLine("</b></size>");
        }

        private static string EscapeRichText(string text)
        {
            return (text ?? "")
                .Replace("<", "‹")
                .Replace(">", "›");
        }

        public void OnObjectsDetected(int objects)
        {
            m_objectsDetected = objects;
            TryBuildPromptPanel();

            UpdateLabelInformation();
        }

        public void OnObjectsIndentified(int objects)
        {
            if (objects < 0)
            {
                // reset the counter
                m_objectsIdentified = 0;
            }
            else
            {
                m_objectsIdentified += objects;
            }
            TryBuildPromptPanel();

            UpdateLabelInformation();
        }
        

        #region DebugUIBuilder prompt box (StartScene-style)
        private void TryBuildPromptPanel()
        {
            if (_promptBuilt) return;
            if (!m_useDebugPromptBox) return;

            var ui = DebugUIBuilder.Instance;
            if (ui == null) return;

            var titleRt = ui.AddLabel(m_promptTitle, m_promptPane);
            var titleText = titleRt.GetComponent<Text>();
            if (titleText != null)
            {
                titleText.alignment = TextAnchor.MiddleLeft;
                titleText.fontSize = 34;
                titleText.color = new Color(0.95f, 1f, 0.96f, 1f);
            }
            ui.AddDivider(m_promptPane);

            CreateBigPromptBox(ui, m_promptPane, m_promptWidth, m_promptHeight);

            ui.Show();
            _promptBuilt = true;
        }

        private void CreateBigPromptBox(DebugUIBuilder ui, int pane, float width, float height)
        {
            var containerRT = ui.AddLabel("", pane);

            // Hide the default Text component (we use our own Text inside)
            var oldText = containerRT.GetComponent<Text>();
            if (oldText != null) oldText.enabled = false;

            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, width);
            containerRT.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, height);

            // Background
            var bgGO = new GameObject("PromptBG", typeof(RectTransform), typeof(Image), typeof(Outline));
            bgGO.transform.SetParent(containerRT, false);
            bgGO.transform.SetAsFirstSibling();

            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            var bgImg = bgGO.GetComponent<Image>();
            bgImg.color = new Color(0.02f, 0.035f, 0.04f, 0.78f);

            var outline = bgGO.GetComponent<Outline>();
            outline.effectColor = new Color(0.55f, 0.95f, 0.74f, 0.45f);
            outline.effectDistance = new Vector2(2f, -2f);

            var accentGO = new GameObject("PromptAccent", typeof(RectTransform), typeof(Image));
            accentGO.transform.SetParent(containerRT, false);
            var accentRT = accentGO.GetComponent<RectTransform>();
            accentRT.anchorMin = new Vector2(0f, 0f);
            accentRT.anchorMax = new Vector2(0f, 1f);
            accentRT.pivot = new Vector2(0f, 0.5f);
            accentRT.offsetMin = new Vector2(0f, 0f);
            accentRT.offsetMax = new Vector2(7f, 0f);
            accentGO.GetComponent<Image>().color = new Color(0.46f, 0.92f, 0.62f, 0.95f);

            // Text
            var textGO = new GameObject("PromptText", typeof(RectTransform), typeof(Text), typeof(Shadow));
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
            _promptAnswerText.fontSize = m_promptFontSize;
            _promptAnswerText.lineSpacing = 1.08f;

            var shadow = textGO.GetComponent<Shadow>();
            shadow.effectColor = new Color(0f, 0f, 0f, 0.65f);
            shadow.effectDistance = new Vector2(1.5f, -1.5f);

            // Match DebugUIBuilder font when possible
            if (oldText != null && oldText.font != null)
            {
                _promptAnswerText.font = oldText.font;
                _promptAnswerText.fontStyle = oldText.fontStyle;
            }
            else
            {
                _promptAnswerText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
        }
        #endregion

#endregion
    }
}
