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

        [Header("Ui elements ref.")]
        [SerializeField] private GameObject m_loadingPanel;
        [SerializeField] private GameObject m_initialPanel;
        [SerializeField] private GameObject m_noPermissionPanel;
        [SerializeField] private Text m_labelInfromation;
        [SerializeField] private AudioSource m_buttonSound;

        [Header("Prompt Display")]
        [Tooltip("If true, connection/advice/pc log will be printed into m_labelInfromation.")]
        [SerializeField] private bool m_useLabelForPrompt = true;

        [Header("DebugUIBuilder Prompt Box (optional)")]
        [Tooltip("If true, also show a StartScene-style big prompt box using DebugUIBuilder.")]
        [SerializeField] private bool m_useDebugPromptBox = true;
        [SerializeField] private int m_promptPane = DebugUIBuilder.DEBUG_PANE_LEFT;
        [SerializeField] private string m_promptTitle = "麻將小幫手";
        [SerializeField] private float m_promptWidth = 720f;
        [SerializeField] private float m_promptHeight = 680f;
        [SerializeField] private float m_promptPadding = 14f;
        [SerializeField] private int m_promptFontSize = 32;


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
            if (!IsInputActive)
                return;

            if (m_initialMenu)
            {
                InitialMenuUpdate();
            }
        }
        #endregion

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

            // Connection line
            sb.Append("[NET] ");
            sb.Append(_connected ? "CONNECTED" : "DISCONNECTED");
            if (!string.IsNullOrEmpty(_connectionStatus))
                sb.Append(" | ").Append(_connectionStatus);
            sb.AppendLine();

            if (!_connected && !string.IsNullOrEmpty(_errorDetail))
            {
                sb.Append("[ERR] ").AppendLine(_errorDetail);
            }

            // PC log block (preferred)
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
                    var benefitId = _lastAdvice?.benefit?.tile_id ?? -1;
                    var benefitSrc = _lastAdvice?.benefit?.source ?? "";
                    var safeTile = _lastAdvice?.safe?.tile ?? "";
                    var safeId = _lastAdvice?.safe?.tile_id ?? -1;
                    var safeSrc = _lastAdvice?.safe?.source ?? "";

                    sb.AppendLine($"benefit: {benefitTile} (id={benefitId}, {benefitSrc}) | safe: {safeTile} (id={safeId}, {safeSrc})");
                    if (!string.IsNullOrEmpty(handStr))
                        sb.AppendLine($"hand: {handStr}");
                }
            }

            // Keep sample stats (so you don't lose debug info)
            sb.AppendLine();
            sb.AppendLine($"Detecting objects: {m_objectsDetected}");
            sb.AppendLine($"Objects identified: {m_objectsIdentified}");

            var text = sb.ToString();

            // 1) Existing label (keep original behavior)
            if (m_useLabelForPrompt && m_labelInfromation != null)
            {
                m_labelInfromation.text = text;
            }

            // 2) DebugUIBuilder big prompt box
            if (_promptBuilt && _promptAnswerText != null)
            {
                _promptAnswerText.text = text;
            }

            // Status line in DebugUIBuilder
            if (_promptBuilt && _promptStatusText != null)
            {
                _promptStatusText.text = _connected ? "狀態：連線中" : "狀態：待命";
            }
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

            ui.AddLabel(m_promptTitle, m_promptPane);
            ui.AddDivider(m_promptPane);

            // Status line
            var statusRt = ui.AddLabel("狀態：待命", m_promptPane);
            _promptStatusText = statusRt.GetComponent<Text>();

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
            _promptAnswerText.fontSize = m_promptFontSize;

            // Match DebugUIBuilder font when possible
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
        #endregion

#endregion
    }
}
