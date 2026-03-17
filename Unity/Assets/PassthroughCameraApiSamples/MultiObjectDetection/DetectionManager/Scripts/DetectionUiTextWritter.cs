// Copyright (c) Meta Platforms, Inc. and affiliates.

using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class DetectionUiTextWritter : MonoBehaviour
    {
        [SerializeField] private Text m_labelInfo;

        [Header("Typewriter")]
        [Tooltip("每個字元的最短間隔 (秒)。數值越小越快。")]
        [SerializeField] private float m_writtingSpeed = 0.00015f;

        [Tooltip("遇到 ':' 時的額外停頓 (秒)。")]
        [SerializeField] private float m_writtingInfoPause = 0.005f;

        [SerializeField] private AudioSource m_writtingSound;

        [Header("Safety")]
        [Tooltip("字串太長時不要打字機，直接一次顯示完整文字，避免只顯示前半段。")]
        [SerializeField] private int m_maxAnimatedChars = 200;

        [Tooltip("若外部頻繁更新文字（例如每幀更新提示），建議關閉打字機。")]
        [SerializeField] private bool m_defaultAnimateOnEnable = false;

        public UnityEvent OnStartWritting;
        public UnityEvent OnFinishWritting;

        private float m_writtingTime = 0;
        private bool m_isWritting = false;
        private string m_targetText = "";
        private int m_index = 0;

        private void Start()
        {
            // 避免一開始就把長文字清空造成「看起來顯示不完全」
            if (m_defaultAnimateOnEnable)
            {
                SetText(m_labelInfo != null ? m_labelInfo.text : string.Empty, animate: true);
            }
        }

        private void OnEnable()
        {
            if (m_defaultAnimateOnEnable)
            {
                SetText(m_labelInfo != null ? m_labelInfo.text : string.Empty, animate: true);
            }
        }

        private void OnDisable()
        {
            // 關閉時把完整內容顯示出來，避免停在半段
            StopAndShowFull();
        }

        private void LateUpdate()
        {
            if (!m_isWritting || m_labelInfo == null)
                return;

            if (m_writtingTime <= 0)
            {
                m_writtingTime = m_writtingSpeed;

                m_writtingSound?.Play();

                if (m_index < 0) m_index = 0;
                if (m_index >= m_targetText.Length)
                {
                    m_isWritting = false;
                    OnFinishWritting?.Invoke();
                    return;
                }

                var nextChar = m_targetText.Substring(m_index, 1);
                m_labelInfo.text += nextChar;

                if (nextChar == ":")
                {
                    m_writtingTime += m_writtingInfoPause;
                }

                m_index++;

                if (m_index >= m_targetText.Length)
                {
                    m_isWritting = false;
                    OnFinishWritting?.Invoke();
                }
            }
            else
            {
                m_writtingTime -= Time.deltaTime;
            }
        }

        /// <summary>
        /// 建議由其他腳本呼叫這個方法更新文字，避免直接改 Text.text 造成打字機卡在半段。
        /// </summary>
        public void SetText(string text, bool animate)
        {
            if (m_labelInfo == null)
                return;

            text ??= string.Empty;

            // 長字串直接顯示，避免 VR 介面只看到前半段
            if (!animate || (m_maxAnimatedChars > 0 && text.Length > m_maxAnimatedChars))
            {
                m_isWritting = false;
                m_writtingTime = 0;
                m_targetText = text;
                m_index = text.Length;
                m_labelInfo.text = text;
                OnFinishWritting?.Invoke();
                return;
            }

            // restart typewriter
            m_isWritting = true;
            m_writtingTime = 0;
            m_targetText = text;
            m_index = 0;
            m_labelInfo.text = "";
            OnStartWritting?.Invoke();
        }

        /// <summary>
        /// 立刻停止打字機並顯示完整文字。
        /// </summary>
        public void StopAndShowFull()
        {
            if (m_labelInfo == null)
                return;

            m_isWritting = false;
            m_writtingTime = 0;
            m_index = m_targetText != null ? m_targetText.Length : 0;

            if (m_targetText != null)
            {
                m_labelInfo.text = m_targetText;
            }
        }

        /// <summary>
        /// 目前是否正在打字機輸出。
        /// </summary>
        public bool IsWriting => m_isWritting;
    }
}
