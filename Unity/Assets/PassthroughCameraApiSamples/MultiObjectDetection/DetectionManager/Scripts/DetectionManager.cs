// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class DetectionManager : MonoBehaviour
    {
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;

        [Header("Controls configuration")]
        [SerializeField] private OVRInput.RawButton m_actionButton = OVRInput.RawButton.A;

        [Header("Ui references")]
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;

        [Header("Placement configureation")]
        [SerializeField] private GameObject m_spwanMarker;
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private float m_spawnDistance = 0.25f;
        [SerializeField] private AudioSource m_placeSound;

        [Header("Sentis inference ref")]
        [SerializeField] private SentisInferenceRunManager m_runInference;
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("Mahjong classifier (second model)")]
        [SerializeField] private MahjongClassifierRunManager m_classifier;   // �ĤG�Ӽҫ��G�P������
        [SerializeField] private bool m_enableClassification = true;         // �O�_�ҥβĤG���q����

        [Space(10)]
        public UnityEvent<int> OnObjectsIdentified;

        private bool m_isPaused = true;
        private List<GameObject> m_spwanedEntities = new();
        private bool m_isStarted = false;
        private bool m_isSentisReady = false;
        private float m_delayPauseBackTime = 0;

        #region Unity Functions
        private void Awake()
        {
            OVRManager.display.RecenteredPose += CleanMarkersCallBack;

            // Robust wiring: even if the UnityEvent listener was not set in the Inspector,
            // make sure pressing A on the start menu can unpause the detection logic.
            if (m_uiMenuManager != null && m_uiMenuManager.OnPause != null)
            {
                m_uiMenuManager.OnPause.AddListener(OnPause);

                // Also ensure the UI can receive input.
                m_uiMenuManager.IsInputActive = true;
            }
        }

        private IEnumerator Start()
        {
            // Wait until Sentis model is loaded (YOLO �ҫ�)
            var sentisInference = FindAnyObjectByType<SentisInferenceRunManager>();
            while (!sentisInference.IsModelLoaded)
            {
                yield return null;
            }
            m_isSentisReady = true;
        }

        private void Update()
        {
            // Get the WebCamTexture CPU image
            var hasWebCamTextureData = m_webCamTextureManager.WebCamTexture != null;

            if (!m_isStarted)
            {
                // Manage the Initial Ui Menu
                if (hasWebCamTextureData && m_isSentisReady)
                {
                    m_uiMenuManager.OnInitialMenu(m_environmentRaycast.HasScenePermission());
                    m_isStarted = true;
                }
            }
            else
            {
                // Press A button to spawn 3d markers
                if (OVRInput.GetUp(m_actionButton) && m_delayPauseBackTime <= 0)
                {
                    SpwanCurrentDetectedObjects();
                }
                // Cooldown for the A button after return from the pause menu
                m_delayPauseBackTime -= Time.deltaTime;
                if (m_delayPauseBackTime <= 0)
                {
                    m_delayPauseBackTime = 0;
                }
            }

            // Not start a sentis inference if the app is paused or we don't have a valid WebCamTexture
            if (m_isPaused || !hasWebCamTextureData)
            {
                if (m_isPaused)
                {
                    // Set the delay time for the A button to return from the pause menu
                    m_delayPauseBackTime = 0.1f;
                }
                return;
            }

            // Run a new inference when the current inference finishes
            if (!m_runInference.IsRunning())
            {
                // �� �b�ҰʤU�@�� YOLO �e�A�Τ����ҫ��B�z�W�@������
                if (m_enableClassification && m_classifier != null)
                {
                    ClassifyCurrentDetections();
                }

                // �A�ҰʤU�@�� YOLO ����
                m_runInference.RunInference(m_webCamTextureManager.WebCamTexture);
            }
        }
        #endregion

        #region Classification Functions

        /// <summary>
        /// �ϥ� MahjongClassifierRunManager ��ثe YOLO �����쪺�C�Ӯضi������A
        /// �ç⵲�G�g�^ BoundingBox.ClassName�]UI �W�N�|��ܷs���P�W�^�A
        /// �P�ɿ�X Debug.Log�C
        /// </summary>
        private void ClassifyCurrentDetections()
        {
            if (m_uiInference == null || m_uiInference.BoxDrawn == null)
                return;

            if (m_webCamTextureManager == null || m_webCamTextureManager.WebCamTexture == null)
                return;

            if (m_classifier == null || !m_classifier.IsModelLoaded)
                return;

            var camTex = m_webCamTextureManager.WebCamTexture;
            int imgW = camTex.width;
            int imgH = camTex.height;

            var boxes = m_uiInference.BoxDrawn;
            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];

                // ���O�� YOLO �쥻�����O�W�١]�q�`�|�O "tile" �����^
                string beforeClass = box.ClassName;

                // UI �y�� -> �v�� pixel �y��
                int w = Mathf.Clamp(Mathf.RoundToInt(box.Width), 1, imgW);
                int h = Mathf.Clamp(Mathf.RoundToInt(box.Height), 1, imgH);

                int cx = Mathf.RoundToInt(box.CenterX + imgW * 0.5f);
                int cy = Mathf.RoundToInt(imgH * 0.5f - box.CenterY); // Y �b����

                int xMin = Mathf.Clamp(cx - w / 2, 0, imgW - 1);
                int yMin = Mathf.Clamp(cy - h / 2, 0, imgH - 1);
                int width = Mathf.Clamp(w, 1, imgW - xMin);
                int height = Mathf.Clamp(h, 1, imgH - yMin);

                // �q WebCamTexture ���X�o�Ӯ�
                Color[] pixels = camTex.GetPixels(xMin, yMin, width, height);
                Texture2D tileTex = new Texture2D(width, height, TextureFormat.RGB24, false);
                tileTex.SetPixels(pixels);
                tileTex.Apply();

                // �ᵹ�������]�ĤG�Ӽҫ�
                m_classifier.RunInference(tileTex);
                string predictedLabel = m_classifier.GetLastResult();

                // �Χ��O�o����Ȧs Texture�A�קK�ֿn�O����
                Object.Destroy(tileTex);

                // ��s�ت� ClassName�]���� 3D marker / UI �|�Ψ�^
                box.ClassName = predictedLabel;
                // �Y�Ʊ� UI �W�u��ܵP�W�A�i�H�P�B�� label�G
                box.Label = $"Class: {predictedLabel}";

                boxes[i] = box;   // BoundingBox �O struct�A�n�g�^ List �~�|�ͮ�

                // �� Debug�G�����Ϥ������G��X�� Log
                Debug.Log(
                    $"[MahjongClassifier] Detected tile #{i}: YOLO={beforeClass} �� Classifier={predictedLabel}, " +
                    $"Pos=({xMin},{yMin}), Size=({width}x{height})"
                );
            }
            // �� �s�W�G���s�L�� BoxDrawn �M�Φ^ UI �W�� Text
            m_uiInference.RefreshBoxLabels(useClassNameOnly: true);
        }

        #endregion

        #region Marker Functions
        /// <summary>
        /// Clean 3d markers when the tracking space is re-centered.
        /// </summary>
        private void CleanMarkersCallBack()
        {
            foreach (var e in m_spwanedEntities)
            {
                Destroy(e, 0.1f);
            }
            m_spwanedEntities.Clear();
            OnObjectsIdentified?.Invoke(-1);
        }
        /// <summary>
        /// Spwan 3d markers for the detected objects
        /// </summary>
        private void SpwanCurrentDetectedObjects()
        {
            var count = 0;
            foreach (var box in m_uiInference.BoxDrawn)
            {
                if (PlaceMarkerUsingEnvironmentRaycast(box.WorldPos, box.ClassName))
                {
                    count++;
                }
            }
            if (count > 0)
            {
                // Play sound if a new marker is placed.
                m_placeSound.Play();
            }
            OnObjectsIdentified?.Invoke(count);
        }

        /// <summary>
        /// Place a marker using the environment raycast
        /// </summary>
        private bool PlaceMarkerUsingEnvironmentRaycast(Vector3? position, string className)
        {
            // Check if the position is valid
            if (!position.HasValue)
            {
                return false;
            }

            // Check if�A spanwed the same object before
            var existMarker = false;
            foreach (var e in m_spwanedEntities)
            {
                var markerClass = e.GetComponent<DetectionSpawnMarkerAnim>();
                if (markerClass)
                {
                    var dist = Vector3.Distance(e.transform.position, position.Value);
                    if (dist < m_spawnDistance && markerClass.GetYoloClassName() == className)
                    {
                        existMarker = true;
                        break;
                    }
                }
            }

            if (!existMarker)
            {
                // spawn a visual marker
                var eMarker = Instantiate(m_spwanMarker);
                m_spwanedEntities.Add(eMarker);

                // Update marker transform with the real world transform
                eMarker.transform.SetPositionAndRotation(position.Value, Quaternion.identity);
                eMarker.GetComponent<DetectionSpawnMarkerAnim>().SetYoloClassName(className);
            }

            return !existMarker;
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Pause the detection logic when the pause menu is active
        /// </summary>
        public void OnPause(bool pause)
        {
            m_isPaused = pause;
        }
        #endregion
    }
}
