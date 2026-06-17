// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using Meta.XR.Samples;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Rendering;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceRunManager : MonoBehaviour
    {
        // =========================================================
        // ✅ Stream-only mode (Quest -> PC save jpg)
        // =========================================================
        [Header("Stream Only (Quest -> PC)")]
        [SerializeField] private bool m_streamOnly = true;

        [SerializeField] private string m_serverIp = "127.0.0.1";
        [SerializeField] private int m_serverPort = 5000;

        [Header("Quest Local Segmentation Latency")]
        [Tooltip("Run the latency segmentation models on Quest with Sentis, and send only latency telemetry to PC.")]
        [SerializeField] private bool m_questLocalSegLatencyMode = true;

        [SerializeField] private BackendType m_questLocalSegLatencyBackend = BackendType.GPUCompute;

        [Tooltip("Warmup forward runs per local Sentis model before A-button measurement.")]
        [SerializeField, Range(0, 10)] private int m_questLocalSegLatencyWarmupRuns = 3;

        [Tooltip("Rotate the first model each measured frame so no model always pays the first-forward idle cost.")]
        [SerializeField] private bool m_rotateQuestLocalLatencyOrder = true;

        [Tooltip("Optional delay between measured Quest-local frame bursts. 0 means run the next burst immediately.")]
        [SerializeField, Range(0f, 2f)] private float m_questLocalLatencyFrameIntervalSec = 0f;

        [Tooltip("Optional local frame cap for unattended tests. 0 means stop only when A is pressed again.")]
        [SerializeField] private int m_questLocalLatencyMaxFrames = 0;

        [Tooltip("Frames per second to stream to PC. Higher resolution needs a little more bandwidth/headroom.")]
        [SerializeField, Range(1, 30)] private int m_sendFps = 8;

        [SerializeField, Range(10, 100)] private int m_jpegQuality = 92;

        [Tooltip("固定輸出尺寸（會把來源 Blit 到這個大小）。Use 4:3 to avoid squashing the camera image.")]
        [SerializeField] private Vector2Int m_streamSize = new(1280, 960);

        [Tooltip("Keep stream settings at the high-resolution defaults even if an old scene/prefab serialized lower values.")]
        [SerializeField] private bool m_forceHighResolutionStreamDefaults = true;

        [SerializeField] private bool m_autoReconnect = true;
        [SerializeField] private float m_reconnectIntervalSec = 2f;

        [Header("Stream Debug")]
        [SerializeField] private bool m_showStreamDebug = true;
        [SerializeField] private float m_debugUiIntervalSec = 0.5f;

        [Header("Camera Source")]
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;

        [Header("Mahjong Scene Reset")]
        [SerializeField] private OVRInput.RawButton m_resetSceneButton = OVRInput.RawButton.X;

        [Header("Segmentation Latency Measurement")]
        [SerializeField] private OVRInput.RawButton m_measurementToggleButton = OVRInput.RawButton.A;

        // Networking
        private TcpClient _client;
        private NetworkStream _stream;
        private Thread _sendThread;
        private Thread _recvThread;
        private volatile bool _netRunning;
        private volatile bool _connected;
        private readonly object _sendLock = new object();
        private readonly ConcurrentQueue<string> _recvJsonQueue = new ConcurrentQueue<string>();
        private readonly ConcurrentQueue<byte[]> _outgoingPackets = new ConcurrentQueue<byte[]>();

        // Capture
        private Texture _latestTexture;
        private RenderTexture _rt;
        private Texture2D _cpuTex;
        private volatile bool _captureInFlight;
        private volatile byte[] _pendingJpeg; // 丟幀：不為 null 代表上一張還沒送出

        private float _nextSendTime;
        private float _nextReconnectTime;
        private float _nextDebugUiTime;

        // Debug counters. Keep these independent from the stream protocol.
        private int _connectAttempts;
        private int _framesQueued;
        private int _framesSent;
        private int _responsesReceived;
        private int _jsonParseErrors;
        private int _sendFailures;
        private int _recvFailures;
        private int _lastSentBytes;
        private int _telemetryPacketsQueued;
        private int _telemetryPacketsSent;
        private int _sceneResetRequests;
        private int _measurementToggleRequests;
        private bool _measurementActive;

        // Quest-local Sentis latency state.
        private readonly List<QuestSegLatencyModel> _questLatencyModels = new List<QuestSegLatencyModel>();
        private Coroutine _questLatencyCoroutine;
        private int _questLatencyFrameIndex;
        private int _questLatencyModelLoadFailures;
        private string _questLatencyLastSummary = "idle";

        // =========================================================
        // ✅ Original Sentis fields (保留，避免 Editor/其他腳本報錯)
        // =========================================================
        [Header("Sentis Model config")]
        [SerializeField] private Vector2Int m_inputSize = new(640, 640);
        [SerializeField] private BackendType m_backend = BackendType.CPU;
        [SerializeField] private ModelAsset m_sentisModel;
        [SerializeField] private int m_layersPerFrame = 25;
        [SerializeField] private TextAsset m_labelsAsset;
        public bool IsModelLoaded { get; private set; } = false;

        [Header("UI display references")]
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("Menu / Prompt UI")]
        [SerializeField] private DetectionUiMenuManager m_menuUi;

        [Header("[Editor Only] Convert to Sentis")]
        public ModelAsset OnnxModel; // ⚠️ 這個一定要留，Editor Converter 會用到
        [SerializeField, Range(0, 1)] private float m_iouThreshold = 0.6f;
        [SerializeField, Range(0, 1)] private float m_scoreThreshold = 0.23f;
        [Space(40)]

        private Worker m_engine;
        private IEnumerator m_schedule;
        private bool m_started = false;
        private Tensor<float> m_input;
        private Model m_model;
        private int m_download_state = 0;
        private Tensor<float> m_output;
        private Tensor<int> m_labelIDs;
        private Tensor<float> m_pullOutput;
        private Tensor<int> m_pullLabelIDs;
        private bool m_isWaiting = false;

        private static readonly QuestSegLatencyModelSpec[] QuestSegLatencySpecs =
        {
            new QuestSegLatencyModelSpec("best_baseline", "BenchmarkModels/latency4_best_baseline", 416, 3264006),
            new QuestSegLatencyModelSpec("best_kd", "BenchmarkModels/latency4_best_kd", 416, 3264006),
            new QuestSegLatencyModelSpec("best_p2", "BenchmarkModels/latency4_best_p2", 416, 3180776),
            new QuestSegLatencyModelSpec("best_p2_and_kd", "BenchmarkModels/latency4_best_p2_and_kd", 416, 3180776),
        };

        private class QuestSegLatencyModelSpec
        {
            public readonly string ModelName;
            public readonly string ResourcePath;
            public readonly int InputSize;
            public readonly int Params;

            public QuestSegLatencyModelSpec(string modelName, string resourcePath, int inputSize, int parameters)
            {
                ModelName = modelName;
                ResourcePath = resourcePath;
                InputSize = inputSize;
                Params = parameters;
            }
        }

        private class QuestSegLatencyModel
        {
            public readonly QuestSegLatencyModelSpec Spec;
            public readonly Worker Worker;

            public QuestSegLatencyModel(QuestSegLatencyModelSpec spec, Worker worker)
            {
                Spec = spec;
                Worker = worker;
            }
        }

        [Serializable]
        private class QuestSegLatencyModelInfo
        {
            public string model_name;
            public string resource;
            public int imgsz;
            public int parameters;
            public string backend;
        }

        [Serializable]
        private class ControlCommandMessage
        {
            public string type = "control";
            public string command;
            public int sequence;
            public string mode;
            public string backend;
            public bool rotate_order;
            public string sync_method;
            public string measurement_scope;
            public QuestSegLatencyModelInfo[] models;
        }

        [Serializable]
        private class QuestSegLatencyFrameMessage
        {
            public string type = "quest_local_seg_latency_frame";
            public string mode = "quest_local_sentis_forward_latency";
            public int frame_index;
            public double quest_realtime_sec;
            public int image_w;
            public int image_h;
            public int order_offset;
            public bool rotate_order;
            public string backend;
            public string sync_method;
            public QuestSegLatencyRecord[] records;
        }

        [Serializable]
        private class QuestSegLatencyRecord
        {
            public string model_name;
            public string resource;
            public int imgsz;
            public int parameters;
            public string backend;
            public int order_index;
            public int order_offset;
            public float latency_ms;
            public string output0_shape;
            public string output1_shape;
        }

        #region Unity Functions
        private IEnumerator Start()
        {
            // Wait for the UI to be ready because when Sentis load the model it will block the main thread.
            yield return new WaitForSeconds(0.05f);

            if (m_menuUi == null)
                m_menuUi = FindFirstObjectByType<DetectionUiMenuManager>();

            if (m_webCamTextureManager == null)
                m_webCamTextureManager = FindFirstObjectByType<WebCamTextureManager>();

            if (m_uiInference != null)
                m_uiInference.SetLabels(m_labelsAsset);

            if (m_streamOnly)
            {
                ApplySegLatencyModeOverrideFromAndroidIntent();
                NormalizeStreamSettings();
                if (m_questLocalSegLatencyMode)
                {
                    LoadQuestLocalSegLatencyModels();
                }
                IsModelLoaded = true;
                Connect();
            }
            else
            {
                LoadModel();
            }
        }

        private void Update()
        {
            if (m_streamOnly)
            {
                if (m_questLocalSegLatencyMode)
                {
                    QuestLocalLatencyTelemetryUpdate();
                }
                else
                {
                    StreamUpdate();
                }
            }
            else
            {
                InferenceUpdate();
            }
        }

        private void OnDestroy()
        {
            // Sentis cleanup
            if (m_schedule != null)
            {
                StopCoroutine(m_schedule);
            }
            m_input?.Dispose();
            m_engine?.Dispose();
            DisposeQuestLocalSegLatencyModels();

            // Stream cleanup
            Disconnect();
            if (_rt != null)
            {
                _rt.Release();
                Destroy(_rt);
                _rt = null;
            }
            if (_cpuTex != null)
            {
                Destroy(_cpuTex);
                _cpuTex = null;
            }
        }
        #endregion

        #region Public Functions
        public void RunInference(Texture targetTexture)
        {
            if (!targetTexture) return;

            // 仍然讓 UI 顯示相機畫面（不管哪個模式）
            if (m_uiInference != null)
                m_uiInference.SetDetectionCapture(targetTexture);

            if (m_streamOnly)
            {
                // Stream：只記住最新 texture，真正送幀由 Update 控制固定 FPS
                _latestTexture = targetTexture;
                return;
            }

            // ---- 原本 Sentis 推論流程（保留） ----
            if (!m_started)
            {
                m_input?.Dispose();
                m_input = TextureConverter.ToTensor(targetTexture, m_inputSize.x, m_inputSize.y, 3);
                m_schedule = m_engine.ScheduleIterable(m_input);
                m_download_state = 0;
                m_started = true;
            }
        }

        public bool IsRunning()
        {
            // 這個方法一定要留，DetectionManager 會用
            return m_started;
        }
        #endregion

        // =========================================================
        // ✅ Stream-only implementation
        // =========================================================
        private void ApplySegLatencyModeOverrideFromAndroidIntent()
        {
            // Default to Quest-local so old serialized scene values cannot accidentally select PC JPEG streaming.
            m_questLocalSegLatencyMode = true;

#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
                using var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
                using var intent = activity?.Call<AndroidJavaObject>("getIntent");
                var mode = intent?.Call<string>("getStringExtra", "seg_latency_mode");

                if (string.Equals(mode, "pc_stream", StringComparison.OrdinalIgnoreCase))
                {
                    m_questLocalSegLatencyMode = false;
                    Debug.Log("[SegLatencyMode] Android intent selected PC stream mode.");
                }
                else if (string.Equals(mode, "quest_local", StringComparison.OrdinalIgnoreCase))
                {
                    m_questLocalSegLatencyMode = true;
                    Debug.Log("[SegLatencyMode] Android intent selected Quest local mode.");
                }
                else
                {
                    Debug.Log("[SegLatencyMode] No Android intent override; using Quest local mode.");
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[SegLatencyMode] Failed to read Android intent override: {e.Message}");
            }
#endif
        }

        private void QuestLocalLatencyTelemetryUpdate()
        {
            HandleMeasurementToggleInput();
            HandleSceneResetInput();
            RefreshLatestStreamTexture();

            if (!_connected && m_autoReconnect && Time.unscaledTime >= _nextReconnectTime)
            {
                _nextReconnectTime = Time.unscaledTime + m_reconnectIntervalSec;
                Connect();
            }

            ProcessRecvQueue();
            UpdateLocalStreamDebug();
        }

        private void LoadQuestLocalSegLatencyModels()
        {
            DisposeQuestLocalSegLatencyModels();
            _questLatencyModelLoadFailures = 0;

            foreach (var spec in QuestSegLatencySpecs)
            {
                try
                {
                    var asset = Resources.Load<ModelAsset>(spec.ResourcePath);
                    if (asset == null)
                    {
                        _questLatencyModelLoadFailures++;
                        Debug.LogError($"[QuestLocalSegLatency] Missing ModelAsset resource: {spec.ResourcePath}");
                        continue;
                    }

                    var model = ModelLoader.Load(asset);
                    var worker = new Worker(model, m_questLocalSegLatencyBackend);
                    var localModel = new QuestSegLatencyModel(spec, worker);
                    _questLatencyModels.Add(localModel);

                    WarmupQuestLocalSegLatencyModel(localModel);
                    Debug.Log(
                        $"[QuestLocalSegLatency] Loaded {spec.ModelName} imgsz={spec.InputSize} " +
                        $"backend={m_questLocalSegLatencyBackend} resource={spec.ResourcePath}");
                }
                catch (Exception e)
                {
                    _questLatencyModelLoadFailures++;
                    Debug.LogError($"[QuestLocalSegLatency] Failed to load {spec.ModelName}: {e}");
                }
            }

            _questLatencyLastSummary =
                $"loaded={_questLatencyModels.Count}/{QuestSegLatencySpecs.Length} backend={m_questLocalSegLatencyBackend}";
        }

        private void DisposeQuestLocalSegLatencyModels()
        {
            foreach (var model in _questLatencyModels)
            {
                try { model.Worker?.Dispose(); } catch { }
            }
            _questLatencyModels.Clear();
        }

        private void WarmupQuestLocalSegLatencyModel(QuestSegLatencyModel model)
        {
            if (m_questLocalSegLatencyWarmupRuns <= 0)
                return;

            var shape = new TensorShape(1, 3, model.Spec.InputSize, model.Spec.InputSize);
            using var input = new Tensor<float>(shape, clearOnInit: true);

            for (var i = 0; i < m_questLocalSegLatencyWarmupRuns; i++)
            {
                model.Worker.Schedule(input);
                CompleteQuestLocalSegLatencyOutputs(model.Worker);
            }
        }

        private void ToggleQuestLocalSegLatencyMeasurement()
        {
            if (_questLatencyModels.Count == 0)
            {
                Debug.LogWarning("[QuestLocalSegLatency] Cannot start: no local Sentis models loaded.");
                return;
            }

            var nextActive = !_measurementActive;
            var command = nextActive ? "seg_latency_start" : "seg_latency_stop";
            var sequence = Interlocked.Increment(ref _measurementToggleRequests);

            if (!SendControlCommand(command, sequence, includeQuestLocalLatencyMetadata: true))
                return;

            _measurementActive = nextActive;
            var state = _measurementActive ? "START" : "STOP";
            Debug.Log($"[QuestLocalSegLatency] Measurement {state} requested by A button.");

            if (_measurementActive)
            {
                _questLatencyFrameIndex = 0;
                _questLatencyLastSummary = "recording";
                if (_questLatencyCoroutine != null)
                    StopCoroutine(_questLatencyCoroutine);
                _questLatencyCoroutine = StartCoroutine(QuestLocalSegLatencyLoop());
            }
            else
            {
                _questLatencyLastSummary = $"stopping at frames={_questLatencyFrameIndex}";
            }
        }

        private IEnumerator QuestLocalSegLatencyLoop()
        {
            while (_measurementActive)
            {
                RefreshLatestStreamTexture();
                if (_latestTexture == null || _latestTexture.width <= 16 || _latestTexture.height <= 16)
                {
                    yield return null;
                    continue;
                }

                var frameIndex = ++_questLatencyFrameIndex;
                var modelCount = _questLatencyModels.Count;
                var orderOffset = m_rotateQuestLocalLatencyOrder && modelCount > 0 ? (frameIndex - 1) % modelCount : 0;
                var records = new QuestSegLatencyRecord[modelCount];

                for (var orderIndex = 0; orderIndex < modelCount; orderIndex++)
                {
                    var modelIndex = (orderOffset + orderIndex) % modelCount;
                    records[orderIndex] = RunQuestLocalSegLatencyModel(_questLatencyModels[modelIndex], orderIndex, orderOffset);
                }

                var packet = new QuestSegLatencyFrameMessage
                {
                    frame_index = frameIndex,
                    quest_realtime_sec = Time.realtimeSinceStartup,
                    image_w = _latestTexture.width,
                    image_h = _latestTexture.height,
                    order_offset = orderOffset,
                    rotate_order = m_rotateQuestLocalLatencyOrder,
                    backend = m_questLocalSegLatencyBackend.ToString(),
                    sync_method = "input.CompleteAllPendingOperations before timer; output0/output1.CompleteAllPendingOperations after Schedule",
                    records = records
                };

                EnqueueJsonPacket(packet);
                _questLatencyLastSummary = FormatQuestLocalLatencySummary(frameIndex, records);

                if (m_questLocalLatencyMaxFrames > 0 && frameIndex >= m_questLocalLatencyMaxFrames)
                {
                    ToggleQuestLocalSegLatencyMeasurement();
                    yield break;
                }

                if (m_questLocalLatencyFrameIntervalSec > 0f)
                    yield return new WaitForSecondsRealtime(m_questLocalLatencyFrameIntervalSec);
                else
                    yield return null;
            }

            _questLatencyCoroutine = null;
        }

        private QuestSegLatencyRecord RunQuestLocalSegLatencyModel(QuestSegLatencyModel model, int orderIndex, int orderOffset)
        {
            using var input = BuildQuestLocalSegLatencyInput(_latestTexture, model.Spec.InputSize);
            input.CompleteAllPendingOperations();

            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            model.Worker.Schedule(input);
            var output0 = model.Worker.PeekOutput(0) as Tensor<float>;
            var output1 = model.Worker.PeekOutput(1) as Tensor<float>;
            CompleteQuestLocalSegLatencyOutput(output0);
            CompleteQuestLocalSegLatencyOutput(output1);
            var elapsedMs = (float)((System.Diagnostics.Stopwatch.GetTimestamp() - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);

            return new QuestSegLatencyRecord
            {
                model_name = model.Spec.ModelName,
                resource = model.Spec.ResourcePath,
                imgsz = model.Spec.InputSize,
                parameters = model.Spec.Params,
                backend = m_questLocalSegLatencyBackend.ToString(),
                order_index = orderIndex,
                order_offset = orderOffset,
                latency_ms = elapsedMs,
                output0_shape = output0 != null ? output0.shape.ToString() : "",
                output1_shape = output1 != null ? output1.shape.ToString() : ""
            };
        }

        private Tensor<float> BuildQuestLocalSegLatencyInput(Texture source, int inputSize)
        {
            var transform = new TextureTransform()
                .SetDimensions(inputSize, inputSize, 3)
                .SetTensorLayout(TensorLayout.NCHW)
                .SetCoordOrigin(CoordOrigin.TopLeft);
            return TextureConverter.ToTensor(source, transform);
        }

        private static void CompleteQuestLocalSegLatencyOutputs(Worker worker)
        {
            CompleteQuestLocalSegLatencyOutput(worker.PeekOutput(0) as Tensor<float>);
            CompleteQuestLocalSegLatencyOutput(worker.PeekOutput(1) as Tensor<float>);
        }

        private static void CompleteQuestLocalSegLatencyOutput(Tensor tensor)
        {
            tensor?.CompleteAllPendingOperations();
        }

        private string FormatQuestLocalLatencySummary(int frameIndex, QuestSegLatencyRecord[] records)
        {
            var sb = new StringBuilder();
            sb.Append($"frame={frameIndex}");
            if (m_rotateQuestLocalLatencyOrder)
            {
                var offset = records.Length > 0 ? records[0].order_offset : 0;
                sb.Append($" offset={offset}");
            }

            foreach (var record in records)
            {
                if (record == null)
                    continue;
                sb.Append($" | {record.model_name}={record.latency_ms:0.0}ms");
            }
            return sb.ToString();
        }

        private QuestSegLatencyModelInfo[] BuildQuestLocalLatencyModelInfo()
        {
            var infos = new QuestSegLatencyModelInfo[_questLatencyModels.Count];
            for (var i = 0; i < _questLatencyModels.Count; i++)
            {
                var spec = _questLatencyModels[i].Spec;
                infos[i] = new QuestSegLatencyModelInfo
                {
                    model_name = spec.ModelName,
                    resource = spec.ResourcePath,
                    imgsz = spec.InputSize,
                    parameters = spec.Params,
                    backend = m_questLocalSegLatencyBackend.ToString()
                };
            }
            return infos;
        }

        private bool EnqueueJsonPacket(object message)
        {
            if (!_connected)
                return false;

            try
            {
                var payload = Encoding.UTF8.GetBytes(JsonUtility.ToJson(message));
                _outgoingPackets.Enqueue(payload);
                Interlocked.Increment(ref _telemetryPacketsQueued);
                return true;
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _sendFailures);
                Debug.LogWarning($"[QuestLocalSegLatency] Failed to queue telemetry JSON: {e.Message}");
                return false;
            }
        }

        private void NormalizeStreamSettings()
        {
            if (!m_forceHighResolutionStreamDefaults)
                return;

            if (m_streamSize.x < 1280 || m_streamSize.y < 960)
            {
                m_streamSize = new Vector2Int(1280, 960);
            }
            m_jpegQuality = Mathf.Max(m_jpegQuality, 92);
            m_sendFps = Mathf.Clamp(m_sendFps, 1, 8);
        }

        private void StreamUpdate()
        {
            HandleMeasurementToggleInput();
            HandleSceneResetInput();
            RefreshLatestStreamTexture();

            // auto reconnect
            if (!_connected && m_autoReconnect && Time.unscaledTime >= _nextReconnectTime)
            {
                _nextReconnectTime = Time.unscaledTime + m_reconnectIntervalSec;
                Connect();
            }

            // fixed FPS send
            if (_connected && _latestTexture != null && Time.unscaledTime >= _nextSendTime)
            {
                _nextSendTime = Time.unscaledTime + (1f / Mathf.Max(1, m_sendFps));
                TryCaptureAndQueueJpeg(_latestTexture);
            }

            // receive & update UI (main thread)
            ProcessRecvQueue();

            UpdateLocalStreamDebug();
        }

        private void RefreshLatestStreamTexture()
        {
            if (m_webCamTextureManager == null)
            {
                m_webCamTextureManager = FindFirstObjectByType<WebCamTextureManager>();
            }

            var texture = m_webCamTextureManager != null ? m_webCamTextureManager.WebCamTexture : null;
            if (texture == null || texture.width <= 16 || texture.height <= 16)
            {
                return;
            }

            _latestTexture = texture;

            if (m_uiInference != null)
            {
                m_uiInference.SetDetectionCapture(texture);
            }
        }

        private void Connect()
        {
            Disconnect();

            try
            {
                Interlocked.Increment(ref _connectAttempts);
                _client = new TcpClient();
                _client.NoDelay = true;
                _client.Connect(m_serverIp, m_serverPort);
                _stream = _client.GetStream();
                try { _stream.ReadTimeout = 1000; _stream.WriteTimeout = 1000; } catch { }

                _netRunning = true;
                _connected = true;

                _sendThread = new Thread(SendLoop) { IsBackground = true };
                _sendThread.Start();

                _recvThread = new Thread(RecvLoop) { IsBackground = true };
                _recvThread.Start();

                Debug.Log($"[Stream] Connected to {m_serverIp}:{m_serverPort}");
                m_menuUi?.SetConnectionState(true, "Connected", null);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Stream] Connect failed: {e.Message}");
                m_menuUi?.SetConnectionState(false, "Connect failed", e.Message);
                _connected = false;
                _netRunning = false;
            }
        }

        private void Disconnect()
        {
            _netRunning = false;
            _connected = false;

            m_menuUi?.SetConnectionState(false, "Disconnected", null);

            try { _stream?.Close(); } catch { }
            try { _client?.Close(); } catch { }

            _stream = null;
            _client = null;

            try { _sendThread?.Join(200); } catch { }
            try { _recvThread?.Join(200); } catch { }

            _pendingJpeg = null;
            while (_outgoingPackets.TryDequeue(out _)) { }
            _captureInFlight = false;
            _measurementActive = false;
            if (_questLatencyCoroutine != null)
            {
                try { StopCoroutine(_questLatencyCoroutine); } catch { }
            }
            _questLatencyCoroutine = null;
        }

        private void HandleMeasurementToggleInput()
        {
            if (OVRInput.GetUp(m_measurementToggleButton) ||
                Input.GetKeyUp(KeyCode.JoystickButton0) ||
                Input.GetKeyUp(KeyCode.A))
            {
                if (m_questLocalSegLatencyMode)
                    ToggleQuestLocalSegLatencyMeasurement();
                else
                    SendSegLatencyMeasurementToggle();
            }
        }

        private void HandleSceneResetInput()
        {
            if (OVRInput.GetUp(m_resetSceneButton) || Input.GetKeyUp(KeyCode.X))
            {
                SendSceneResetControl();
            }
        }

        private bool SendControlCommand(string command, int sequence, bool includeQuestLocalLatencyMetadata = false)
        {
            if (_stream == null || !_connected)
                return false;

            try
            {
                var msg = new ControlCommandMessage
                {
                    command = command,
                    sequence = sequence,
                    mode = includeQuestLocalLatencyMetadata ? "quest_local_sentis_forward_latency" : "pc_stream",
                    backend = includeQuestLocalLatencyMetadata ? m_questLocalSegLatencyBackend.ToString() : "",
                    rotate_order = includeQuestLocalLatencyMetadata && m_rotateQuestLocalLatencyOrder,
                    sync_method = includeQuestLocalLatencyMetadata
                        ? "input.CompleteAllPendingOperations before timer; output0/output1.CompleteAllPendingOperations after Schedule"
                        : "",
                    measurement_scope = includeQuestLocalLatencyMetadata
                        ? "Quest local Sentis model forward only; texture-to-tensor/preprocess, postprocess, network, and UI excluded"
                        : "",
                    models = includeQuestLocalLatencyMetadata ? BuildQuestLocalLatencyModelInfo() : null
                };
                var payload = Encoding.UTF8.GetBytes(JsonUtility.ToJson(msg));
                WriteLengthPrefixed(payload);
                return true;
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _sendFailures);
                Debug.LogWarning($"[Stream] Control command '{command}' failed: {e.Message}");
                return false;
            }
        }

        private void SendSceneResetControl()
        {
            try
            {
                Interlocked.Increment(ref _sceneResetRequests);
                if (SendControlCommand("scene_reset", _sceneResetRequests))
                    Debug.Log("[Stream] Scene reset requested");
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _sendFailures);
                Debug.LogWarning($"[Stream] Scene reset control failed: {e.Message}");
            }
        }

        private void SendSegLatencyMeasurementToggle()
        {
            var nextActive = !_measurementActive;
            var command = nextActive ? "seg_latency_start" : "seg_latency_stop";
            var sequence = Interlocked.Increment(ref _measurementToggleRequests);

            if (SendControlCommand(command, sequence))
            {
                _measurementActive = nextActive;
                var state = _measurementActive ? "START" : "STOP";
                Debug.Log($"[Stream] Seg latency measurement {state} requested by A button.");
            }
        }

        private void EnsureBuffers()
        {
            if (_rt == null || _rt.width != m_streamSize.x || _rt.height != m_streamSize.y)
            {
                if (_rt != null)
                {
                    _rt.Release();
                    Destroy(_rt);
                }
                _rt = new RenderTexture(m_streamSize.x, m_streamSize.y, 0, RenderTextureFormat.ARGB32);
                _rt.Create();
            }

            if (_cpuTex == null || _cpuTex.width != m_streamSize.x || _cpuTex.height != m_streamSize.y)
            {
                if (_cpuTex != null) Destroy(_cpuTex);
                _cpuTex = new Texture2D(m_streamSize.x, m_streamSize.y, TextureFormat.RGB24, false);
            }
        }

        private void TryCaptureAndQueueJpeg(Texture src)
        {
            // 丟幀：上一張還沒送出去就不抓新的
            if (_pendingJpeg != null) return;
            if (_captureInFlight) return;

            EnsureBuffers();
            _captureInFlight = true;

            // resize
            Graphics.Blit(src, _rt);

            if (SystemInfo.supportsAsyncGPUReadback)
            {
                AsyncGPUReadback.Request(_rt, 0, TextureFormat.RGB24, OnReadback);
            }
            else
            {
                FallbackReadPixelsAndEncode();
                _captureInFlight = false;
            }
        }

        private void OnReadback(AsyncGPUReadbackRequest req)
        {
            _captureInFlight = false;
            if (req.hasError) return;

            try
            {
                var data = req.GetData<byte>();
                _cpuTex.LoadRawTextureData(data);
                _cpuTex.Apply(false);
                _pendingJpeg = ImageConversion.EncodeToJPG(_cpuTex, m_jpegQuality);
                Interlocked.Increment(ref _framesQueued);
            }
            catch { }
        }

        private void FallbackReadPixelsAndEncode()
        {
            var prev = RenderTexture.active;
            RenderTexture.active = _rt;

            _cpuTex.ReadPixels(new Rect(0, 0, _rt.width, _rt.height), 0, 0, false);
            _cpuTex.Apply(false);

            RenderTexture.active = prev;

            _pendingJpeg = ImageConversion.EncodeToJPG(_cpuTex, m_jpegQuality);
            Interlocked.Increment(ref _framesQueued);
        }

        private void WriteLengthPrefixed(byte[] payload)
        {
            if (payload == null || _stream == null)
                return;

            int len = payload.Length;

            byte[] header = new byte[4];
            header[0] = (byte)((len >> 24) & 0xFF);
            header[1] = (byte)((len >> 16) & 0xFF);
            header[2] = (byte)((len >> 8) & 0xFF);
            header[3] = (byte)(len & 0xFF);

            byte[] packet = new byte[4 + len];
            Buffer.BlockCopy(header, 0, packet, 0, 4);
            Buffer.BlockCopy(payload, 0, packet, 4, len);

            lock (_sendLock)
            {
                if (_stream == null)
                    return;

                _stream.Write(packet, 0, packet.Length);
                _stream.Flush();
            }
        }

        private void SendLoop()
        {
            try
            {
                while (_netRunning && _stream != null)
                {
                    var jpg = _pendingJpeg;
                    if (_outgoingPackets.TryDequeue(out var payload))
                    {
                        WriteLengthPrefixed(payload);
                        Interlocked.Increment(ref _telemetryPacketsSent);
                        continue;
                    }

                    if (jpg == null)
                    {
                        Thread.Sleep(1);
                        continue;
                    }

                    _pendingJpeg = null;

                    int len = jpg.Length;
                    WriteLengthPrefixed(jpg);

                    Interlocked.Increment(ref _framesSent);
                    Interlocked.Exchange(ref _lastSentBytes, len);
                    Debug.Log($"[Stream] Sent frame: {len / 1024f:0.0} KB");
                }
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _sendFailures);
                Debug.LogWarning($"[Stream] SendLoop stopped: {e.Message}");
                m_menuUi?.SetConnectionState(false, "SendLoop stopped", e.Message);
            }

            _connected = false;
            _netRunning = false;
        }

        // -----------------------------
        // ✅ Receive JSON results from PC
        // -----------------------------
        [Serializable]
        private class ServerResponse
        {
            public double ts;
            public int img_w;
            public int img_h;
            public SentisInferenceUiManager.RemoteTile[] tiles;
            public string[] hand;
            public bool hand_stable;
            public SentisInferenceUiManager.Advice advice;
            public string pc_log;
        }

        private static bool ReadExact(NetworkStream stream, byte[] buf, int offset, int count)
        {
            int got = 0;
            while (got < count)
            {
                int n;
                try
                {
                    n = stream.Read(buf, offset + got, count - got);
                }
                catch (IOException)
                {
                    return false; // timeout or stream error
                }
                if (n <= 0) return false;
                got += n;
            }
            return true;
        }

        private void RecvLoop()
        {
            try
            {
                var header = new byte[4];

                while (_netRunning && _stream != null)
                {
                    // read length prefix (big-endian)
                    if (!ReadExact(_stream, header, 0, 4))
                    {
                        Thread.Sleep(1);
                        continue;
                    }

                    int len = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
                    if (len <= 0 || len > 50_000_000)
                    {
                        Debug.LogWarning($"[Stream] Invalid packet len={len}");
                        Thread.Sleep(1);
                        continue;
                    }

                    var payload = new byte[len];
                    if (!ReadExact(_stream, payload, 0, len))
                    {
                        Thread.Sleep(1);
                        continue;
                    }

                    string json = Encoding.UTF8.GetString(payload);
                    _recvJsonQueue.Enqueue(json);
                }
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _recvFailures);
                Debug.LogWarning($"[Stream] RecvLoop stopped: {e.Message}");
                m_menuUi?.SetConnectionState(false, "RecvLoop stopped", e.Message);
            }
        }

        private void ProcessRecvQueue()
        {
            if (m_uiInference == null && m_menuUi == null) return;

            // keep latest only
            string latest = null;
            while (_recvJsonQueue.TryDequeue(out var js))
                latest = js;

            if (string.IsNullOrEmpty(latest)) return;

            try
            {
                var resp = JsonUtility.FromJson<ServerResponse>(latest);
                if (resp == null) return;
                Interlocked.Increment(ref _responsesReceived);

                // Update prompt panel (preferred)
                if (m_menuUi != null)
                {
                    if (!string.IsNullOrEmpty(resp.pc_log))
                        m_menuUi.SetPcLog(resp.pc_log);
                    else
                        m_menuUi.SetMahjongAdvice(resp.advice, resp.hand, resp.ts);
                }

                if (m_uiInference != null)
                {
                    // draw boxes + update hand
                    m_uiInference.DrawRemoteBoxes(resp.tiles, resp.img_w, resp.img_h, resp.hand);

                    // update advice (2 buttons)
                    m_uiInference.SetAdvice(resp.advice);
                }
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref _jsonParseErrors);
                Debug.LogWarning($"[Stream] JSON parse failed: {e.Message}");
            }
        }

        private void UpdateLocalStreamDebug()
        {
            if (!m_showStreamDebug || m_menuUi == null) return;
            if (Time.unscaledTime < _nextDebugUiTime) return;
            _nextDebugUiTime = Time.unscaledTime + Mathf.Max(0.1f, m_debugUiIntervalSec);

            if (m_questLocalSegLatencyMode)
            {
                string questLocalDebug =
                    $"[QUEST] localSentis={(_connected ? "connected" : "disconnected")} " +
                    $"backend={m_questLocalSegLatencyBackend} models={_questLatencyModels.Count}/{QuestSegLatencySpecs.Length} " +
                    $"loadErr={_questLatencyModelLoadFailures} recording={(_measurementActive ? "yes" : "no")} " +
                    $"frames={_questLatencyFrameIndex} rotateOrder={(m_rotateQuestLocalLatencyOrder ? "yes" : "no")} " +
                    $"queuedTelemetry={_telemetryPacketsQueued} sentTelemetry={_telemetryPacketsSent} recv={_responsesReceived} " +
                    $"toggles={_measurementToggleRequests} connects={_connectAttempts} sendErr={_sendFailures} recvErr={_recvFailures} " +
                    $"last={_questLatencyLastSummary}";
                m_menuUi.SetStreamDebug(questLocalDebug);
                return;
            }

            string pending = _pendingJpeg != null ? "yes" : "no";
            string capture = _captureInFlight ? "yes" : "no";
            string debug =
                $"[QUEST] stream={(_connected ? "connected" : "disconnected")} " +
                $"targetFps={m_sendFps} size={m_streamSize.x}x{m_streamSize.y} q={m_jpegQuality} " +
                $"queued={_framesQueued} sent={_framesSent} recv={_responsesReceived} " +
                $"sceneReset={_sceneResetRequests} segLatency={(_measurementActive ? "recording" : "idle")} toggles={_measurementToggleRequests} " +
                $"pending={pending} capture={capture} lastKB={_lastSentBytes / 1024f:0.0} " +
                $"connects={_connectAttempts} sendErr={_sendFailures} recvErr={_recvFailures} jsonErr={_jsonParseErrors}";
            m_menuUi.SetStreamDebug(debug);
        }

        // =========================================================
        // ✅ Original Sentis functions (保留)
        // =========================================================
        private void LoadModel()
        {
            var model = ModelLoader.Load(m_sentisModel);
            Debug.Log($"Sentis model loaded correctly with iouThreshold: {m_iouThreshold} and scoreThreshold: {m_scoreThreshold}");
            m_engine = new Worker(model, m_backend);

            var input = TextureConverter.ToTensor(new Texture2D(m_inputSize.x, m_inputSize.y), m_inputSize.x, m_inputSize.y, 3);
            m_engine.Schedule(input);
            IsModelLoaded = true;
        }

        private void InferenceUpdate()
        {
            if (m_started)
            {
                try
                {
                    if (m_download_state == 0)
                    {
                        var it = 0;
                        while (m_schedule.MoveNext())
                        {
                            if (++it % m_layersPerFrame == 0)
                                return;
                        }
                        m_download_state = 1;
                    }
                    else
                    {
                        GetInferencesResults();
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Sentis error: {e.Message}");
                }
            }
        }

        private void PollRequestOuput()
        {
            m_pullOutput = m_engine.PeekOutput(0) as Tensor<float>;
            if (m_pullOutput.dataOnBackend != null)
            {
                m_pullOutput.ReadbackRequest();
                m_isWaiting = true;
            }
            else
            {
                Debug.LogError("Sentis: No data output m_output");
                m_download_state = 4;
            }
        }

        private void PollRequestLabelIDs()
        {
            m_pullLabelIDs = m_engine.PeekOutput(1) as Tensor<int>;
            if (m_pullLabelIDs.dataOnBackend != null)
            {
                m_pullLabelIDs.ReadbackRequest();
                m_isWaiting = true;
            }
            else
            {
                Debug.LogError("Sentis: No data output m_labelIDs");
                m_download_state = 4;
            }
        }

        private void GetInferencesResults()
        {
            switch (m_download_state)
            {
                case 1:
                    if (!m_isWaiting) PollRequestOuput();
                    else
                    {
                        if (m_pullOutput.IsReadbackRequestDone())
                        {
                            m_output = m_pullOutput.ReadbackAndClone();
                            m_isWaiting = false;

                            if (m_output.shape[0] > 0) m_download_state = 2;
                            else m_download_state = 4;
                        }
                    }
                    break;

                case 2:
                    if (!m_isWaiting) PollRequestLabelIDs();
                    else
                    {
                        if (m_pullLabelIDs.IsReadbackRequestDone())
                        {
                            m_labelIDs = m_pullLabelIDs.ReadbackAndClone();
                            m_isWaiting = false;

                            if (m_labelIDs.shape[0] > 0) m_download_state = 3;
                            else m_download_state = 4;
                        }
                    }
                    break;

                case 3:
                    m_uiInference.DrawUIBoxes(m_output, m_labelIDs, m_inputSize.x, m_inputSize.y);
                    m_download_state = 5;
                    break;

                case 4:
                    m_uiInference.OnObjectDetectionError();
                    m_download_state = 5;
                    break;

                case 5:
                    m_download_state++;
                    m_started = false;
                    m_output?.Dispose();
                    m_labelIDs?.Dispose();
                    break;
            }
        }
    }
}
