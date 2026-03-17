// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections;
using System.Collections.Concurrent;
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

        [Tooltip("你要每秒一張就填 1")]
        [SerializeField, Range(1, 30)] private int m_sendFps = 1;

        [SerializeField, Range(10, 100)] private int m_jpegQuality = 85;

        [Tooltip("固定輸出尺寸（會把來源 Blit 到這個大小）")]
        [SerializeField] private Vector2Int m_streamSize = new(640, 640);

        [SerializeField] private bool m_autoReconnect = true;
        [SerializeField] private float m_reconnectIntervalSec = 2f;

        // Networking
        private TcpClient _client;
        private NetworkStream _stream;
        private Thread _sendThread;
        private Thread _recvThread;
        private volatile bool _netRunning;
        private volatile bool _connected;
        private readonly object _sendLock = new object();
        private readonly ConcurrentQueue<string> _recvJsonQueue = new ConcurrentQueue<string>();

        // Capture
        private Texture _latestTexture;
        private RenderTexture _rt;
        private Texture2D _cpuTex;
        private volatile bool _captureInFlight;
        private volatile byte[] _pendingJpeg; // 丟幀：不為 null 代表上一張還沒送出

        private float _nextSendTime;
        private float _nextReconnectTime;

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

        #region Unity Functions
        private IEnumerator Start()
        {
            // Wait for the UI to be ready because when Sentis load the model it will block the main thread.
            yield return new WaitForSeconds(0.05f);

            if (m_menuUi == null)
                m_menuUi = FindFirstObjectByType<DetectionUiMenuManager>();

            if (m_uiInference != null)
                m_uiInference.SetLabels(m_labelsAsset);

            if (m_streamOnly)
            {
                // Stream only：不載 Sentis
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
                StreamUpdate();
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
        private void StreamUpdate()
        {
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
        }

        private void Connect()
        {
            Disconnect();

            try
            {
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
            _captureInFlight = false;
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
        }

        private void SendLoop()
        {
            try
            {
                while (_netRunning && _stream != null)
                {
                    var jpg = _pendingJpeg;
                    if (jpg == null)
                    {
                        Thread.Sleep(1);
                        continue;
                    }

                    _pendingJpeg = null;

                    int len = jpg.Length;

                    // 4-byte big-endian length prefix
                    byte[] header = new byte[4];
                    header[0] = (byte)((len >> 24) & 0xFF);
                    header[1] = (byte)((len >> 16) & 0xFF);
                    header[2] = (byte)((len >> 8) & 0xFF);
                    header[3] = (byte)(len & 0xFF);

                    lock (_sendLock)
                    {
                        _stream.Write(header, 0, 4);
                        _stream.Write(jpg, 0, jpg.Length);
                        _stream.Flush();
                    }

                    Debug.Log($"[Stream] Sent frame: {len / 1024f:0.0} KB");
                }
            }
            catch (Exception e)
            {
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
                Debug.LogWarning($"[Stream] JSON parse failed: {e.Message}");
            }
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
