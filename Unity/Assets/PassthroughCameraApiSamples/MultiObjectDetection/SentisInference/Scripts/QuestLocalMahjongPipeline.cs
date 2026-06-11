using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Unity.Sentis;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    public sealed class QuestLocalMahjongPipeline : IDisposable
    {
        private const int ObsDim = 175;
        private const int ActionDim = 39;
        private const int ActPass = 34;
        private const int ActPon = 35;
        private const int ActChiLow = 36;
        private const int ActChiMid = 37;
        private const int ActChiHigh = 38;

        private const int PhaseDiscard = 0;
        private const int PhaseClaim = 1;
        private const int TurnCorrectionCooldownFrames = 15;
        private const float TableMatchDist = 0.12f;
        private const int LatencyLogFlushRows = 5;
        private const string LatencyLogFolderName = "MahjongLatencyLogs";
        private static readonly char[] CsvSpecialChars = { ',', '"', '\n', '\r' };

        private static readonly string[] TileLabels =
        {
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
            "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
            "east", "south", "west", "north", "white", "green", "red"
        };

        // Classifier order from server/mahjong_labels.txt.
        private static readonly string[] ClassifierLabels =
        {
            "1m", "1p", "1s", "2m", "2p", "2s", "3m", "3p", "3s",
            "4m", "4p", "4s", "5m", "5p", "5s", "6m", "6p", "6s",
            "7m", "7p", "7s", "8m", "8p", "8s", "9m", "9p", "9s",
            "east", "flower", "green", "north", "red", "south", "west", "white"
        };

        private readonly BackendType _backend;
        private readonly int _segInputSize;
        private readonly int _clsInputSize;
        private readonly float _detScoreThreshold;
        private readonly float _detIouThreshold;
        private readonly float _clsConfThreshold;
        private readonly int _handStableFrames;
        private readonly int _tableStableFrames;

        private Worker _segWorker;
        private Worker _clsWorker;
        private Worker _ppoWorker;
        private Texture2D _frameTexture;

        private int _nextTrackId;
        private readonly List<TrackState> _tracks = new List<TrackState>();

        private readonly Queue<string> _recentHandSignatures = new Queue<string>();
        private string _stableHandSignature = "";
        private List<int> _stableHandTiles = new List<int>();
        private List<string> _stableHandLabels = new List<string>();
        private bool _stableHandActive;
        private int _stableRecommendedDiscard = -1;

        private string _tableCandidateSignature = "";
        private int _tableCandidateFrames;
        private List<TableObs> _tableCandidateObs = new List<TableObs>();
        private string _stableTableSignature = "";
        private List<TableObs> _stableTableObs = new List<TableObs>();
        private bool _hasStableTable;

        private readonly List<int>[] _discards =
        {
            new List<int>(), new List<int>(), new List<int>(), new List<int>()
        };

        private int _frameIndex;
        private int _cooldownFramesRemaining;
        private int _currentPlayer;
        private int _phase = PhaseDiscard;
        private int _expectedDiscarder;
        private int _lastDiscard = -1;
        private int _lastDiscarder = -1;
        private int _pendingSelfTableTile = -1;
        private bool _selfDiscardTurnActive;
        private bool _resetTableBaselineOnNextStable;
        private int _stableHandCount = -1;
        private string _lastFlowEvent = "";

        private bool _loaded;
        private string _lastError = "";
        private bool _latencyLogEnabled;
        private string _latencyLogPath = "";
        private readonly StringBuilder _latencyLogBuffer = new StringBuilder(8192);
        private int _latencyRowsBuffered;
        private int _latencySamples;
        private bool _latencyLogErrorReported;

        public bool IsLoaded { get { return _loaded; } }
        public string LastError { get { return _lastError; } }
        public string LatencyLogPath { get { return _latencyLogPath; } }
        public int LatencySampleCount { get { return _latencySamples; } }

        public QuestLocalMahjongPipeline(
            BackendType backend,
            int segInputSize,
            int clsInputSize,
            float detScoreThreshold,
            float detIouThreshold,
            float clsConfThreshold,
            int handStableFrames,
            int tableStableFrames)
        {
            _backend = backend;
            _segInputSize = Mathf.Max(32, segInputSize);
            _clsInputSize = Mathf.Max(32, clsInputSize);
            _detScoreThreshold = Mathf.Clamp01(detScoreThreshold);
            _detIouThreshold = Mathf.Clamp01(detIouThreshold);
            _clsConfThreshold = Mathf.Clamp01(clsConfThreshold);
            _handStableFrames = Mathf.Max(1, handStableFrames);
            _tableStableFrames = Mathf.Max(1, tableStableFrames);
        }

        public bool Load(string segmentationResource, string classifierResource, string ppoResource)
        {
            try
            {
                Debug.Log("[QuestLocal] Loading segmentation resource: " + segmentationResource);
                var segAsset = Resources.Load<ModelAsset>(segmentationResource);
                Debug.Log("[QuestLocal] Loading classifier resource: " + classifierResource);
                var clsAsset = Resources.Load<ModelAsset>(classifierResource);
                Debug.Log("[QuestLocal] Loading PPO resource: " + ppoResource);
                var ppoAsset = Resources.Load<ModelAsset>(ppoResource);

                if (segAsset == null)
                    throw new InvalidOperationException("Missing segmentation model: " + segmentationResource);
                if (clsAsset == null)
                    throw new InvalidOperationException("Missing classifier model: " + classifierResource);
                if (ppoAsset == null)
                    throw new InvalidOperationException("Missing PPO model: " + ppoResource);

                Debug.Log("[QuestLocal] Creating segmentation worker");
                _segWorker = new Worker(ModelLoader.Load(segAsset), _backend);
                Debug.Log("[QuestLocal] Creating classifier worker");
                _clsWorker = new Worker(ModelLoader.Load(clsAsset), _backend);
                Debug.Log("[QuestLocal] Creating PPO worker");
                _ppoWorker = new Worker(ModelLoader.Load(ppoAsset), _backend);
                _loaded = true;
                _lastError = "";
                InitializeLatencyLog();
                Debug.Log("[QuestLocal] Models loaded");
                return true;
            }
            catch (Exception e)
            {
                _lastError = e.Message;
                _loaded = false;
                _latencyLogEnabled = false;
                Debug.LogError("[QuestLocal] Load failed: " + e);
                return false;
            }
        }

        public void ResetScene()
        {
            FlushLatencyLog();
            _tracks.Clear();
            _nextTrackId = 0;
            _recentHandSignatures.Clear();
            _stableHandSignature = "";
            _stableHandTiles.Clear();
            _stableHandLabels.Clear();
            _stableHandActive = false;
            _stableRecommendedDiscard = -1;
            _tableCandidateSignature = "";
            _tableCandidateFrames = 0;
            _tableCandidateObs.Clear();
            _stableTableSignature = "";
            _stableTableObs.Clear();
            _hasStableTable = false;
            foreach (var pile in _discards)
                pile.Clear();
            _frameIndex = 0;
            _cooldownFramesRemaining = 0;
            _currentPlayer = 0;
            _phase = PhaseDiscard;
            _expectedDiscarder = 0;
            _lastDiscard = -1;
            _lastDiscarder = -1;
            _pendingSelfTableTile = -1;
            _selfDiscardTurnActive = false;
            _resetTableBaselineOnNextStable = false;
            _stableHandCount = -1;
            _lastFlowEvent = "scene_reset";
        }

        public LocalResult Run(Texture source)
        {
            var result = new LocalResult();
            if (!_loaded || source == null)
            {
                result.PcLog = BuildLog(
                    new List<string>(),
                    StableTableLabels(),
                    "Detecting",
                    "",
                    false,
                    0,
                    0,
                    0,
                    0,
                    default);
                result.Advice = EmptyAdvice();
                return result;
            }

            var start = Time.realtimeSinceStartup;
            try
            {
                _frameIndex++;
                if (_cooldownFramesRemaining > 0)
                    _cooldownFramesRemaining--;

                var frameStart = Time.realtimeSinceStartup;
                EnsureFrameTexture(source);
                var frameMs = (Time.realtimeSinceStartup - frameStart) * 1000f;

                var segStart = Time.realtimeSinceStartup;
                var detections = Detect(source);
                var segMs = (Time.realtimeSinceStartup - segStart) * 1000f;

                var clsStart = Time.realtimeSinceStartup;
                var classified = ClassifyDetections(detections);
                var clsMs = (Time.realtimeSinceStartup - clsStart) * 1000f;

                var trackStart = Time.realtimeSinceStartup;
                var tracked = UpdateTracks(classified);
                var trackMs = (Time.realtimeSinceStartup - trackStart) * 1000f;

                var remoteTiles = new List<SentisInferenceUiManager.RemoteTile>();
                var handLabels = new List<string>();
                var handTiles = new List<int>();
                var tableObs = new List<TableObs>();

                foreach (var tr in tracked)
                {
                    remoteTiles.Add(new SentisInferenceUiManager.RemoteTile
                    {
                        id = tr.Id,
                        cls = tr.Label,
                        conf = tr.Confidence,
                        cx = tr.Cx,
                        cy = tr.Cy,
                        w = tr.W,
                        h = tr.H
                    });

                    var tileId = TileLabelToId(tr.Label);
                    if (tileId < 0)
                        continue;

                    if (tr.Area == "hand")
                    {
                        handLabels.Add(tr.Label);
                        handTiles.Add(tileId);
                    }
                    else if (tr.Area == "table")
                    {
                        tableObs.Add(new TableObs
                        {
                            TrackId = tr.Id,
                            TileId = tileId,
                            Label = TileIdToLabel(tileId),
                            Cx = tr.Cx,
                            Cy = tr.Cy
                        });
                    }
                }

                handTiles.Sort();
                handLabels = handTiles.Select(TileIdToLabel).ToList();
                tableObs.Sort((a, b) =>
                {
                    var c = a.Cx.CompareTo(b.Cx);
                    if (c != 0) return c;
                    c = a.Cy.CompareTo(b.Cy);
                    if (c != 0) return c;
                    return a.TileId.CompareTo(b.TileId);
                });

                var selfDiscard = TryCompleteSelfDiscard(handTiles);
                if (selfDiscard >= 0)
                    CompleteSelfDiscard(selfDiscard);

                var handStable = UpdateHandStability(handTiles, handLabels);
                UpdateTableStability(tableObs);

                if (handStable && IsStableSelfTurn(handTiles) && !_selfDiscardTurnActive)
                    CorrectToSelfTurn(handTiles, handLabels);

                var effectiveHandTiles = _stableHandActive ? new List<int>(_stableHandTiles) : handTiles;
                var effectiveHandLabels = _stableHandActive ? new List<string>(_stableHandLabels) : handLabels;

                var ppoStart = Time.realtimeSinceStartup;
                var advice = BuildAdvice(effectiveHandTiles, handStable);
                var ppoMs = (Time.realtimeSinceStartup - ppoStart) * 1000f;
                if (_phase == PhaseDiscard && _currentPlayer == 0 && advice.benefit != null)
                    _stableRecommendedDiscard = advice.benefit.tile_id;

                var action = advice.benefit != null ? advice.benefit.tile : "";
                var stable = _stableHandActive || _hasStableTable;
                var elapsedMs = (Time.realtimeSinceStartup - start) * 1000f;
                var latency = new LatencyInfo
                {
                    FrameMs = frameMs,
                    SegMs = segMs,
                    ClsMs = clsMs,
                    TrackMs = trackMs,
                    PpoMs = ppoMs,
                    TotalMs = elapsedMs
                };
                RecordLatencySample(
                    latency,
                    detections.Count,
                    classified.Count,
                    tracked.Count,
                    handTiles.Count,
                    tableObs.Count,
                    _stableTableObs.Count,
                    stable,
                    action);
                result.Tiles = remoteTiles.ToArray();
                result.Hand = effectiveHandLabels.ToArray();
                result.Advice = advice;
                result.PcLog = BuildLog(
                    effectiveHandLabels,
                    StableTableLabels(),
                    string.IsNullOrEmpty(action) ? "Detecting" : action,
                    LastDiscardText(),
                    stable,
                    tracked.Count,
                    tableObs.Count,
                    detections.Count,
                    classified.Count,
                    latency);
                return result;
            }
            catch (Exception e)
            {
                _lastError = e.Message;
                Debug.LogWarning("[QuestLocal] Run failed: " + e);
                result.Advice = EmptyAdvice();
                result.PcLog = "Mode: Quest local\nError: " + e.Message;
                return result;
            }
        }

        public void Dispose()
        {
            FlushLatencyLog();
            _segWorker?.Dispose();
            _clsWorker?.Dispose();
            _ppoWorker?.Dispose();
            if (_frameTexture != null)
            {
                UnityEngine.Object.Destroy(_frameTexture);
                _frameTexture = null;
            }
        }

        private List<Detection> Detect(Texture source)
        {
            using (var input = TextureConverter.ToTensor(source, _segInputSize, _segInputSize, 3))
            {
                _segWorker.Schedule(input);

                var output = _segWorker.PeekOutput(0).ReadbackAndClone() as Tensor<float>;
                if (output == null)
                    return new List<Detection>();

                try
                {
                    var data = output.DownloadToArray();
                    var channels = output.shape[1];
                    var count = output.shape[2];
                    var candidates = new List<Detection>();

                    for (var i = 0; i < count; i++)
                    {
                        var handScore = ReadYolo(data, channels, count, 4, i);
                        var tableScore = ReadYolo(data, channels, count, 5, i);
                        var clsId = handScore >= tableScore ? 0 : 1;
                        var score = Mathf.Max(handScore, tableScore);
                        if (score < _detScoreThreshold)
                            continue;

                        var cx = ReadYolo(data, channels, count, 0, i) / _segInputSize;
                        var cy = ReadYolo(data, channels, count, 1, i) / _segInputSize;
                        var w = ReadYolo(data, channels, count, 2, i) / _segInputSize;
                        var h = ReadYolo(data, channels, count, 3, i) / _segInputSize;
                        if (w <= 0.001f || h <= 0.001f)
                            continue;

                        candidates.Add(new Detection
                        {
                            Cx = Mathf.Clamp01(cx),
                            Cy = Mathf.Clamp01(cy),
                            W = Mathf.Clamp01(w),
                            H = Mathf.Clamp01(h),
                            Score = score,
                            Area = clsId == 0 ? "hand" : "table"
                        });
                    }

                    candidates.Sort((a, b) => b.Score.CompareTo(a.Score));
                    return Nms(candidates, 48);
                }
                finally
                {
                    output.Dispose();
                }
            }
        }

        private List<ClassifiedDetection> ClassifyDetections(List<Detection> detections)
        {
            var result = new List<ClassifiedDetection>();
            foreach (var det in detections)
            {
                var crop = MakeCrop(det);
                if (crop == null)
                    continue;

                try
                {
                    var label = "UNKNOWN";
                    var conf = 0f;
                    using (var input = TextureConverter.ToTensor(crop, _clsInputSize, _clsInputSize, 3))
                    {
                        _clsWorker.Schedule(input);

                        var output = _clsWorker.PeekOutput().ReadbackAndClone() as Tensor<float>;
                        if (output != null)
                        {
                            try
                            {
                                var data = output.DownloadToArray();
                                var best = ArgMax(data);
                                if (best >= 0 && best < ClassifierLabels.Length)
                                {
                                    label = ClassifierLabels[best];
                                    conf = data[best];
                                }
                            }
                            finally
                            {
                                output.Dispose();
                            }
                        }
                    }

                    if (label != "UNKNOWN" && label != "flower" && conf >= _clsConfThreshold)
                    {
                        result.Add(new ClassifiedDetection
                        {
                            Cx = det.Cx,
                            Cy = det.Cy,
                            W = det.W,
                            H = det.H,
                            Area = det.Area,
                            Label = label,
                            Confidence = conf
                        });
                    }
                }
                finally
                {
                    UnityEngine.Object.Destroy(crop);
                }
            }
            return result;
        }

        private Texture2D MakeCrop(Detection det)
        {
            if (_frameTexture == null)
                return null;

            var imgW = _frameTexture.width;
            var imgH = _frameTexture.height;
            var x1 = Mathf.Clamp(Mathf.RoundToInt((det.Cx - det.W * 0.5f) * imgW), 0, imgW - 1);
            var x2 = Mathf.Clamp(Mathf.RoundToInt((det.Cx + det.W * 0.5f) * imgW), 0, imgW);
            var yTop = Mathf.Clamp(Mathf.RoundToInt((det.Cy - det.H * 0.5f) * imgH), 0, imgH - 1);
            var yBottom = Mathf.Clamp(Mathf.RoundToInt((det.Cy + det.H * 0.5f) * imgH), 0, imgH);
            var y1 = Mathf.Clamp(imgH - yBottom, 0, imgH - 1);
            var width = Mathf.Clamp(x2 - x1, 1, imgW - x1);
            var height = Mathf.Clamp(yBottom - yTop, 1, imgH - y1);

            var pixels = _frameTexture.GetPixels(x1, y1, width, height);
            var crop = new Texture2D(width, height, TextureFormat.RGB24, false);
            crop.SetPixels(pixels);
            crop.Apply(false);
            return crop;
        }

        private List<TrackState> UpdateTracks(List<ClassifiedDetection> detections)
        {
            var now = Time.realtimeSinceStartup;
            var used = new HashSet<int>();

            foreach (var det in detections)
            {
                var best = -1;
                var bestIou = 0f;
                for (var i = 0; i < _tracks.Count; i++)
                {
                    if (used.Contains(i))
                        continue;
                    var score = Iou(_tracks[i], det);
                    if (score > bestIou)
                    {
                        bestIou = score;
                        best = i;
                    }
                }

                if (best >= 0 && bestIou >= 0.25f)
                {
                    _tracks[best].Update(det, now);
                    used.Add(best);
                }
                else
                {
                    _tracks.Add(new TrackState(_nextTrackId++, det, now));
                }
            }

            _tracks.RemoveAll(t => now - t.LastSeen > 0.45f);
            return new List<TrackState>(_tracks);
        }

        private bool UpdateHandStability(List<int> handTiles, List<string> handLabels)
        {
            var sig = Signature(handTiles);
            _recentHandSignatures.Enqueue(sig);
            while (_recentHandSignatures.Count > _handStableFrames)
                _recentHandSignatures.Dequeue();

            if (_recentHandSignatures.Count < _handStableFrames)
                return false;

            foreach (var item in _recentHandSignatures)
            {
                if (item != sig)
                    return false;
            }

            _stableHandSignature = sig;
            _stableHandCount = handTiles.Count;
            if (IsSelfTurnCount(handTiles.Count))
            {
                _stableHandActive = true;
                _stableHandTiles = new List<int>(handTiles);
                _stableHandLabels = new List<string>(handLabels);
            }
            else
            {
                _selfDiscardTurnActive = false;
            }
            return true;
        }

        private int TryCompleteSelfDiscard(List<int> liveHandTiles)
        {
            if (!_stableHandActive || _stableRecommendedDiscard < 0)
                return -1;
            if (!IsSelfTurnCount(_stableHandTiles.Count))
                return -1;
            if (liveHandTiles.Count != _stableHandTiles.Count - 1 || liveHandTiles.Count % 3 != 1)
                return -1;
            if (CountTile(liveHandTiles, _stableRecommendedDiscard) >= CountTile(_stableHandTiles, _stableRecommendedDiscard))
                return -1;

            var discarded = _stableRecommendedDiscard;
            _stableHandActive = false;
            _stableHandTiles.Clear();
            _stableHandLabels.Clear();
            _stableRecommendedDiscard = -1;
            _stableHandCount = -1;
            _lastFlowEvent = "stable_hand_self_discard_detected";
            return discarded;
        }

        private void CompleteSelfDiscard(int tileId)
        {
            if (tileId < 0)
                return;
            _discards[0].Add(tileId);
            _lastDiscard = tileId;
            _lastDiscarder = 0;
            _expectedDiscarder = 1;
            _currentPlayer = 1;
            _phase = PhaseDiscard;
            _selfDiscardTurnActive = false;
            _pendingSelfTableTile = tileId;
            _resetTableBaselineOnNextStable = false;
            _lastFlowEvent = "self_discard_left_hand:" + TileIdToLabel(tileId);
        }

        private void UpdateTableStability(List<TableObs> tableObs)
        {
            var sig = TableSignature(tableObs);
            if (sig == _tableCandidateSignature)
            {
                _tableCandidateFrames++;
                _tableCandidateObs = new List<TableObs>(tableObs);
            }
            else
            {
                _tableCandidateSignature = sig;
                _tableCandidateFrames = 1;
                _tableCandidateObs = new List<TableObs>(tableObs);
            }

            if (_tableCandidateFrames < _tableStableFrames)
                return;

            var cur = new List<TableObs>(_tableCandidateObs);
            if (_resetTableBaselineOnNextStable)
            {
                SetStableTable(sig, cur);
                _resetTableBaselineOnNextStable = false;
                _lastFlowEvent = "table_baseline_reset";
                return;
            }

            if (_hasStableTable && sig == _stableTableSignature)
                return;

            if (_pendingSelfTableTile >= 0)
            {
                if (!_hasStableTable)
                {
                    SetStableTable(sig, cur);
                    _lastFlowEvent = "table_self_discard_baseline:" + TileIdToLabel(_pendingSelfTableTile);
                    _pendingSelfTableTile = -1;
                    return;
                }

                if (cur.Count == _stableTableObs.Count + 1)
                {
                    var newTiles = FindNewTableObservations(_stableTableObs, cur);
                    if (newTiles.Count > 0)
                    {
                        SetStableTable(sig, cur);
                        _lastFlowEvent = "table_self_discard_baseline:" + TileIdToLabel(_pendingSelfTableTile);
                        _pendingSelfTableTile = -1;
                    }
                    else
                    {
                        _lastFlowEvent = "table_new_tile_unmatched";
                    }
                }
                else if (cur.Count == _stableTableObs.Count)
                {
                    _lastFlowEvent = "table_waiting_self_discard_baseline";
                }
                else if (cur.Count > _stableTableObs.Count + 1)
                {
                    _lastFlowEvent = "table_count_jump_ignored";
                }
                else
                {
                    _lastFlowEvent = "table_count_drop_ignored";
                }
                return;
            }

            if (!_hasStableTable)
            {
                SetStableTable(sig, cur);
                _lastFlowEvent = "table_baseline_stable";
                return;
            }

            if (cur.Count == _stableTableObs.Count + 1)
            {
                var newTiles = FindNewTableObservations(_stableTableObs, cur);
                if (newTiles.Count > 0)
                {
                    if (_expectedDiscarder == 0 && _selfDiscardTurnActive)
                    {
                        _lastFlowEvent = "table_waiting_self_discard_hand_exit";
                    }
                    else
                    {
                        AcceptNewDiscard(newTiles[0]);
                        SetStableTable(sig, cur);
                    }
                }
                else
                {
                    _lastFlowEvent = "table_new_tile_unmatched";
                }
            }
            else if (cur.Count == _stableTableObs.Count)
            {
                _lastFlowEvent = "table_same_count_stable";
            }
            else if (cur.Count > _stableTableObs.Count + 1)
            {
                _lastFlowEvent = "table_count_jump_ignored";
            }
            else
            {
                _lastFlowEvent = "table_count_drop_ignored";
            }
        }

        private void SetStableTable(string signature, List<TableObs> obs)
        {
            _stableTableSignature = signature;
            _stableTableObs = new List<TableObs>(obs);
            _hasStableTable = true;
        }

        private void AcceptNewDiscard(TableObs obs)
        {
            var discarder = _expectedDiscarder;
            _discards[discarder].Add(obs.TileId);
            _lastDiscard = obs.TileId;
            _lastDiscarder = discarder;
            _expectedDiscarder = (discarder + 1) % 4;
            _currentPlayer = _expectedDiscarder;
            if (discarder == 0)
            {
                _phase = PhaseDiscard;
                _selfDiscardTurnActive = false;
            }
            else
            {
                _phase = PhaseClaim;
            }
            _lastFlowEvent = "table_discard:" + TurnLabel(discarder) + ":" + TileIdToLabel(obs.TileId);
        }

        private void CorrectToSelfTurn(List<int> handTiles, List<string> handLabels)
        {
            _currentPlayer = 0;
            _phase = PhaseDiscard;
            _expectedDiscarder = 0;
            _selfDiscardTurnActive = true;
            _resetTableBaselineOnNextStable = true;
            _cooldownFramesRemaining = TurnCorrectionCooldownFrames;
            _stableHandActive = true;
            _stableHandTiles = new List<int>(handTiles);
            _stableHandLabels = new List<string>(handLabels);
            _stableHandCount = handTiles.Count;
            _lastFlowEvent = "stable_hand_promoted_to_self_turn";
        }

        private bool IsStableSelfTurn(List<int> handTiles)
        {
            if (!IsSelfTurnCount(handTiles.Count))
                return false;
            var sig = Signature(handTiles);
            return _stableHandSignature == sig;
        }

        private SentisInferenceUiManager.Advice BuildAdvice(List<int> handTiles, bool handStable)
        {
            var canDiscard = handStable && _phase == PhaseDiscard && _currentPlayer == 0 && IsSelfTurnCount(handTiles.Count);
            var canClaim = handStable && _phase == PhaseClaim && _lastDiscarder != 0 && _lastDiscard >= 0;
            if (!canDiscard && !canClaim)
                return EmptyAdvice();

            var mask = LegalActionMask(handTiles);
            var action = PredictAction(handTiles, mask);
            if (action < 0 || action >= ActionDim || !mask[action])
                action = FallbackAction(handTiles);

            var safe = FallbackAction(handTiles);
            return new SentisInferenceUiManager.Advice
            {
                benefit = new SentisInferenceUiManager.AdviceItem
                {
                    tile_id = action,
                    tile = ActionToLabel(action),
                    source = "quest-ppo",
                    reason = _phase == PhaseClaim ? "claim/pass action from local PPO" : "discard action from local PPO"
                },
                safe = new SentisInferenceUiManager.AdviceItem
                {
                    tile_id = safe,
                    tile = ActionToLabel(safe),
                    source = "fallback",
                    reason = "local legal fallback"
                }
            };
        }

        private int PredictAction(List<int> handTiles, bool[] mask)
        {
            using (var obs = new Tensor<float>(new TensorShape(1, ObsDim), clearOnInit: true))
            {
                FillObservation(obs, handTiles);
                _ppoWorker.Schedule(obs);

                var output = _ppoWorker.PeekOutput().ReadbackAndClone() as Tensor<float>;
                if (output == null)
                    return FallbackAction(handTiles);

                try
                {
                    var logits = output.DownloadToArray();
                    var best = -1;
                    var bestScore = float.NegativeInfinity;
                    for (var i = 0; i < Mathf.Min(ActionDim, logits.Length); i++)
                    {
                        if (!mask[i])
                            continue;
                        if (logits[i] > bestScore)
                        {
                            bestScore = logits[i];
                            best = i;
                        }
                    }
                    return best;
                }
                finally
                {
                    output.Dispose();
                }
            }
        }

        private void FillObservation(Tensor<float> obs, List<int> handTiles)
        {
            var handCounts = Count34(handTiles);
            for (var i = 0; i < 34; i++)
                obs[0, i] = handCounts[i];

            var offset = 34;
            for (var player = 0; player < 4; player++)
            {
                var counts = Count34(_discards[player]);
                for (var i = 0; i < 34; i++)
                    obs[0, offset + i] = counts[i];
                offset += 34;
            }

            obs[0, 170] = _lastDiscard < 0 ? 0f : (_lastDiscard + 1f) / 34f;
            obs[0, 171] = _lastDiscarder < 0 ? 0f : (_lastDiscarder + 1f) / 4f;
            obs[0, 172] = _phase;
            obs[0, 173] = 0f;
            obs[0, 174] = 0f;
        }

        private bool[] LegalActionMask(List<int> handTiles)
        {
            var mask = new bool[ActionDim];
            if (_phase == PhaseClaim && _lastDiscarder != 0 && _lastDiscard >= 0)
            {
                mask[ActPass] = true;
                if (CountTile(handTiles, _lastDiscard) >= 2)
                    mask[ActPon] = true;
                foreach (var action in ChiActions(handTiles, _lastDiscard, _lastDiscarder))
                    mask[action] = true;
                return mask;
            }

            if (_phase == PhaseDiscard && _currentPlayer == 0)
            {
                foreach (var tile in handTiles)
                {
                    if (tile >= 0 && tile < 34)
                        mask[tile] = true;
                }
                if (!mask.Any(x => x))
                    mask[ActPass] = true;
                return mask;
            }

            mask[ActPass] = true;
            return mask;
        }

        private IEnumerable<int> ChiActions(List<int> handTiles, int tile, int discarder)
        {
            if (discarder != 3 || tile >= 27)
                yield break;
            var suit = tile / 9;
            var idx = tile % 9;
            var baseId = suit * 9;
            if (idx >= 2 && CountTile(handTiles, baseId + idx - 2) > 0 && CountTile(handTiles, baseId + idx - 1) > 0)
                yield return ActChiLow;
            if (idx >= 1 && idx <= 7 && CountTile(handTiles, baseId + idx - 1) > 0 && CountTile(handTiles, baseId + idx + 1) > 0)
                yield return ActChiMid;
            if (idx <= 6 && CountTile(handTiles, baseId + idx + 1) > 0 && CountTile(handTiles, baseId + idx + 2) > 0)
                yield return ActChiHigh;
        }

        private int FallbackAction(List<int> handTiles)
        {
            if (_phase == PhaseClaim)
                return ActPass;
            return handTiles.Count > 0 ? handTiles[0] : ActPass;
        }

        private string BuildLog(
            List<string> hand,
            List<string> table,
            string action,
            string discard,
            bool stable,
            int tracked,
            int tableCount,
            int detectionCount,
            int classifiedCount,
            LatencyInfo latency)
        {
            var sb = new StringBuilder(512);
            sb.AppendLine("Mode: Quest local");
            sb.Append("Hand: ").AppendLine(hand.Count > 0 ? string.Join(" ", hand) : "detecting");
            sb.Append("Table: ").AppendLine(table.Count > 0 ? string.Join(" ", table) : "detecting");
            if (!string.IsNullOrEmpty(discard))
                sb.Append("Discard: ").AppendLine(discard);
            sb.Append("Turn: ").AppendLine(TurnLabel(_currentPlayer));
            sb.Append("Stable: ").AppendLine(stable ? "yes" : "no");
            sb.Append("Action: ").AppendLine(action);
            sb.Append("Flow: ").Append(string.IsNullOrEmpty(_lastFlowEvent) ? "none" : _lastFlowEvent)
                .Append(" exp=").Append(TurnLabel(_expectedDiscarder))
                .Append(" cd=").AppendLine(_cooldownFramesRemaining.ToString());
            sb.Append("Debug: det=").Append(detectionCount)
                .Append(" cls=").Append(classifiedCount)
                .Append(" tracked=").Append(tracked)
                .Append(" table=").Append(tableCount)
                .AppendLine();
            sb.Append("Latency: frame=").Append(latency.FrameMs.ToString("0.0"))
                .Append(" seg=").Append(latency.SegMs.ToString("0.0"))
                .Append(" cls=").Append(latency.ClsMs.ToString("0.0"))
                .Append(" trk=").Append(latency.TrackMs.ToString("0.0"))
                .Append(" ppo=").Append(latency.PpoMs.ToString("0.0"))
                .Append(" total=").Append(latency.TotalMs.ToString("0.0"))
                .Append(" ms");
            return sb.ToString();
        }

        private void InitializeLatencyLog()
        {
            try
            {
                var folder = Path.Combine(Application.persistentDataPath, LatencyLogFolderName);
                Directory.CreateDirectory(folder);
                var stamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
                _latencyLogPath = Path.Combine(folder, "quest_latency_" + stamp + ".csv");
                _latencyLogBuffer.Length = 0;
                _latencyRowsBuffered = 0;
                _latencySamples = 0;
                _latencyLogErrorReported = false;
                _latencyLogEnabled = true;

                File.WriteAllText(
                    _latencyLogPath,
                    "timestamp_utc,frame_index,unity_time_s,detected_count,classified_count,tracked_count,hand_count,table_live_count,table_stable_count,stable_hand,stable_table,stable_any,phase,current_turn,expected_discarder,last_discarder,last_discard,action,flow_event,frame_ms,segmentation_ms,classification_ms,tracking_ms,ppo_ms,total_pipeline_ms\n");
                Debug.Log("[QuestLocal] Latency log: " + _latencyLogPath);
            }
            catch (Exception e)
            {
                _latencyLogEnabled = false;
                _latencyLogPath = "";
                Debug.LogWarning("[QuestLocal] Latency log disabled: " + e.Message);
            }
        }

        private void RecordLatencySample(
            LatencyInfo latency,
            int detectionCount,
            int classifiedCount,
            int trackedCount,
            int handCount,
            int tableLiveCount,
            int tableStableCount,
            bool stableAny,
            string action)
        {
            if (!_latencyLogEnabled || string.IsNullOrEmpty(_latencyLogPath))
                return;

            try
            {
                _latencySamples++;
                _latencyLogBuffer
                    .Append(Csv(DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture))).Append(',')
                    .Append(_frameIndex).Append(',')
                    .Append(Time.realtimeSinceStartup.ToString("0.000", CultureInfo.InvariantCulture)).Append(',')
                    .Append(detectionCount).Append(',')
                    .Append(classifiedCount).Append(',')
                    .Append(trackedCount).Append(',')
                    .Append(handCount).Append(',')
                    .Append(tableLiveCount).Append(',')
                    .Append(tableStableCount).Append(',')
                    .Append(_stableHandActive ? "1" : "0").Append(',')
                    .Append(_hasStableTable ? "1" : "0").Append(',')
                    .Append(stableAny ? "1" : "0").Append(',')
                    .Append(_phase == PhaseClaim ? "claim" : "discard").Append(',')
                    .Append(Csv(TurnLabel(_currentPlayer))).Append(',')
                    .Append(Csv(TurnLabel(_expectedDiscarder))).Append(',')
                    .Append(_lastDiscarder).Append(',')
                    .Append(_lastDiscard).Append(',')
                    .Append(Csv(action)).Append(',')
                    .Append(Csv(_lastFlowEvent)).Append(',')
                    .Append(latency.FrameMs.ToString("0.000", CultureInfo.InvariantCulture)).Append(',')
                    .Append(latency.SegMs.ToString("0.000", CultureInfo.InvariantCulture)).Append(',')
                    .Append(latency.ClsMs.ToString("0.000", CultureInfo.InvariantCulture)).Append(',')
                    .Append(latency.TrackMs.ToString("0.000", CultureInfo.InvariantCulture)).Append(',')
                    .Append(latency.PpoMs.ToString("0.000", CultureInfo.InvariantCulture)).Append(',')
                    .Append(latency.TotalMs.ToString("0.000", CultureInfo.InvariantCulture))
                    .AppendLine();
                _latencyRowsBuffered++;

                if (_latencyRowsBuffered >= LatencyLogFlushRows)
                    FlushLatencyLog();
            }
            catch (Exception e)
            {
                DisableLatencyLogAfterError(e);
            }
        }

        private void FlushLatencyLog()
        {
            if (!_latencyLogEnabled || _latencyRowsBuffered <= 0 || string.IsNullOrEmpty(_latencyLogPath))
                return;

            try
            {
                File.AppendAllText(_latencyLogPath, _latencyLogBuffer.ToString());
                _latencyLogBuffer.Length = 0;
                _latencyRowsBuffered = 0;
            }
            catch (Exception e)
            {
                DisableLatencyLogAfterError(e);
            }
        }

        private void DisableLatencyLogAfterError(Exception e)
        {
            _latencyLogEnabled = false;
            _latencyLogBuffer.Length = 0;
            _latencyRowsBuffered = 0;
            if (!_latencyLogErrorReported)
            {
                _latencyLogErrorReported = true;
                Debug.LogWarning("[QuestLocal] Latency log write failed: " + e.Message);
            }
        }

        private static string Csv(string value)
        {
            if (string.IsNullOrEmpty(value))
                return "";
            if (value.IndexOfAny(CsvSpecialChars) < 0)
                return value;
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private List<string> StableTableLabels()
        {
            var ids = _stableTableObs.Select(x => x.TileId).ToList();
            ids.Sort();
            return ids.Select(TileIdToLabel).ToList();
        }

        private string LastDiscardText()
        {
            if (_lastDiscard < 0 || _lastDiscarder < 0)
                return "";
            return TurnLabel(_lastDiscarder) + " " + TileIdToLabel(_lastDiscard);
        }

        private static SentisInferenceUiManager.Advice EmptyAdvice()
        {
            return new SentisInferenceUiManager.Advice
            {
                benefit = new SentisInferenceUiManager.AdviceItem { tile_id = -1, tile = "", source = "", reason = "" },
                safe = new SentisInferenceUiManager.AdviceItem { tile_id = -1, tile = "", source = "", reason = "" }
            };
        }

        private void EnsureFrameTexture(Texture source)
        {
            var w = Mathf.Max(1, source.width);
            var h = Mathf.Max(1, source.height);
            if (_frameTexture == null || _frameTexture.width != w || _frameTexture.height != h)
            {
                if (_frameTexture != null)
                    UnityEngine.Object.Destroy(_frameTexture);
                _frameTexture = new Texture2D(w, h, TextureFormat.RGB24, false);
            }

            var webCam = source as WebCamTexture;
            if (webCam != null)
            {
                _frameTexture.SetPixels32(webCam.GetPixels32());
                _frameTexture.Apply(false);
                return;
            }

            var prev = RenderTexture.active;
            var rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(source, rt);
            RenderTexture.active = rt;
            _frameTexture.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            _frameTexture.Apply(false);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
        }

        private static float ReadYolo(float[] data, int channels, int count, int channel, int index)
        {
            if (channel >= channels || index >= count)
                return 0f;
            return data[channel * count + index];
        }

        private List<Detection> Nms(List<Detection> detections, int maxCount)
        {
            var selected = new List<Detection>();
            foreach (var det in detections)
            {
                var keep = true;
                foreach (var prev in selected)
                {
                    if (det.Area == prev.Area && Iou(det, prev) > _detIouThreshold)
                    {
                        keep = false;
                        break;
                    }
                }
                if (!keep)
                    continue;
                selected.Add(det);
                if (selected.Count >= maxCount)
                    break;
            }
            return selected;
        }

        private static float Iou(Detection a, Detection b)
        {
            var ax1 = a.Cx - a.W * 0.5f;
            var ay1 = a.Cy - a.H * 0.5f;
            var ax2 = a.Cx + a.W * 0.5f;
            var ay2 = a.Cy + a.H * 0.5f;
            var bx1 = b.Cx - b.W * 0.5f;
            var by1 = b.Cy - b.H * 0.5f;
            var bx2 = b.Cx + b.W * 0.5f;
            var by2 = b.Cy + b.H * 0.5f;
            var ix1 = Mathf.Max(ax1, bx1);
            var iy1 = Mathf.Max(ay1, by1);
            var ix2 = Mathf.Min(ax2, bx2);
            var iy2 = Mathf.Min(ay2, by2);
            var iw = Mathf.Max(0f, ix2 - ix1);
            var ih = Mathf.Max(0f, iy2 - iy1);
            var inter = iw * ih;
            var union = a.W * a.H + b.W * b.H - inter + 1e-6f;
            return inter / union;
        }

        private static float Iou(TrackState a, ClassifiedDetection b)
        {
            var det = new Detection { Cx = b.Cx, Cy = b.Cy, W = b.W, H = b.H, Area = b.Area };
            var prev = new Detection { Cx = a.Cx, Cy = a.Cy, W = a.W, H = a.H, Area = a.Area };
            return Iou(prev, det);
        }

        private static int ArgMax(float[] data)
        {
            if (data == null || data.Length == 0)
                return -1;
            var best = 0;
            var bestVal = data[0];
            for (var i = 1; i < data.Length; i++)
            {
                if (data[i] > bestVal)
                {
                    bestVal = data[i];
                    best = i;
                }
            }
            return best;
        }

        private static int TileLabelToId(string label)
        {
            if (string.IsNullOrEmpty(label) || label == "flower")
                return -1;
            for (var i = 0; i < TileLabels.Length; i++)
            {
                if (string.Equals(TileLabels[i], label, StringComparison.OrdinalIgnoreCase))
                    return i;
            }
            return -1;
        }

        private static string TileIdToLabel(int tileId)
        {
            return tileId >= 0 && tileId < TileLabels.Length ? TileLabels[tileId] : "";
        }

        private static string ActionToLabel(int action)
        {
            if (action >= 0 && action < 34)
                return TileIdToLabel(action);
            if (action == ActPass) return "PASS";
            if (action == ActPon) return "PON";
            if (action == ActChiLow) return "CHI_LOW";
            if (action == ActChiMid) return "CHI_MID";
            if (action == ActChiHigh) return "CHI_HIGH";
            return "";
        }

        private static string TurnLabel(int player)
        {
            switch ((player % 4 + 4) % 4)
            {
                case 0: return "self";
                case 1: return "next";
                case 2: return "opposite";
                case 3: return "previous";
                default: return "unknown";
            }
        }

        private static bool IsSelfTurnCount(int count)
        {
            return count > 0 && count % 3 == 2;
        }

        private static string Signature(List<int> tiles)
        {
            return string.Join(",", tiles);
        }

        private static string TableSignature(List<TableObs> obs)
        {
            var counts = new int[34];
            foreach (var item in obs)
            {
                if (item.TileId >= 0 && item.TileId < 34)
                    counts[item.TileId]++;
            }
            var parts = new List<string>();
            for (var i = 0; i < counts.Length; i++)
            {
                if (counts[i] > 0)
                    parts.Add(i + ":" + counts[i]);
            }
            return string.Join(",", parts);
        }

        private static int CountTile(List<int> tiles, int tile)
        {
            var count = 0;
            foreach (var t in tiles)
            {
                if (t == tile)
                    count++;
            }
            return count;
        }

        private static float[] Count34(List<int> tiles)
        {
            var counts = new float[34];
            foreach (var tile in tiles)
            {
                if (tile >= 0 && tile < 34)
                    counts[tile] += 1f;
            }
            return counts;
        }

        private static List<TableObs> FindNewTableObservations(List<TableObs> previous, List<TableObs> current)
        {
            var used = new bool[current.Count];

            foreach (var prev in previous)
            {
                if (prev.TrackId < 0)
                    continue;
                var matched = -1;
                for (var i = 0; i < current.Count; i++)
                {
                    if (used[i] || current[i].TrackId != prev.TrackId)
                        continue;
                    if (ObsDistance(prev, current[i]) <= TableMatchDist)
                    {
                        matched = i;
                        break;
                    }
                }
                if (matched >= 0)
                    used[matched] = true;
            }

            foreach (var prev in previous)
            {
                var best = -1;
                var bestDist = TableMatchDist;
                for (var i = 0; i < current.Count; i++)
                {
                    if (used[i] || current[i].TileId != prev.TileId)
                        continue;
                    var dist = ObsDistance(prev, current[i]);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        best = i;
                    }
                }
                if (best >= 0)
                    used[best] = true;
            }

            var newItems = new List<TableObs>();
            for (var i = 0; i < current.Count; i++)
            {
                if (!used[i])
                    newItems.Add(current[i]);
            }
            newItems.Sort((a, b) =>
            {
                var c = a.Cx.CompareTo(b.Cx);
                if (c != 0) return c;
                c = a.Cy.CompareTo(b.Cy);
                if (c != 0) return c;
                return a.TileId.CompareTo(b.TileId);
            });
            return newItems;
        }

        private static float ObsDistance(TableObs a, TableObs b)
        {
            var dx = a.Cx - b.Cx;
            var dy = a.Cy - b.Cy;
            return Mathf.Sqrt(dx * dx + dy * dy);
        }

        public sealed class LocalResult
        {
            public SentisInferenceUiManager.RemoteTile[] Tiles = Array.Empty<SentisInferenceUiManager.RemoteTile>();
            public string[] Hand = Array.Empty<string>();
            public SentisInferenceUiManager.Advice Advice = EmptyAdvice();
            public string PcLog = "";
        }

        private struct LatencyInfo
        {
            public float FrameMs;
            public float SegMs;
            public float ClsMs;
            public float TrackMs;
            public float PpoMs;
            public float TotalMs;
        }

        private struct Detection
        {
            public float Cx;
            public float Cy;
            public float W;
            public float H;
            public float Score;
            public string Area;
        }

        private struct ClassifiedDetection
        {
            public float Cx;
            public float Cy;
            public float W;
            public float H;
            public string Area;
            public string Label;
            public float Confidence;
        }

        private struct TableObs
        {
            public int TrackId;
            public int TileId;
            public string Label;
            public float Cx;
            public float Cy;
        }

        private sealed class TrackState
        {
            public readonly int Id;
            public float Cx;
            public float Cy;
            public float W;
            public float H;
            public string Area;
            public string Label;
            public float Confidence;
            public float LastSeen;

            public TrackState(int id, ClassifiedDetection det, float now)
            {
                Id = id;
                Update(det, now);
            }

            public void Update(ClassifiedDetection det, float now)
            {
                Cx = det.Cx;
                Cy = det.Cy;
                W = det.W;
                H = det.H;
                Area = det.Area;
                Label = det.Label;
                Confidence = det.Confidence;
                LastSeen = now;
            }
        }
    }
}
