from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import re
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse


DISTANCE_OPTIONS = ["遠", "中", "近"]
ANGLE_OPTIONS = ["正常", "偏上", "偏下", "偏左", "偏右"]
LIGHTING_OPTIONS = ["正常", "亮", "不足"]
STATUS_OPTIONS = [
    {"value": "valid", "label": "有效"},
    {"value": "ignore", "label": "忽略"},
    {"value": "false_positive", "label": "誤檢"},
    {"value": "missed", "label": "漏檢"},
]

SAMPLE_ID_RE = re.compile(r"^[0-9A-Za-z_.-]+$")


INDEX_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Segmentation 標註工具</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #17202a;
      --muted: #637083;
      --brand: #2563eb;
      --brand-dark: #174bb7;
      --ok: #0f8f61;
      --bad: #c2410c;
      --warn: #946200;
      --focus: #111827;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Microsoft JhengHei", Arial, sans-serif;
      letter-spacing: 0;
    }

    button, select, input, textarea {
      font: inherit;
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      min-height: 36px;
      padding: 0 12px;
      border-radius: 6px;
      cursor: pointer;
    }

    button:hover { border-color: #98a7bc; }
    button.primary {
      background: var(--brand);
      border-color: var(--brand);
      color: #fff;
    }
    button.primary:hover { background: var(--brand-dark); }
    button.ghost { background: transparent; }
    button.danger { color: var(--bad); }
    button:disabled { cursor: not-allowed; opacity: 0.55; }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .topbar {
      min-height: 64px;
      padding: 10px 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 12px;
      align-items: center;
    }

    .title {
      display: flex;
      gap: 12px;
      align-items: baseline;
      min-width: 0;
    }
    .title h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      white-space: nowrap;
    }
    .sample-name {
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }

    .main {
      display: grid;
      grid-template-columns: minmax(420px, 1.2fr) minmax(430px, 0.8fr);
      min-height: 0;
    }

    .viewer {
      min-height: 0;
      border-right: 1px solid var(--line);
      display: grid;
      grid-template-rows: 1fr auto;
      background: #eef1f5;
    }

    .image-stage {
      min-height: 320px;
      position: relative;
      display: grid;
      place-items: center;
      overflow: hidden;
      padding: 18px;
    }

    .image-wrap {
      position: relative;
      max-width: 100%;
      max-height: 100%;
      display: inline-block;
      box-shadow: 0 8px 28px rgba(15, 23, 42, 0.16);
      background: #111827;
    }

    #sampleImage {
      display: block;
      max-width: min(100%, calc(100vw - 520px));
      max-height: calc(100vh - 156px);
      width: auto;
      height: auto;
    }

    #overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    .stats-strip {
      min-height: 64px;
      border-top: 1px solid var(--line);
      background: #fff;
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 1px;
    }

    .metric {
      padding: 10px 14px;
      border-right: 1px solid var(--line);
    }
    .metric:last-child { border-right: 0; }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }
    .metric-value {
      font-size: 18px;
      font-weight: 700;
    }

    .side {
      min-height: 0;
      overflow: auto;
      background: var(--panel);
      display: grid;
      grid-template-rows: auto auto auto 1fr auto;
    }

    .conditions {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 36px;
      padding: 6px 9px;
      color: var(--text);
      background: #fff;
      width: 100%;
    }
    textarea {
      min-height: 68px;
      resize: vertical;
    }

    .summary {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f9fafb;
      color: var(--text);
      font-weight: 600;
    }
    .pill.ok { color: var(--ok); }
    .pill.bad { color: var(--bad); }

    .breakdown {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
      font-size: 13px;
    }
    .breakdown-group {
      display: grid;
      grid-template-columns: 58px 1fr;
      gap: 8px;
      align-items: start;
    }
    .breakdown-title {
      color: var(--muted);
      font-weight: 700;
      padding-top: 4px;
    }
    .breakdown-items {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .breakdown .pill {
      min-height: 24px;
      font-size: 12px;
      font-weight: 600;
    }

    .objects {
      overflow: auto;
    }
    .label-review {
      display: grid;
      gap: 14px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .label-review label {
      color: var(--muted);
      font-size: 13px;
    }
    .label-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-height: 28px;
      align-items: center;
    }
    .label-chip {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 7px;
      background: #f8fafc;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
    }
    .label-review textarea {
      min-height: 86px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      line-height: 1.5;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    th, td {
      padding: 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
    }
    th {
      position: sticky;
      top: 0;
      background: #f9fafb;
      z-index: 1;
      text-align: left;
      color: var(--muted);
      font-size: 12px;
    }
    th:nth-child(1), td:nth-child(1) { width: 46px; text-align: center; }
    th:nth-child(2), td:nth-child(2) { width: 112px; }
    th:nth-child(3), td:nth-child(3) { width: 74px; }
    th:nth-child(5), td:nth-child(5) { width: 124px; }
    th:nth-child(6), td:nth-child(6) { width: 64px; text-align: center; }
    tr.active { background: #eff6ff; }
    tr.correct { box-shadow: inset 4px 0 0 var(--ok); }
    tr.wrong { box-shadow: inset 4px 0 0 var(--bad); }
    .pred {
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .conf {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .row-actions button {
      min-width: 34px;
      padding: 0;
    }

    .footer {
      border-top: 1px solid var(--line);
      padding: 12px 16px;
      display: grid;
      gap: 12px;
      background: #fff;
    }
    .footer-actions {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
    .left-actions, .right-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .empty {
      color: var(--muted);
      text-align: center;
      padding: 28px;
      line-height: 1.7;
    }

    @media (max-width: 980px) {
      .topbar { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
      .main { grid-template-columns: 1fr; }
      .viewer { border-right: 0; border-bottom: 1px solid var(--line); }
      #sampleImage {
        max-width: calc(100vw - 36px);
        max-height: 52vh;
      }
      .conditions { grid-template-columns: 1fr; }
      .stats-strip { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="title">
        <h1>Segmentation 標註工具</h1>
        <div class="sample-name" id="sampleName">尚未載入</div>
      </div>
      <div class="toolbar">
        <button id="prevBtn">上一張</button>
        <button id="nextBtn">下一張</button>
        <button id="saveBtn" class="primary">儲存</button>
        <button id="saveNextBtn" class="primary">儲存並下一張</button>
        <button id="reloadBtn" class="ghost">重新載入</button>
        <button id="exportBtn">匯出 CSV</button>
      </div>
    </header>

    <main class="main">
      <section class="viewer">
        <div class="image-stage">
          <div class="image-wrap" id="imageWrap">
            <img id="sampleImage" alt="">
            <canvas id="overlay"></canvas>
          </div>
          <div class="empty" id="emptyState" hidden>目前沒有可標註的樣本。</div>
        </div>
        <div class="stats-strip">
          <div class="metric"><div class="metric-label">整體正確率</div><div class="metric-value" id="statOverall">-</div></div>
          <div class="metric"><div class="metric-label">比對結果</div><div class="metric-value" id="statObjects">-</div></div>
          <div class="metric"><div class="metric-label">已標註圖片</div><div class="metric-value" id="statImages">-</div></div>
          <div class="metric"><div class="metric-label">目前圖片</div><div class="metric-value" id="statCurrent">-</div></div>
        </div>
      </section>

      <section class="side">
        <div class="conditions">
          <label>距離
            <select id="distanceSelect"></select>
          </label>
          <label>角度
            <select id="angleSelect"></select>
          </label>
          <label>光線
            <select id="lightingSelect"></select>
          </label>
        </div>

        <div class="summary" id="sampleSummary">
          <span class="pill">0 / 0</span>
        </div>

        <div class="breakdown" id="breakdown"></div>

        <div class="label-review">
          <label>模型預測
            <div class="label-list" id="predictedLabels"></div>
          </label>
          <label>正確牌清單
            <textarea id="trueLabelsInput" placeholder="例如：1m 1m 3p east white"></textarea>
          </label>
        </div>

        <div class="footer">
          <label>備註
            <textarea id="notesInput"></textarea>
          </label>
          <div class="footer-actions">
            <div class="left-actions">
              <button id="usePredictedBtn">套用模型牌</button>
            </div>
            <div class="right-actions">
              <button id="saveBottomBtn" class="primary">儲存</button>
              <button id="saveNextBottomBtn" class="primary">儲存並下一張</button>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <datalist id="labelOptions"></datalist>

  <script>
    const state = {
      config: null,
      samples: [],
      index: 0,
      sample: null,
      annotation: null,
      active: 0,
      stats: null,
    };

    const $ = (id) => document.getElementById(id);

    async function api(path, options) {
      const res = await fetch(path, options);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      const contentType = res.headers.get("content-type") || "";
      return contentType.includes("application/json") ? res.json() : res.text();
    }

    function pctValue(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${(Number(value) * 100).toFixed(1)}%`;
    }

    function pct(correct, total) {
      if (!total) return "-";
      return pctValue(correct / total);
    }

    function optionHtml(values, selected) {
      return `<option value=""></option>` + values.map((value) => {
        const s = value === selected ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${s}>${escapeHtml(value)}</option>`;
      }).join("");
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function labelsEqual(a, b) {
      return String(a || "").trim().toLowerCase() === String(b || "").trim().toLowerCase();
    }

    function parseLabels(text) {
      return String(text || "")
        .split(/[\s,，、]+/)
        .map((label) => label.trim())
        .filter(Boolean);
    }

    function labelsToText(labels) {
      return (labels || []).filter(Boolean).join(" ");
    }

    function predictionLabels(prediction) {
      return (prediction?.tiles || [])
        .map((tile) => tile.predicted_label || tile.cls || "")
        .filter(Boolean);
    }

    function countLabels(labels) {
      const counts = new Map();
      for (const label of labels || []) {
        const key = String(label || "").trim().toLowerCase();
        if (!key) continue;
        counts.set(key, (counts.get(key) || 0) + 1);
      }
      return counts;
    }

    function labelMetrics(predictedLabels, trueLabels) {
      const predCounts = countLabels(predictedLabels);
      const trueCounts = countLabels(trueLabels);
      const keys = new Set([...predCounts.keys(), ...trueCounts.keys()]);
      let matched = 0;
      for (const key of keys) {
        matched += Math.min(predCounts.get(key) || 0, trueCounts.get(key) || 0);
      }
      const predicted = (predictedLabels || []).length;
      const expected = (trueLabels || []).length;
      const total = Math.max(predicted, expected);
      return {
        correct: matched,
        total,
        wrong: Math.max(0, total - matched),
        accuracy: total ? matched / total : null,
        predicted,
        expected,
      };
    }

    function labelsFromObjects(objects) {
      const labels = [];
      for (const obj of objects || []) {
        const status = obj.status || "valid";
        if (status === "ignore" || status === "false_positive") continue;
        const label = obj.true_label || obj.predicted_label || "";
        if (label) labels.push(label);
      }
      return labels;
    }

    async function init() {
      state.config = await api("/api/config");
      fillConfig();
      await reloadSamples();
      bindEvents();
    }

    function fillConfig() {
      $("labelOptions").innerHTML = state.config.labels
        .map((label) => `<option value="${escapeHtml(label)}"></option>`)
        .join("");
    }

    async function reloadSamples() {
      state.samples = (await api("/api/samples")).samples || [];
      await refreshStats();
      if (!state.samples.length) {
        state.sample = null;
        state.annotation = null;
        renderEmpty();
        return;
      }
      if (state.index >= state.samples.length) state.index = state.samples.length - 1;
      await loadSample(state.index);
    }

    async function refreshStats() {
      state.stats = await api("/api/stats");
      const overall = state.stats.overall || { correct: 0, total: 0 };
      $("statOverall").textContent = pct(overall.correct || 0, overall.total || 0);
      $("statObjects").textContent = `${overall.correct || 0} / ${overall.total || 0}`;
      $("statImages").textContent = `${state.stats.annotated_samples || 0} / ${state.stats.sample_count || 0}`;
      renderBreakdown();
    }

    function renderBreakdown() {
      if (!$("breakdown") || !state.stats) return;
      const groups = [
        ["距離", state.stats.by_distance || {}],
        ["角度", state.stats.by_angle || {}],
        ["光線", state.stats.by_lighting || {}],
      ];
      $("breakdown").innerHTML = groups.map(([title, data]) => {
        const entries = Object.entries(data);
        const items = entries.length
          ? entries.map(([key, value]) => {
              const groupCls = value.total && value.correct === value.total ? "ok" : value.total ? "bad" : "";
              return `<span class="pill ${groupCls}">${escapeHtml(key)} ${pct(value.correct || 0, value.total || 0)} (${value.correct || 0}/${value.total || 0})</span>`;
            }).join("")
          : `<span class="pill">-</span>`;
        return `<div class="breakdown-group"><div class="breakdown-title">${title}</div><div class="breakdown-items">${items}</div></div>`;
      }).join("");
    }

    async function loadSample(index) {
      if (!state.samples.length) return;
      state.index = Math.max(0, Math.min(index, state.samples.length - 1));
      const id = state.samples[state.index].sample_id;
      state.sample = await api(`/api/sample/${encodeURIComponent(id)}`);
      state.annotation = buildAnnotation(state.sample);
      state.active = 0;
      renderSample();
    }

    function buildAnnotation(data) {
      const prediction = data.prediction || {};
      const defaults = data.defaults || {};
      const saved = data.annotation;
      const tiles = prediction.tiles || [];
      const predictedLabels = predictionLabels(prediction);
      const savedTrueLabels = saved?.true_labels_text || labelsToText(saved?.true_labels || labelsFromObjects(saved?.objects || []));
      const defaultTrueLabels = defaults.true_labels_text || labelsToText(defaults.true_labels || labelsFromObjects(defaults.objects || []));
      const trueLabelsText = savedTrueLabels || defaultTrueLabels || labelsToText(predictedLabels);
      const objects = tiles.map((tile, index) => ({
        index,
        track_id: tile.track_id ?? index,
        predicted_label: tile.predicted_label || tile.cls || "",
        confidence: tile.confidence ?? tile.conf ?? 0,
        area: tile.area || "",
        bbox_xyxy: tile.bbox_xyxy || null,
      }));
      return {
        conditions: {
          distance: saved?.conditions?.distance || defaults.conditions?.distance || prediction.conditions?.distance || "",
          angle: saved?.conditions?.angle || defaults.conditions?.angle || prediction.conditions?.angle || "",
          lighting: saved?.conditions?.lighting || defaults.conditions?.lighting || prediction.conditions?.lighting || "",
        },
        predicted_labels: predictedLabels,
        true_labels_text: trueLabelsText,
        notes: saved?.notes || "",
        objects,
      };
    }

    function renderEmpty() {
      $("emptyState").hidden = false;
      $("imageWrap").hidden = true;
      $("sampleName").textContent = "尚未載入";
      $("predictedLabels").innerHTML = "";
      $("trueLabelsInput").value = "";
      $("sampleSummary").innerHTML = `<span class="pill">0 / 0</span>`;
      $("statCurrent").textContent = "-";
      setButtonsDisabled(true);
    }

    function setButtonsDisabled(disabled) {
      ["prevBtn", "nextBtn", "saveBtn", "saveNextBtn", "saveBottomBtn", "saveNextBottomBtn", "usePredictedBtn"].forEach((id) => {
        $(id).disabled = disabled;
      });
    }

    function renderSample() {
      setButtonsDisabled(false);
      $("emptyState").hidden = true;
      $("imageWrap").hidden = false;

      const prediction = state.sample.prediction || {};
      const imageUrl = state.sample.image_url;
      $("sampleName").textContent = `${state.index + 1} / ${state.samples.length} - ${prediction.sample_id}`;
      $("statCurrent").textContent = `${state.index + 1} / ${state.samples.length}`;

      const conditions = state.annotation.conditions || {};
      $("distanceSelect").innerHTML = optionHtml(state.config.distance_options, conditions.distance || "");
      $("angleSelect").innerHTML = optionHtml(state.config.angle_options, conditions.angle || "");
      $("lightingSelect").innerHTML = optionHtml(state.config.lighting_options, conditions.lighting || "");
      $("trueLabelsInput").value = state.annotation.true_labels_text || "";
      $("notesInput").value = state.annotation.notes || "";

      const img = $("sampleImage");
      img.onload = drawOverlay;
      img.src = imageUrl;

      renderPredictedLabels();
      updateSampleSummary();
      drawOverlay();
    }

    function renderPredictedLabels() {
      const labels = state.annotation.predicted_labels || [];
      $("predictedLabels").innerHTML = labels.length
        ? labels.map((label) => `<span class="label-chip">${escapeHtml(label)}</span>`).join("")
        : `<span class="pill">-</span>`;
    }

    function updateSampleSummary() {
      const result = labelMetrics(state.annotation.predicted_labels || [], parseLabels(state.annotation.true_labels_text || ""));
      const cls = result.total && result.correct === result.total ? "ok" : result.total ? "bad" : "";
      $("sampleSummary").innerHTML = `
        <span class="pill ${cls}">${result.correct} / ${result.total}</span>
        <span>正確率：${pct(result.correct, result.total)}</span>
        <span>模型：${result.predicted}</span>
        <span>正確：${result.expected}</span>
        <span>差異：${result.wrong}</span>
      `;
    }

    function drawOverlay() {
      const img = $("sampleImage");
      const canvas = $("overlay");
      if (!state.sample || !img.complete || !img.clientWidth || !img.clientHeight) return;
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const imageInfo = state.sample.prediction?.image || {};
      const iw = imageInfo.width || img.naturalWidth;
      const ih = imageInfo.height || img.naturalHeight;
      if (!iw || !ih) return;

      const rows = state.annotation.objects || [];
      rows.forEach((obj, i) => {
        if (!obj.bbox_xyxy || obj.bbox_xyxy.length !== 4) return;
        const [x1, y1, x2, y2] = obj.bbox_xyxy;
        const x = x1 / iw * canvas.width;
        const y = y1 / ih * canvas.height;
        const w = (x2 - x1) / iw * canvas.width;
        const h = (y2 - y1) / ih * canvas.height;
        const color = obj.area === "hand" ? "#65a30d" : obj.area === "table" ? "#0284c7" : "#2563eb";
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(x, y, Math.max(2, w), Math.max(2, h));
        ctx.fillStyle = color;
        ctx.font = "11px Segoe UI, Arial";
        const label = `${obj.index ?? i}`;
        const textW = ctx.measureText(label).width + 6;
        ctx.fillRect(x, Math.max(0, y - 16), textW, 14);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 3, Math.max(10, y - 5));
      });
    }

    function collectAnnotation() {
      state.annotation.conditions = {
        distance: $("distanceSelect").value,
        angle: $("angleSelect").value,
        lighting: $("lightingSelect").value,
      };
      state.annotation.true_labels_text = $("trueLabelsInput").value;
      state.annotation.notes = $("notesInput").value;
      return state.annotation;
    }

    async function saveAnnotation(goNext) {
      if (!state.sample) return;
      const id = state.sample.prediction.sample_id;
      const annotation = collectAnnotation();
      await api(`/api/sample/${encodeURIComponent(id)}/annotation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(annotation),
      });
      state.samples[state.index].has_annotation = true;
      await refreshStats();
      if (goNext && state.index < state.samples.length - 1) {
        await loadSample(state.index + 1);
      } else {
        await loadSample(state.index);
      }
    }

    function usePredictedLabels() {
      state.annotation.true_labels_text = labelsToText(state.annotation.predicted_labels || []);
      $("trueLabelsInput").value = state.annotation.true_labels_text;
      updateSampleSummary();
    }

    function bindEvents() {
      $("prevBtn").onclick = () => loadSample(state.index - 1);
      $("nextBtn").onclick = () => loadSample(state.index + 1);
      $("saveBtn").onclick = () => saveAnnotation(false);
      $("saveNextBtn").onclick = () => saveAnnotation(true);
      $("saveBottomBtn").onclick = () => saveAnnotation(false);
      $("saveNextBottomBtn").onclick = () => saveAnnotation(true);
      $("reloadBtn").onclick = reloadSamples;
      $("usePredictedBtn").onclick = usePredictedLabels;
      $("exportBtn").onclick = () => { window.location.href = "/api/export.csv"; };
      ["distanceSelect", "angleSelect", "lightingSelect", "trueLabelsInput", "notesInput"].forEach((id) => {
        $(id).addEventListener("input", () => {
          collectAnnotation();
          updateSampleSummary();
        });
      });
      window.addEventListener("resize", drawOverlay);
    }

    init().catch((err) => {
      console.error(err);
      $("sampleName").textContent = err.message;
      renderEmpty();
    });
  </script>
</body>
</html>
"""


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def labels_equal(a: Any, b: Any) -> bool:
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def parse_labels_text(text: Any) -> List[str]:
    raw = str(text or "")
    for sep in [",", "，", "、", "\n", "\t", "\r"]:
        raw = raw.replace(sep, " ")
    return [part.strip() for part in raw.split(" ") if part.strip()]


def labels_to_text(labels: List[str]) -> str:
    return " ".join([str(label).strip() for label in labels if str(label).strip()])


def prediction_labels(prediction: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for tile in prediction.get("tiles", []) or []:
        if not isinstance(tile, dict):
            continue
        label = str(tile.get("predicted_label", tile.get("cls", "")) or "").strip()
        if label:
            labels.append(label)
    return labels


def labels_from_objects(objects: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for obj in objects or []:
        if not isinstance(obj, dict):
            continue
        status = str(obj.get("status") or "valid")
        if status in ("ignore", "false_positive"):
            continue
        label = str(obj.get("true_label", obj.get("predicted_label", "")) or "").strip()
        if label:
            labels.append(label)
    return labels


def annotation_true_labels(annotation: Optional[Dict[str, Any]]) -> List[str]:
    if not annotation:
        return []
    if "true_labels_text" in annotation:
        return parse_labels_text(annotation.get("true_labels_text", ""))
    if isinstance(annotation.get("true_labels"), list):
        return [str(label).strip() for label in annotation.get("true_labels", []) if str(label).strip()]
    return labels_from_objects(annotation.get("objects", []) or [])


def label_counts(labels: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels or []:
        key = str(label or "").strip().lower()
        if not key:
            continue
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def classification_metrics_from_counts(correct: int = 0, total: int = 0) -> Dict[str, Any]:
    correct = int(correct or 0)
    total = int(total or 0)
    wrong = max(0, total - correct)
    accuracy = correct / total if total else None
    return {
        "correct": correct,
        "total": total,
        "wrong": wrong,
        "accuracy": accuracy,
    }


def build_label_metrics(predicted_labels: List[str], true_labels: List[str]) -> Dict[str, Any]:
    pred_counts = label_counts(predicted_labels)
    true_counts = label_counts(true_labels)
    matched = 0
    for key in set(pred_counts) | set(true_counts):
        matched += min(pred_counts.get(key, 0), true_counts.get(key, 0))
    total = max(len(predicted_labels or []), len(true_labels or []))
    metrics = classification_metrics_from_counts(matched, total)
    metrics["predicted"] = len(predicted_labels or [])
    metrics["expected"] = len(true_labels or [])
    return metrics


class AnnotationStore:
    def __init__(self, samples_dir: Path, labels_path: Path):
        self.samples_dir = samples_dir
        self.labels_path = labels_path
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def load_labels(self) -> List[str]:
        if not self.labels_path.exists():
            return []
        return [line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def sample_dirs(self) -> List[Path]:
        dirs = [
            path
            for path in self.samples_dir.iterdir()
            if path.is_dir() and (path / "prediction.json").exists()
        ]
        return sorted(dirs, key=lambda p: p.name)

    def sample_path(self, sample_id: str) -> Path:
        if not SAMPLE_ID_RE.match(sample_id):
            raise ValueError("invalid sample id")
        path = self.samples_dir / sample_id
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(sample_id)
        return path

    def load_prediction(self, sample_id: str) -> Dict[str, Any]:
        path = self.sample_path(sample_id) / "prediction.json"
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(path)
        return data

    def load_annotation(self, sample_id: str) -> Optional[Dict[str, Any]]:
        return read_json(self.sample_path(sample_id) / "annotation.json")

    def list_samples(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for path in self.sample_dirs():
            prediction = read_json(path / "prediction.json") or {}
            annotation = read_json(path / "annotation.json")
            items.append(
                {
                    "sample_id": path.name,
                    "captured_at_local": prediction.get("captured_at_local", ""),
                    "object_count": len(prediction.get("tiles", []) or []),
                    "has_annotation": annotation is not None,
                    "accuracy": self.sample_accuracy(annotation, prediction) if annotation else build_label_metrics([], []),
                }
            )
        return items

    def previous_annotation(self, sample_id: str) -> Dict[str, Any]:
        dirs = self.sample_dirs()
        names = [path.name for path in dirs]
        if sample_id not in names:
            return {}
        idx = names.index(sample_id)
        for prev in reversed(dirs[:idx]):
            annotation = read_json(prev / "annotation.json")
            if annotation:
                return {
                    "conditions": annotation.get("conditions", {}),
                    "true_labels_text": annotation.get("true_labels_text", labels_to_text(annotation_true_labels(annotation))),
                    "true_labels": annotation_true_labels(annotation),
                    "objects": annotation.get("objects", []),
                }
        return {}

    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        prediction = self.load_prediction(sample_id)
        annotation = self.load_annotation(sample_id)
        image_file = str(prediction.get("image_file") or "original.jpg")
        preview_file = str(prediction.get("preview_file") or "preview.jpg")
        return {
            "prediction": prediction,
            "annotation": annotation,
            "defaults": self.previous_annotation(sample_id),
            "image_url": f"/files/{sample_id}/{image_file}",
            "preview_url": f"/files/{sample_id}/{preview_file}",
        }

    def save_annotation(self, sample_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prediction = self.load_prediction(sample_id)
        predicted = prediction_labels(prediction)
        true_labels_text = str(payload.get("true_labels_text", "") or "").strip()
        true_labels = parse_labels_text(true_labels_text)
        conditions = payload.get("conditions", {}) if isinstance(payload.get("conditions"), dict) else {}
        metrics = build_label_metrics(predicted, true_labels)
        annotation = {
            "schema_version": 2,
            "sample_id": sample_id,
            "updated_at_epoch": time.time(),
            "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": {
                "distance": str(conditions.get("distance", "") or ""),
                "angle": str(conditions.get("angle", "") or ""),
                "lighting": str(conditions.get("lighting", "") or ""),
            },
            "predicted_labels": predicted,
            "true_labels": true_labels,
            "true_labels_text": labels_to_text(true_labels),
            "metrics": metrics,
            "notes": str(payload.get("notes", "") or ""),
            "accuracy": metrics,
        }
        write_json(self.sample_path(sample_id) / "annotation.json", annotation)
        return annotation

    @staticmethod
    def sample_accuracy(annotation: Optional[Dict[str, Any]], prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not annotation:
            return build_label_metrics([], [])
        predicted = annotation.get("predicted_labels")
        if not isinstance(predicted, list):
            predicted = prediction_labels(prediction or {})
        true_labels = annotation_true_labels(annotation)
        return build_label_metrics([str(label) for label in predicted], true_labels)

    @staticmethod
    def _add_group_counts(groups: Dict[str, Dict[str, int]], key: str, correct: int, total: int) -> None:
        key = key or "未分類"
        bucket = groups.setdefault(key, {"correct": 0, "total": 0})
        bucket["correct"] = int(bucket.get("correct", 0)) + int(correct or 0)
        bucket["total"] = int(bucket.get("total", 0)) + int(total or 0)

    def stats(self) -> Dict[str, Any]:
        overall = {"correct": 0, "total": 0}
        by_distance: Dict[str, Dict[str, int]] = {}
        by_angle: Dict[str, Dict[str, int]] = {}
        by_lighting: Dict[str, Dict[str, int]] = {}
        sample_count = 0
        annotated_samples = 0

        for path in self.sample_dirs():
            sample_count += 1
            annotation = read_json(path / "annotation.json")
            if not annotation:
                continue
            annotated_samples += 1
            prediction = read_json(path / "prediction.json") or {}
            conditions = annotation.get("conditions", {}) or {}
            metrics = self.sample_accuracy(annotation, prediction)
            overall["correct"] = int(overall.get("correct", 0)) + int(metrics.get("correct", 0))
            overall["total"] = int(overall.get("total", 0)) + int(metrics.get("total", 0))
            self._add_group_counts(
                by_distance,
                str(conditions.get("distance", "") or ""),
                int(metrics.get("correct", 0)),
                int(metrics.get("total", 0)),
            )
            self._add_group_counts(
                by_angle,
                str(conditions.get("angle", "") or ""),
                int(metrics.get("correct", 0)),
                int(metrics.get("total", 0)),
            )
            self._add_group_counts(
                by_lighting,
                str(conditions.get("lighting", "") or ""),
                int(metrics.get("correct", 0)),
                int(metrics.get("total", 0)),
            )

        return {
            "sample_count": sample_count,
            "annotated_samples": annotated_samples,
            "overall": classification_metrics_from_counts(
                overall.get("correct", 0),
                overall.get("total", 0),
            ),
            "by_distance": self._with_accuracy(by_distance),
            "by_angle": self._with_accuracy(by_angle),
            "by_lighting": self._with_accuracy(by_lighting),
        }

    @staticmethod
    def _with_accuracy(groups: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        return {
            key: classification_metrics_from_counts(
                value.get("correct", 0),
                value.get("total", 0),
            )
            for key, value in groups.items()
        }

    def export_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "sample_id",
                "distance",
                "angle",
                "lighting",
                "predicted_labels",
                "true_labels",
                "predicted_count",
                "true_count",
                "matched",
                "total",
                "accuracy",
                "notes",
            ]
        )
        for path in self.sample_dirs():
            annotation = read_json(path / "annotation.json")
            if not annotation:
                continue
            prediction = read_json(path / "prediction.json") or {}
            conditions = annotation.get("conditions", {}) or {}
            predicted = annotation.get("predicted_labels")
            if not isinstance(predicted, list):
                predicted = prediction_labels(prediction)
            true_labels = annotation_true_labels(annotation)
            metrics = build_label_metrics([str(label) for label in predicted], true_labels)
            writer.writerow(
                [
                    path.name,
                    conditions.get("distance", ""),
                    conditions.get("angle", ""),
                    conditions.get("lighting", ""),
                    labels_to_text([str(label) for label in predicted]),
                    labels_to_text(true_labels),
                    metrics.get("predicted", 0),
                    metrics.get("expected", 0),
                    metrics.get("correct", 0),
                    metrics.get("total", 0),
                    metrics.get("accuracy", ""),
                    annotation.get("notes", ""),
                ]
            )
        return buf.getvalue()


class AnnotationHandler(BaseHTTPRequestHandler):
    store: AnnotationStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[Annotator] {self.address_string()} - {fmt % args}")

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_error_text(self, status: HTTPStatus, message: str) -> None:
        self.send_text(message, "text/plain; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/config":
                self.send_json(
                    {
                        "labels": self.store.load_labels(),
                        "distance_options": DISTANCE_OPTIONS,
                        "angle_options": ANGLE_OPTIONS,
                        "lighting_options": LIGHTING_OPTIONS,
                        "status_options": STATUS_OPTIONS,
                    }
                )
            elif path == "/api/samples":
                self.send_json({"samples": self.store.list_samples()})
            elif path == "/api/stats":
                self.send_json(self.store.stats())
            elif path == "/api/export.csv":
                csv_text = self.store.export_csv()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="segmentation_annotations.csv"')
                payload = csv_text.encode("utf-8-sig")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif path.startswith("/api/sample/"):
                sample_id = unquote(path.split("/", 3)[3])
                self.send_json(self.store.get_sample(sample_id))
            elif path.startswith("/files/"):
                self.serve_file(path)
            else:
                self.send_error_text(HTTPStatus.NOT_FOUND, "not found")
        except FileNotFoundError:
            self.send_error_text(HTTPStatus.NOT_FOUND, "sample not found")
        except ValueError as exc:
            self.send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.send_error_text(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            match = re.match(r"^/api/sample/([^/]+)/annotation$", path)
            if not match:
                self.send_error_text(HTTPStatus.NOT_FOUND, "not found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            sample_id = unquote(match.group(1))
            annotation = self.store.save_annotation(sample_id, payload)
            self.send_json(annotation)
        except FileNotFoundError:
            self.send_error_text(HTTPStatus.NOT_FOUND, "sample not found")
        except ValueError as exc:
            self.send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.send_error_text(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def serve_file(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 4:
            raise FileNotFoundError(path)
        sample_id = unquote(parts[2])
        rel_name = unquote("/".join(parts[3:]))
        if "/" in rel_name or "\\" in rel_name or rel_name.startswith("."):
            raise ValueError("invalid file name")
        sample_dir = self.store.sample_path(sample_id)
        file_path = sample_dir / rel_name
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(file_path)
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_argparser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Review and label saved segmentation samples.")
    parser.add_argument("--samples-dir", default=str(script_dir / "segmentation_samples"))
    parser.add_argument("--labels", default=str(script_dir / "mahjong_labels.txt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the tool in the default browser")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    store = AnnotationStore(Path(args.samples_dir), Path(args.labels))
    AnnotationHandler.store = store
    httpd = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[Annotator] Serving {store.samples_dir}")
    print(f"[Annotator] Open {url}")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("[Annotator] stopped")


if __name__ == "__main__":
    main()
