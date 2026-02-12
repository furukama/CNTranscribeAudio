const form = document.getElementById("transcribe-form");
const urlInput = document.getElementById("youtube-url");
const useMlxInput = document.getElementById("use-mlx");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const tabChinese = document.getElementById("tab-chinese");
const tabInterlinear = document.getElementById("tab-interlinear");
const tabEnglish = document.getElementById("tab-english");
const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
const progressWrap = document.getElementById("progress-wrap");
const progressLabel = document.getElementById("progress-label");
const progressPct = document.getElementById("progress-pct");
const progressBar = document.getElementById("progress-bar");

const STORAGE_LAST_RESULT = "cntranscribe:last_result:v1";
const STORAGE_LAST_URL = "cntranscribe:last_url:v1";
const STORAGE_LAST_MLX = "cntranscribe:last_use_mlx:v1";
const DEBUG_CLIP_DURATION_SECONDS = 60;
const FALLBACK_SECONDS_PER_SEGMENT = 6;
const MIN_PAUSE_JUMP_SECONDS = 1.25;
const SYNC_INTERVAL_MS = 350;
const INTERLINEAR_TOKENS_PER_ROW = 8;

let ytPlayer = null;
let ytReadyPromise = null;
let syncTimer = null;
let renderedSegments = [];
let activeSegmentIndex = -1;

function ensureSegmentTimes(segments, totalDurationHint = null, warnings = []) {
  if (!Array.isArray(segments) || !segments.length) {
    return;
  }

  const hasAnyStart = segments.some((s) => s.start !== null && s.start !== undefined);
  if (hasAnyStart) {
    for (let i = 0; i < segments.length; i += 1) {
      if ((segments[i].end === null || segments[i].end === undefined) && i + 1 < segments.length) {
        const nextStart = segments[i + 1].start;
        if (nextStart !== null && nextStart !== undefined) {
          segments[i].end = nextStart;
        }
      }
    }
    return;
  }

  let totalDuration = Number(totalDurationHint);
  if (!Number.isFinite(totalDuration) || totalDuration <= 0) {
    const debugWarning = Array.isArray(warnings)
      ? warnings.find((w) => typeof w === "string" && w.includes("first 60 seconds"))
      : null;
    if (debugWarning) {
      totalDuration = DEBUG_CLIP_DURATION_SECONDS;
    }
  }

  if (segments.length === 1) {
    segments[0].start = 0;
    segments[0].end =
      Number.isFinite(totalDuration) && totalDuration > 0 ? totalDuration : DEBUG_CLIP_DURATION_SECONDS;
    return;
  }

  let cursor = 0;
  let totalWeight = 0;
  const weights = [];
  for (const seg of segments) {
    const w = Math.max(1, (seg.chinese || "").length);
    weights.push(w);
    totalWeight += w;
  }

  const fallbackDuration =
    Number.isFinite(totalDuration) && totalDuration > 0
      ? totalDuration
      : segments.length * FALLBACK_SECONDS_PER_SEGMENT;
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    const textLen = (seg.chinese || "").length;
    const weighted = (weights[i] / totalWeight) * fallbackDuration;
    const dur = Math.max(2, Math.min(18, weighted || Math.round(textLen / 3) || 4));
    seg.start = cursor;
    seg.end = i === segments.length - 1 ? fallbackDuration : cursor + dur;
    cursor += dur;
  }
}

function toHms(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "--:--";
  }

  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;

  if (h > 0) {
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatTimeRange(start, end) {
  if (start === null && end === null) {
    return "";
  }
  if (start === null || end === null) {
    return "";
  }
  return `${toHms(start)} - ${toHms(end)}`;
}

function seekTo(time) {
  if (!ytPlayer || typeof ytPlayer.seekTo !== "function") {
    return;
  }
  if (time !== null && time !== undefined) {
    ytPlayer.seekTo(time, true);
    const idx = findSegmentIndexForTime(time);
    setActiveSegment(idx);
  }
}

function createTimeButton(start, end) {
  if (start === null || end === null) {
    return null;
  }
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "segment-time-btn";
  btn.textContent = formatTimeRange(start, end);
  btn.addEventListener("click", (event) => {
    event.stopPropagation();
    seekTo(start);
  });
  return btn;
}

function maybeAddPauseJump(container, previousSegment, nextSegment) {
  if (!previousSegment || !nextSegment) {
    return;
  }
  if (previousSegment.end === null || nextSegment.start === null) {
    return;
  }

  const gapSeconds = nextSegment.start - previousSegment.end;
  if (gapSeconds < MIN_PAUSE_JUMP_SECONDS) {
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "pause-jump";

  const label = document.createElement("span");
  label.textContent = `Pause ${gapSeconds.toFixed(1)}s`;

  const jumpBtn = document.createElement("button");
  jumpBtn.type = "button";
  jumpBtn.textContent = `Jump ${toHms(nextSegment.start)}`;
  jumpBtn.addEventListener("click", () => seekTo(nextSegment.start));

  wrap.appendChild(label);
  wrap.appendChild(jumpBtn);
  container.appendChild(wrap);
}

function clearOutput() {
  tabChinese.innerHTML = "";
  tabInterlinear.innerHTML = "";
  tabEnglish.innerHTML = "";
  renderedSegments = [];
}

function renderEmptyState(text) {
  clearOutput();
  tabChinese.innerHTML = `<p class="empty">${text}</p>`;
  tabInterlinear.innerHTML = `<p class="empty">${text}</p>`;
  tabEnglish.innerHTML = `<p class="empty">${text}</p>`;
}

function setProgress(percent, message) {
  const p = Math.max(0, Math.min(100, Number(percent) || 0));
  progressWrap.classList.remove("hidden");
  progressBar.style.width = `${p}%`;
  progressPct.textContent = `${p}%`;
  progressLabel.textContent = message || "Working...";
}

function hideProgress() {
  progressWrap.classList.add("hidden");
}

function saveUiPrefs() {
  try {
    localStorage.setItem(STORAGE_LAST_URL, urlInput.value || "");
    localStorage.setItem(STORAGE_LAST_MLX, useMlxInput.checked ? "1" : "0");
  } catch (_err) {
    // Ignore storage errors (private mode / quota).
  }
}

function saveLastResult(result) {
  try {
    localStorage.setItem(STORAGE_LAST_RESULT, JSON.stringify(result));
    saveUiPrefs();
  } catch (_err) {
    // Ignore storage errors.
  }
}

function loadLastResult() {
  try {
    const raw = localStorage.getItem(STORAGE_LAST_RESULT);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.segments)) {
      return null;
    }
    return parsed;
  } catch (_err) {
    return null;
  }
}

function loadUiPrefs() {
  try {
    const savedUrl = localStorage.getItem(STORAGE_LAST_URL);
    if (savedUrl) {
      urlInput.value = savedUrl;
    }
    const savedMlx = localStorage.getItem(STORAGE_LAST_MLX);
    if (savedMlx === "0" || savedMlx === "1") {
      useMlxInput.checked = savedMlx === "1";
    }
  } catch (_err) {
    // Ignore storage errors.
  }
}

function parseYouTubeVideoId(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("youtu.be")) {
      return parsed.pathname.replace("/", "").trim();
    }
    return parsed.searchParams.get("v");
  } catch (_err) {
    return null;
  }
}

function loadYouTubeApi() {
  if (window.YT && window.YT.Player) {
    return Promise.resolve();
  }

  if (ytReadyPromise) {
    return ytReadyPromise;
  }

  ytReadyPromise = new Promise((resolve) => {
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);

    const previous = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      if (typeof previous === "function") {
        previous();
      }
      resolve();
    };
  });

  return ytReadyPromise;
}

async function ensurePlayer(url) {
  const videoId = parseYouTubeVideoId(url);
  if (!videoId) {
    throw new Error("Could not parse YouTube video ID from URL.");
  }

  await loadYouTubeApi();

  if (!ytPlayer) {
    ytPlayer = new window.YT.Player("youtube-player", {
      height: "390",
      width: "640",
      videoId,
      playerVars: {
        rel: 0,
        modestbranding: 1,
      },
    });
  } else {
    ytPlayer.cueVideoById(videoId);
  }
}

function setActiveSegment(index) {
  for (const segment of renderedSegments) {
    segment.elements.forEach((el) => el.classList.remove("active"));
  }

  activeSegmentIndex = index;
  if (index >= 0 && index < renderedSegments.length) {
    const active = renderedSegments[index];
    active.elements.forEach((el) => el.classList.add("active"));
    const tabName = getCurrentTab();
    const tabToIdx = { chinese: 0, interlinear: 1, english: 2 };
    const el = active.elements[tabToIdx[tabName] ?? 0];
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function startSyncLoop() {
  if (syncTimer) {
    window.clearInterval(syncTimer);
  }

  syncTimer = window.setInterval(() => {
    if (!ytPlayer || typeof ytPlayer.getCurrentTime !== "function") {
      return;
    }

    const now = ytPlayer.getCurrentTime();
    const activeIndex = findSegmentIndexForTime(now);
    setActiveSegment(activeIndex);
  }, SYNC_INTERVAL_MS);
}

function findSegmentIndexForTime(now) {
  if (!renderedSegments.length) {
    return -1;
  }

  // First pass: exact interval match (with inferred end from next segment start).
  for (let i = 0; i < renderedSegments.length; i += 1) {
    const start = renderedSegments[i].start;
    if (start === null) {
      continue;
    }
    let end = renderedSegments[i].end;
    if (end === null && i + 1 < renderedSegments.length) {
      const nextStart = renderedSegments[i + 1].start;
      if (nextStart !== null) {
        end = nextStart;
      }
    }
    if (end === null) {
      end = start + 6;
    }
    if (now >= start && now <= end) {
      return i;
    }
  }

  // Fallback: nearest segment start.
  let bestIdx = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < renderedSegments.length; i += 1) {
    const start = renderedSegments[i].start;
    if (start === null) {
      continue;
    }
    const dist = Math.abs(now - start);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function buildChineseCard(segment) {
  const card = document.createElement("section");
  card.className = "segment segment-sync";

  const header = document.createElement("header");
  const timeBtn = createTimeButton(segment.start, segment.end);
  if (timeBtn) {
    header.appendChild(timeBtn);
  }

  const chinese = document.createElement("div");
  chinese.className = "chinese chinese-tokens";
  if (segment.tokens && segment.tokens.length) {
    for (const token of segment.tokens) {
      const span = document.createElement("span");
      span.className = "ch-token";
      span.textContent = token.word;
      span.title = `${token.word} | ${token.pinyin || "-"} | ${token.english || "-"}`;
      chinese.appendChild(span);
    }
  } else {
    const fallback = document.createElement("span");
    fallback.className = "ch-token";
    fallback.textContent = segment.chinese;
    fallback.title = segment.chinese;
    chinese.appendChild(fallback);
  }
  header.appendChild(chinese);

  card.appendChild(header);
  card.addEventListener("click", () => {
    const idx = findSegmentIndexForTime(segment.start ?? 0);
    setActiveSegment(idx);
    seekTo(segment.start);
  });
  return card;
}

function buildInterlinearCard(segment) {
  const card = document.createElement("section");
  card.className = "segment segment-sync";

  const header = document.createElement("header");
  const timeBtn = createTimeButton(segment.start, segment.end);
  if (timeBtn) {
    header.appendChild(timeBtn);
  }
  card.appendChild(header);

  const interlinear = document.createElement("div");
  interlinear.className = "interlinear";

  const tokens = segment.tokens.length
    ? segment.tokens
    : [{ word: segment.chinese, pinyin: "", english: segment.full_english || "" }];
  for (let i = 0; i < tokens.length; i += INTERLINEAR_TOKENS_PER_ROW) {
    const chunk = tokens.slice(i, i + INTERLINEAR_TOKENS_PER_ROW);
    const cols = `repeat(${chunk.length}, minmax(72px, 1fr))`;

    const lineZh = document.createElement("div");
    lineZh.className = "interlinear-line zh";
    lineZh.style.gridTemplateColumns = cols;

    const linePy = document.createElement("div");
    linePy.className = "interlinear-line py";
    linePy.style.gridTemplateColumns = cols;

    const lineEn = document.createElement("div");
    lineEn.className = "interlinear-line en";
    lineEn.style.gridTemplateColumns = cols;

    for (const token of chunk) {
      const zh = document.createElement("span");
      zh.textContent = token.word;
      lineZh.appendChild(zh);

      const py = document.createElement("span");
      py.textContent = token.pinyin || " ";
      linePy.appendChild(py);

      const en = document.createElement("span");
      en.textContent = token.english || " ";
      lineEn.appendChild(en);
    }

    interlinear.appendChild(lineZh);
    interlinear.appendChild(linePy);
    interlinear.appendChild(lineEn);

    if (i + INTERLINEAR_TOKENS_PER_ROW < tokens.length) {
      const sep = document.createElement("hr");
      sep.className = "interlinear-sep";
      interlinear.appendChild(sep);
    }
  }

  if (!tokens.length) {
    const sep = document.createElement("hr");
    sep.className = "interlinear-sep";
    interlinear.appendChild(sep);
  }

  card.appendChild(interlinear);
  card.addEventListener("click", () => {
    const idx = findSegmentIndexForTime(segment.start ?? 0);
    setActiveSegment(idx);
    seekTo(segment.start);
  });
  return card;
}

function buildEnglishCard(segment) {
  const card = document.createElement("section");
  card.className = "segment segment-sync";

  const header = document.createElement("header");
  const timeBtn = createTimeButton(segment.start, segment.end);
  if (timeBtn) {
    header.appendChild(timeBtn);
  }
  card.appendChild(header);

  const english = document.createElement("div");
  english.className = "english-block";
  english.textContent = segment.full_english;
  card.appendChild(english);

  card.addEventListener("click", () => {
    const idx = findSegmentIndexForTime(segment.start ?? 0);
    setActiveSegment(idx);
    seekTo(segment.start);
  });
  return card;
}

function renderResults(data) {
  ensureSegmentTimes(data.segments, data.audio_duration_seconds, data.warnings);
  clearOutput();

  let previous = null;
  for (const segment of data.segments) {
    maybeAddPauseJump(tabChinese, previous, segment);
    maybeAddPauseJump(tabInterlinear, previous, segment);
    maybeAddPauseJump(tabEnglish, previous, segment);

    const chineseCard = buildChineseCard(segment);
    const interlinearCard = buildInterlinearCard(segment);
    const englishCard = buildEnglishCard(segment);

    tabChinese.appendChild(chineseCard);
    tabInterlinear.appendChild(interlinearCard);
    tabEnglish.appendChild(englishCard);

    renderedSegments.push({
      elements: [chineseCard, interlinearCard, englishCard],
      start: segment.start,
      end: segment.end,
    });

    previous = segment;
  }

  startSyncLoop();
  if (renderedSegments.length) {
    const firstTs = renderedSegments.find((s) => s.start !== null)?.start ?? 0;
    setActiveSegment(findSegmentIndexForTime(firstTs));
  }
}

function activateTab(name) {
  const panels = {
    chinese: tabChinese,
    interlinear: tabInterlinear,
    english: tabEnglish,
  };

  for (const [key, panel] of Object.entries(panels)) {
    panel.classList.toggle("hidden", key !== name);
  }

  for (const btn of tabButtons) {
    btn.classList.toggle("active", btn.dataset.tab === name);
  }

  // Keep tab content synced with current video position when switching tabs.
  if (activeSegmentIndex >= 0 && activeSegmentIndex < renderedSegments.length) {
    const tabToIdx = { chinese: 0, interlinear: 1, english: 2 };
    const el = renderedSegments[activeSegmentIndex].elements[tabToIdx[name] ?? 0];
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function getCurrentTab() {
  const activeBtn = tabButtons.find((btn) => btn.classList.contains("active"));
  return activeBtn ? activeBtn.dataset.tab : "chinese";
}

for (const btn of tabButtons) {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
}

async function startTranscriptionJob(youtubeUrl, translationBackend) {
  const response = await fetch("/api/transcribe/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ youtube_url: youtubeUrl, translation_backend: translationBackend }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to start transcription job.");
  }
  return payload.job_id;
}

async function fetchJob(jobId) {
  const response = await fetch(`/api/transcribe/${jobId}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to fetch job status.");
  }
  return payload;
}

async function waitForJob(jobId) {
  while (true) {
    const job = await fetchJob(jobId);
    setProgress(job.progress || 0, job.message || "Working...");

    if (job.status === "completed") {
      return job.result;
    }
    if (job.status === "failed") {
      throw new Error(job.error || job.message || "Transcription failed.");
    }

    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = urlInput.value.trim();
  const translationBackend = useMlxInput.checked ? "mlx" : "hf";
  if (!url) {
    return;
  }
  saveUiPrefs();

  submitBtn.disabled = true;
  statusEl.textContent = "Job started...";
  setProgress(0, "Queued");
  renderEmptyState("Processing...");

  try {
    await ensurePlayer(url);
    const jobId = await startTranscriptionJob(url, translationBackend);
    const result = await waitForJob(jobId);

    renderResults(result);
    saveLastResult(result);
    const warning = (result.warnings || []).join(" ");
    statusEl.textContent = `Done (${result.translation_backend}). Processed ${result.segments.length} segment(s). ${warning}`.trim();
    setProgress(100, "Complete");
    window.setTimeout(hideProgress, 2000);
  } catch (error) {
    renderEmptyState("No output. See error above.");
    statusEl.textContent = `Error: ${error.message}`;
    setProgress(100, "Failed");
  } finally {
    submitBtn.disabled = false;
  }
});

async function restoreLastSession() {
  loadUiPrefs();
  const last = loadLastResult();
  if (!last) {
    return;
  }
  renderResults(last);
  statusEl.textContent = `Restored last result (${last.segments.length} segment(s)).`;
  const restoreUrl = last.source_url || urlInput.value;
  if (restoreUrl) {
    try {
      await ensurePlayer(restoreUrl);
    } catch (_err) {
      // Keep transcript even if player restore fails.
    }
  }
}

restoreLastSession();
