/**
 * TwinTube Tracker - YouTube browsing behavior tracker.
 *
 * Tracks:
 *   1. Impressions: videos visible in the user's viewport
 *   2. Clicks: videos the user navigates to watch (one per video, with title)
 *   3. Watch duration: how long the user watches each video
 */

const BACKEND_URL = "http://localhost:4000";
const IMPRESSION_THRESHOLD_MS = 1000;
const SEND_INTERVAL_MS = 5000;
const SESSION_ID = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

// ── State ──────────────────────────────────────────────────────────────────

const pendingImpressions = new Map();
const sentImpressions = new Set();
const sentClicks = new Set(); // prevent duplicate click events per video
let eventQueue = [];
let currentWatchVideoId = null;
let watchStartTime = null;
let lastCheckedUrl = null; // for URL-change detection outside debounce

// ── Video extraction from DOM ──────────────────────────────────────────────

function extractVideoFromCard(card) {
  let title = null;
  let titleEl = card.querySelector("#video-title");
  if (titleEl) {
    title = titleEl.getAttribute("title") || titleEl.textContent || "";
  }
  if (!title || title.trim().length < 3) {
    const titleLink = card.querySelector("a[title]");
    if (titleLink) {
      title = titleLink.getAttribute("title") || "";
    }
  }
  if (!title || title.trim().length < 3) {
    const ytStr = card.querySelector("yt-formatted-string#video-title, h3 a, h3 yt-formatted-string");
    if (ytStr) {
      title = ytStr.textContent || "";
    }
  }
  title = (title || "").trim();
  if (!title || title.length < 3) return null;

  // Video ID from link
  let videoId = null;
  const links = card.querySelectorAll("a[href]");
  for (const link of links) {
    const href = link.getAttribute("href") || "";
    const match = href.match(/\/watch\?v=([a-zA-Z0-9_-]{11})/);
    if (match) { videoId = match[1]; break; }
    const shortsMatch = href.match(/\/shorts\/([a-zA-Z0-9_-]{11})/);
    if (shortsMatch) { videoId = shortsMatch[1]; break; }
  }
  if (!videoId) return null;

  // Channel
  let channel = null;
  for (const sel of [
    "ytd-channel-name a",
    "ytd-channel-name yt-formatted-string",
    "#channel-name a",
    "#channel-name yt-formatted-string",
    ".ytd-channel-name",
  ]) {
    const el = card.querySelector(sel);
    if (el && el.textContent.trim()) { channel = el.textContent.trim(); break; }
  }

  // Views + posted time
  let views = null;
  let postedAgo = null;
  card.querySelectorAll("#metadata-line span, .inline-metadata-item").forEach((span) => {
    const txt = span.textContent.trim();
    if (/views?/i.test(txt) && !views) views = txt;
    else if (/ago|streamed|premiered/i.test(txt) && !postedAgo) postedAgo = txt;
  });

  // Duration
  let duration = null;
  for (const sel of [
    "ytd-thumbnail-overlay-time-status-renderer span",
    "badge-shape .badge-shape-wiz__text",
    "#time-status span",
  ]) {
    const el = card.querySelector(sel);
    if (el && el.textContent.trim()) { duration = el.textContent.trim(); break; }
  }

  return { videoId, title, channel, views, duration, postedAgo };
}

// ── Watch page title extraction ─────────────────────────────────────────────

function extractWatchPageInfo() {
  const titleSelectors = [
    "h1.ytd-watch-metadata yt-formatted-string",
    "yt-formatted-string.ytd-watch-metadata",
    "h1.ytd-video-primary-info-renderer yt-formatted-string",
    "#title h1 yt-formatted-string",
    "ytd-watch-metadata #title yt-formatted-string",
    "#above-the-fold #title yt-formatted-string",
  ];
  let title = null;
  for (const sel of titleSelectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim().length > 2) {
      title = el.textContent.trim();
      break;
    }
  }

  const channelSelectors = [
    "ytd-video-owner-renderer ytd-channel-name a",
    "ytd-video-owner-renderer ytd-channel-name yt-formatted-string",
    "#owner ytd-channel-name a",
    "#channel-name a",
    "ytd-channel-name yt-formatted-string a",
  ];
  let channel = null;
  for (const sel of channelSelectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) {
      channel = el.textContent.trim();
      break;
    }
  }

  return { title, channel };
}

// ── Impression tracking with IntersectionObserver ──────────────────────────

const observer = new IntersectionObserver(
  (entries) => {
    const now = Date.now();
    entries.forEach((entry) => {
      const card = entry.target;
      const data = extractVideoFromCard(card);
      if (!data) return;

      if (entry.isIntersecting) {
        if (!pendingImpressions.has(data.videoId)) {
          pendingImpressions.set(data.videoId, { ...data, visibleSince: now });
        }
      } else {
        const pending = pendingImpressions.get(data.videoId);
        if (pending) {
          const visibleMs = now - pending.visibleSince;
          if (visibleMs >= IMPRESSION_THRESHOLD_MS && !sentImpressions.has(data.videoId)) {
            queueEvent("impression", pending);
            sentImpressions.add(data.videoId);
          }
          pendingImpressions.delete(data.videoId);
        }
      }
    });
  },
  { threshold: 0.5 }
);

function observeVideoCards() {
  const selectors = [
    "ytd-rich-item-renderer",
    "ytd-compact-video-renderer",
    "ytd-video-renderer",
    "ytd-reel-item-renderer",
  ];
  selectors.forEach((sel) => {
    document.querySelectorAll(sel).forEach((card) => {
      if (!card.dataset.twintube) {
        card.dataset.twintube = "1";
        observer.observe(card);
      }
    });
  });
}

// ── Click / watch tracking ─────────────────────────────────────────────────

/**
 * Immediately detect URL changes for watch tracking.
 * Called on every mutation (NOT debounced) so we never miss a navigation.
 */
function checkUrlChange() {
  const url = window.location.href;
  if (url === lastCheckedUrl) return;
  lastCheckedUrl = url;

  const match = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
  const videoId = match ? match[1] : null;

  if (videoId && videoId !== currentWatchVideoId) {
    // ── End previous watch ──
    if (currentWatchVideoId && watchStartTime) {
      const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
      if (watchDurationSec > 0) {
        queueEvent("watch_end", { videoId: currentWatchVideoId, watchDurationSec });
      }
    }

    // ── Start new watch ──
    currentWatchVideoId = videoId;
    watchStartTime = Date.now();

    // Schedule a single click event, retrying until we get the title
    scheduleClickEvent(videoId);

  } else if (!videoId && currentWatchVideoId) {
    // ── Left watch page (went to home, search, etc.) ──
    const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
    if (watchDurationSec > 0) {
      queueEvent("watch_end", { videoId: currentWatchVideoId, watchDurationSec });
    }
    currentWatchVideoId = null;
    watchStartTime = null;
  }
}

/**
 * Send exactly ONE click event per video. Retries to capture the title
 * before sending, but sends with whatever we have after all retries.
 */
function scheduleClickEvent(videoId) {
  if (sentClicks.has(videoId)) return;

  const retryDelays = [0, 400, 1200, 2500, 5000];
  let attempt = 0;

  function tryClick() {
    // Navigated away — don't send
    if (currentWatchVideoId !== videoId) return;
    // Already sent (e.g., from a parallel retry)
    if (sentClicks.has(videoId)) return;

    const info = extractWatchPageInfo();

    if (info.title || attempt >= retryDelays.length - 1) {
      // Send it — we either have the title or have exhausted retries
      sentClicks.add(videoId);
      queueEvent("click", {
        videoId,
        title: info.title,
        channel: info.channel,
      });
      return;
    }

    attempt++;
    setTimeout(tryClick, retryDelays[attempt]);
  }

  // Start first attempt immediately (delay = 0)
  tryClick();
}

// ── Event queue and sending ────────────────────────────────────────────────

function queueEvent(type, data) {
  eventQueue.push({
    type,
    sessionId: SESSION_ID,
    timestamp: new Date().toISOString(),
    url: window.location.href,
    ...data,
  });
  console.debug("[TwinTube]", type, data.title || data.videoId);
}

async function flushEvents() {
  // Flush visible impressions that have been on screen long enough
  const now = Date.now();
  pendingImpressions.forEach((pending, videoId) => {
    if (
      now - pending.visibleSince >= IMPRESSION_THRESHOLD_MS &&
      !sentImpressions.has(videoId)
    ) {
      queueEvent("impression", {
        videoId: pending.videoId,
        title: pending.title,
        channel: pending.channel,
        views: pending.views,
        duration: pending.duration,
        postedAgo: pending.postedAgo,
      });
      sentImpressions.add(videoId);
    }
  });

  if (eventQueue.length === 0) return;

  const batch = eventQueue.splice(0);
  try {
    const resp = await fetch(`${BACKEND_URL}/api/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: batch }),
    });
    if (!resp.ok) {
      console.warn("[TwinTube] Send failed:", resp.status);
      eventQueue.unshift(...batch);
    }
  } catch (err) {
    eventQueue.unshift(...batch);
    console.debug("[TwinTube] Send failed, will retry:", err.message);
  }
}

// ── MutationObserver for dynamic content ───────────────────────────────────

let mutationTimer = null;
const domObserver = new MutationObserver(() => {
  // URL check runs on EVERY mutation — never debounced, so we catch navigations instantly
  checkUrlChange();

  // Card observation is debounced (YouTube fires many mutations)
  if (mutationTimer) clearTimeout(mutationTimer);
  mutationTimer = setTimeout(() => {
    observeVideoCards();
  }, 300);
});

// ── Init ───────────────────────────────────────────────────────────────────

function init() {
  console.log("[TwinTube] Tracker loaded, session:", SESSION_ID);

  lastCheckedUrl = window.location.href;
  observeVideoCards();
  checkUrlChange();

  domObserver.observe(document.body, { childList: true, subtree: true });

  setInterval(flushEvents, SEND_INTERVAL_MS);

  // Flush on page unload — capture final watch_end
  window.addEventListener("beforeunload", () => {
    if (currentWatchVideoId && watchStartTime) {
      const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
      if (watchDurationSec > 0) {
        queueEvent("watch_end", { videoId: currentWatchVideoId, watchDurationSec });
      }
    }
    if (eventQueue.length > 0) {
      navigator.sendBeacon(
        `${BACKEND_URL}/api/events`,
        new Blob([JSON.stringify({ events: eventQueue })], { type: "application/json" })
      );
      eventQueue = [];
    }
  });

  // Also listen for visibilitychange — catches tab switches that YouTube
  // sometimes uses for SPA navigation without changing the <title>
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      checkUrlChange();
    }
  });

  // YouTube SPA navigation via yt-navigate-finish (more reliable than title observer)
  window.addEventListener("yt-navigate-finish", () => {
    checkUrlChange();
    sentImpressions.clear();
    pendingImpressions.clear();
    setTimeout(observeVideoCards, 500);
  });

  // Fallback: title change observer
  const titleEl = document.querySelector("title");
  if (titleEl) {
    new MutationObserver(() => {
      checkUrlChange();
      sentImpressions.clear();
      pendingImpressions.clear();
      setTimeout(observeVideoCards, 1000);
    }).observe(titleEl, { childList: true });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
