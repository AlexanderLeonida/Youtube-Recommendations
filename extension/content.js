/**
 * TwinTube Tracker - YouTube browsing behavior tracker.
 *
 * Tracks two types of events:
 *   1. Impressions: videos visible in the user's viewport (feed, sidebar, search)
 *   2. Clicks: videos the user navigates to watch
 *
 * Sends events to the backend API for storage and ML training.
 */

const BACKEND_URL = "http://localhost:4000";
const IMPRESSION_THRESHOLD_MS = 1000; // Video must be visible for 1s to count
const SEND_INTERVAL_MS = 5000; // Batch-send events every 5s
const SESSION_ID = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

// ── State ──────────────────────────────────────────────────────────────────

const pendingImpressions = new Map(); // videoId -> {videoData, visibleSince}
const sentImpressions = new Set();    // videoIds already sent this page
let eventQueue = [];                  // batched events waiting to send
let currentWatchVideoId = null;       // video ID if we're on a watch page
let watchStartTime = null;            // when the current watch started

// ── Video extraction from DOM ──────────────────────────────────────────────

function extractVideoFromCard(card) {
  const titleEl = card.querySelector("#video-title");
  if (!titleEl) return null;

  const title = (titleEl.getAttribute("title") || titleEl.textContent || "").trim();
  if (!title || title.length < 3) return null;

  // Video ID from link
  let videoId = null;
  const link = card.querySelector("a#thumbnail, a#video-title, a[href*='watch']");
  if (link) {
    const href = link.getAttribute("href") || "";
    const match = href.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
    if (match) videoId = match[1];
  }
  if (!videoId) return null;

  // Channel
  let channel = null;
  const chanEl =
    card.querySelector("ytd-channel-name a") ||
    card.querySelector("ytd-channel-name yt-formatted-string");
  if (chanEl) channel = chanEl.textContent.trim();

  // Views + posted time
  let views = null;
  let postedAgo = null;
  const metaSpans = card.querySelectorAll("#metadata-line span");
  metaSpans.forEach((span) => {
    const txt = span.textContent.trim();
    if (/views?/i.test(txt)) views = txt;
    else if (/ago|streamed|premiered/i.test(txt)) postedAgo = txt;
  });

  // Duration
  let duration = null;
  const durEl = card.querySelector(
    "ytd-thumbnail-overlay-time-status-renderer span"
  );
  if (durEl) duration = durEl.textContent.trim();

  return { videoId, title, channel, views, duration, postedAgo };
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
        // Video entered viewport
        if (!pendingImpressions.has(data.videoId)) {
          pendingImpressions.set(data.videoId, {
            ...data,
            visibleSince: now,
          });
        }
      } else {
        // Video left viewport — check if it was visible long enough
        const pending = pendingImpressions.get(data.videoId);
        if (pending) {
          const visibleMs = now - pending.visibleSince;
          if (visibleMs >= IMPRESSION_THRESHOLD_MS && !sentImpressions.has(data.videoId)) {
            queueEvent("impression", data);
            sentImpressions.add(data.videoId);
          }
          pendingImpressions.delete(data.videoId);
        }
      }
    });
  },
  { threshold: 0.5 } // 50% of the card must be visible
);

function observeVideoCards() {
  const selectors = [
    "ytd-rich-item-renderer",       // homepage feed
    "ytd-compact-video-renderer",   // watch page sidebar
    "ytd-video-renderer",           // search results
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

function checkWatchPage() {
  const url = window.location.href;
  const match = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
  const videoId = match ? match[1] : null;

  if (videoId && videoId !== currentWatchVideoId) {
    // User navigated to a new video — record the click
    if (currentWatchVideoId && watchStartTime) {
      // End previous watch session
      const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
      queueEvent("watch_end", {
        videoId: currentWatchVideoId,
        watchDurationSec,
      });
    }

    currentWatchVideoId = videoId;
    watchStartTime = Date.now();

    // Extract title from the watch page
    const titleEl = document.querySelector(
      "yt-formatted-string.ytd-watch-metadata, h1.ytd-watch-metadata yt-formatted-string"
    );
    const channelEl = document.querySelector(
      "ytd-channel-name yt-formatted-string a, ytd-video-owner-renderer ytd-channel-name a"
    );

    queueEvent("click", {
      videoId,
      title: titleEl ? titleEl.textContent.trim() : null,
      channel: channelEl ? channelEl.textContent.trim() : null,
    });
  } else if (!videoId && currentWatchVideoId) {
    // Left watch page
    const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
    queueEvent("watch_end", {
      videoId: currentWatchVideoId,
      watchDurationSec,
    });
    currentWatchVideoId = null;
    watchStartTime = null;
  }
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
}

async function flushEvents() {
  // Also flush any currently-visible impressions that have been visible long enough
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
    await fetch(`${BACKEND_URL}/api/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: batch }),
    });
  } catch (err) {
    // Put events back if send failed
    eventQueue.unshift(...batch);
    console.debug("[TwinTube] Send failed, will retry:", err.message);
  }
}

// ── MutationObserver to catch dynamically-loaded video cards ───────────────

const domObserver = new MutationObserver(() => {
  observeVideoCards();
  checkWatchPage();
});

// ── Init ───────────────────────────────────────────────────────────────────

function init() {
  console.log("[TwinTube] Tracker loaded, session:", SESSION_ID);

  observeVideoCards();
  checkWatchPage();

  // Watch for new video cards added to the DOM (infinite scroll, navigation)
  domObserver.observe(document.body, { childList: true, subtree: true });

  // Batch-send events periodically
  setInterval(flushEvents, SEND_INTERVAL_MS);

  // Flush on page unload
  window.addEventListener("beforeunload", () => {
    if (currentWatchVideoId && watchStartTime) {
      const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
      queueEvent("watch_end", {
        videoId: currentWatchVideoId,
        watchDurationSec,
      });
    }
    flushEvents();
  });

  // YouTube uses History API for navigation (SPA), so listen for URL changes
  let lastUrl = location.href;
  new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      // Reset impression tracking for new page
      sentImpressions.clear();
      pendingImpressions.clear();
      // Re-scan after YouTube renders new content
      setTimeout(() => {
        observeVideoCards();
        checkWatchPage();
      }, 1500);
    }
  }).observe(document.querySelector("title"), { childList: true });
}

// YouTube might not be fully loaded yet
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
