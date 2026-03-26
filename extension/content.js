/**
 * TwinTube Tracker - YouTube browsing behavior tracker.
 *
 * Tracks:
 *   1. Impressions: videos visible in the user's viewport (home, search, sidebar, etc.)
 *   2. Clicks: videos the user navigates to watch (one per video, with title)
 *   3. Watch duration: how long the user watches each video
 */

const BACKEND_URL = "http://localhost:4000";
const IMPRESSION_THRESHOLD_MS = 1000;
const SEND_INTERVAL_MS = 5000;
const SESSION_ID = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

// ── State ──────────────────────────────────────────────────────────────────

const sentImpressions = new Set();      // videoIds already sent as impressions
const sentClicks = new Set();           // prevent duplicate click events
let eventQueue = [];
let currentWatchVideoId = null;
let watchStartTime = null;
let lastCheckedUrl = null;
let sidebarScanTimer = null;
let previousWatchTitle = null;          // title of the video we just left

// ── Video card selectors (all page types) ─────────────────────────────────

const CARD_SELECTORS = [
  "ytd-rich-item-renderer",          // home feed grid items
  "ytd-compact-video-renderer",      // sidebar (legacy)
  "ytd-video-renderer",              // search results
  "ytd-reel-item-renderer",          // shorts shelf
  "ytd-grid-video-renderer",         // channel page grid
  "ytd-playlist-video-renderer",     // playlist items
  "ytd-playlist-panel-video-renderer", // radio/mix playlist sidebar items
  "yt-lockup-view-model",            // sidebar recommendations (current YouTube)
];

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
      title = ytStr.getAttribute("title") || ytStr.textContent || "";
    }
  }
  // yt-lockup-view-model: title in metadata h3 a or aria-label
  if (!title || title.trim().length < 3) {
    const lockupMeta = card.querySelector("yt-lockup-metadata-view-model h3 a");
    if (lockupMeta) {
      title = lockupMeta.getAttribute("aria-label") || lockupMeta.textContent || "";
    }
  }
  title = (title || "").trim();
  if (!title || title.length < 3) return null;

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

  let channel = null;
  for (const sel of [
    "ytd-channel-name a",
    "ytd-channel-name yt-formatted-string",
    "#channel-name a",
    "#channel-name yt-formatted-string",
    ".ytd-channel-name",
    "#byline a",                       // playlist panel renderer
    "#byline yt-formatted-string",     // playlist panel renderer
    "span#byline",                     // playlist panel renderer alt
    "yt-lockup-metadata-view-model .yt-content-metadata-view-model-wiz__metadata-text a", // lockup sidebar
    "yt-lockup-metadata-view-model a[href*='/@']", // lockup sidebar channel link
    ".yt-content-metadata-view-model-wiz__metadata-text a", // lockup metadata
    ".publisher",                      // generic fallback
  ]) {
    const el = card.querySelector(sel);
    if (el && el.textContent.trim()) { channel = el.textContent.trim(); break; }
  }

  let views = null;
  let postedAgo = null;
  card.querySelectorAll("#metadata-line span, .inline-metadata-item, .yt-content-metadata-view-model-wiz__metadata-text span").forEach((span) => {
    const txt = span.textContent.trim();
    if (/views?/i.test(txt) && !views) views = txt;
    else if (/ago|streamed|premiered/i.test(txt) && !postedAgo) postedAgo = txt;
  });

  let duration = null;
  for (const sel of [
    "ytd-thumbnail-overlay-time-status-renderer span",
    "badge-shape .badge-shape-wiz__text",
    "#time-status span",
    ".yt-thumbnail-overlay-badge-view-model .badge-shape-wiz__text", // lockup sidebar
  ]) {
    const el = card.querySelector(sel);
    if (el && el.textContent.trim()) { duration = el.textContent.trim(); break; }
  }

  return { videoId, title, channel, views, duration, postedAgo };
}

// ── Watch page title extraction ─────────────────────────────────────────────

function extractWatchPageTitle() {
  for (const sel of [
    "h1.ytd-watch-metadata yt-formatted-string",
    "yt-formatted-string.ytd-watch-metadata",
    "h1.ytd-video-primary-info-renderer yt-formatted-string",
    "#title h1 yt-formatted-string",
    "ytd-watch-metadata #title yt-formatted-string",
    "#above-the-fold #title yt-formatted-string",
  ]) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim().length > 2) {
      return el.textContent.trim();
    }
  }
  return null;
}

function extractWatchPageChannel() {
  for (const sel of [
    "ytd-video-owner-renderer ytd-channel-name a",
    "ytd-video-owner-renderer ytd-channel-name yt-formatted-string",
    "#owner ytd-channel-name a",
    "#channel-name a",
    "ytd-channel-name yt-formatted-string a",
  ]) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) {
      return el.textContent.trim();
    }
  }
  return null;
}

// ── Impression tracking ───────────────────────────────────────────────────

// IntersectionObserver for home/search feeds (cards that scroll in and out)
const observer = new IntersectionObserver(
  (entries) => {
    const now = Date.now();
    entries.forEach((entry) => {
      const card = entry.target;
      const data = extractVideoFromCard(card);
      if (!data) return;

      if (entry.isIntersecting) {
        if (!card.dataset.twintubeVisible) {
          card.dataset.twintubeVisible = String(now);
        }
      } else {
        const visibleSince = parseInt(card.dataset.twintubeVisible || "0");
        if (visibleSince > 0) {
          const visibleMs = now - visibleSince;
          if (visibleMs >= IMPRESSION_THRESHOLD_MS && !sentImpressions.has(data.videoId)) {
            queueEvent("impression", data);
            sentImpressions.add(data.videoId);
          }
          card.dataset.twintubeVisible = "";
        }
      }
    });
  },
  { threshold: 0.2 }
);

function observeVideoCards() {
  let newCards = 0;
  CARD_SELECTORS.forEach((sel) => {
    document.querySelectorAll(sel).forEach((card) => {
      if (!card.dataset.twintube) {
        card.dataset.twintube = "1";
        observer.observe(card);
        newCards++;
      }
    });
  });
  if (newCards > 0) {
    console.debug(`[TwinTube] Observing ${newCards} new video cards`);
  }
}

/**
 * Directly scan all video cards and IMMEDIATELY record visible ones as
 * impressions. This is critical for sidebar cards which stay visible
 * and never trigger IntersectionObserver's "exit" callback.
 */
function scanAndRecordVisibleCards() {
  let found = 0;

  CARD_SELECTORS.forEach((sel) => {
    document.querySelectorAll(sel).forEach((card) => {
      if (!card.dataset.twintube) {
        card.dataset.twintube = "1";
        observer.observe(card);
      }

      const data = extractVideoFromCard(card);
      if (!data) return;
      if (sentImpressions.has(data.videoId)) return;

      // Skip the currently watched video — that's a click, not an impression
      if (data.videoId === currentWatchVideoId) return;

      // Check if any part of the card is in the viewport
      const rect = card.getBoundingClientRect();
      const isVisible = rect.top < window.innerHeight && rect.bottom > 0
        && rect.left < window.innerWidth && rect.right > 0
        && rect.height > 0 && rect.width > 0;

      if (isVisible) {
        queueEvent("impression", data);
        sentImpressions.add(data.videoId);
        found++;
      }
    });
  });

  if (found > 0) {
    console.debug(`[TwinTube] Recorded ${found} sidebar/visible impressions`);
  }
}

// ── Click / watch tracking ─────────────────────────────────────────────────

function endCurrentWatch() {
  if (currentWatchVideoId && watchStartTime) {
    const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
    const title = extractWatchPageTitle();
    const channel = extractWatchPageChannel();
    if (watchDurationSec > 0) {
      queueEvent("watch_end", { videoId: currentWatchVideoId, watchDurationSec, title, channel });
    }
    // Remember the title we're leaving so we can detect stale DOM
    previousWatchTitle = title;
  }
  currentWatchVideoId = null;
  watchStartTime = null;
}

function checkUrlChange() {
  const url = window.location.href;
  if (url === lastCheckedUrl) return;
  lastCheckedUrl = url;

  const match = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
  const videoId = match ? match[1] : null;

  if (videoId && videoId !== currentWatchVideoId) {
    endCurrentWatch();

    currentWatchVideoId = videoId;
    watchStartTime = Date.now();

    scheduleClickEvent(videoId);

    // Reset impression tracking for new page context
    sentImpressions.clear();

    // Repeatedly scan sidebar as YouTube lazily loads cards
    startSidebarScanning();

  } else if (!videoId && currentWatchVideoId) {
    endCurrentWatch();
    sentImpressions.clear();
    stopSidebarWatching();
  }
}

/**
 * After navigating to a watch page, aggressively scan for sidebar cards.
 * YouTube lazy-loads sidebar recommendations, so we scan repeatedly and
 * also watch for new cards being added to the sidebar container.
 */
let sidebarMutationObserver = null;
let sidebarScrollHandler = null;

function startSidebarScanning() {
  if (sidebarScanTimer) clearInterval(sidebarScanTimer);
  stopSidebarWatching();

  // Immediate first scan
  observeVideoCards();
  scanAndRecordVisibleCards();

  // Then scan every 2s for 30 seconds (15 scans) to catch lazy-loaded cards
  let scansRemaining = 15;
  sidebarScanTimer = setInterval(() => {
    observeVideoCards();
    scanAndRecordVisibleCards();
    scansRemaining--;
    if (scansRemaining <= 0) {
      clearInterval(sidebarScanTimer);
      sidebarScanTimer = null;
    }
  }, 2000);

  // Watch the sidebar container for new cards being inserted
  startSidebarWatching();
}

function startSidebarWatching() {
  // Find the sidebar container and observe it for new children
  const sidebarSelectors = [
    "ytd-watch-next-secondary-results-renderer",
    "#secondary-inner",
    "#related",
    "#items.ytd-watch-next-secondary-results-renderer",
    "#secondary",
  ];

  for (const sel of sidebarSelectors) {
    const container = document.querySelector(sel);
    if (container) {
      sidebarMutationObserver = new MutationObserver(() => {
        observeVideoCards();
        setTimeout(scanAndRecordVisibleCards, 500);
      });
      sidebarMutationObserver.observe(container, { childList: true, subtree: true });
      console.debug(`[TwinTube] Watching sidebar via "${sel}"`);
      break;
    }
  }

  // If we didn't find a sidebar container yet, retry after a delay
  if (!sidebarMutationObserver) {
    setTimeout(() => {
      if (!sidebarMutationObserver && window.location.href.includes("/watch")) {
        startSidebarWatching();
      }
    }, 3000);
  }

  // Also listen for scroll (sidebar scrolls with page on most layouts)
  sidebarScrollHandler = () => {
    scanAndRecordVisibleCards();
  };
  window.addEventListener("scroll", sidebarScrollHandler, { passive: true });
}

function stopSidebarWatching() {
  if (sidebarMutationObserver) {
    sidebarMutationObserver.disconnect();
    sidebarMutationObserver = null;
  }
  if (sidebarScrollHandler) {
    window.removeEventListener("scroll", sidebarScrollHandler);
    sidebarScrollHandler = null;
  }
}

/**
 * Send exactly ONE click event per video. Critically, it waits until
 * the DOM title is DIFFERENT from the previous video's title, confirming
 * YouTube has actually rendered the new video's info.
 */
function scheduleClickEvent(videoId) {
  if (sentClicks.has(videoId)) return;

  const maxAttempts = 10;
  let attempt = 0;

  function tryClick() {
    if (currentWatchVideoId !== videoId) return;
    if (sentClicks.has(videoId)) return;
    if (!window.location.href.includes(videoId)) return;

    const title = extractWatchPageTitle();
    const channel = extractWatchPageChannel();

    // Check if the title is stale (same as the video we just left)
    const titleIsStale = title && previousWatchTitle && title === previousWatchTitle;

    if (title && !titleIsStale) {
      // Title is fresh — send the click
      sentClicks.add(videoId);
      queueEvent("click", { videoId, title, channel });
      previousWatchTitle = null;
      return;
    }

    attempt++;
    if (attempt >= maxAttempts) {
      // Give up waiting — send with whatever we have (or no title)
      sentClicks.add(videoId);
      const finalTitle = titleIsStale ? null : title;
      queueEvent("click", { videoId, title: finalTitle, channel });
      previousWatchTitle = null;
      return;
    }

    // Retry: 500ms intervals for the first few, then slower
    const delay = attempt < 4 ? 500 : 1000;
    setTimeout(tryClick, delay);
  }

  // First attempt after 600ms — gives YouTube time to start rendering
  setTimeout(tryClick, 600);
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
  checkUrlChange();

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

  // Periodic flush + visible card scan
  setInterval(() => {
    flushEvents();
    // Also periodically scan for visible cards (catches sidebar, lazy-loaded content)
    if (window.location.href.includes("/watch")) {
      scanAndRecordVisibleCards();
    }
  }, SEND_INTERVAL_MS);

  // Flush on page unload
  window.addEventListener("beforeunload", () => {
    if (currentWatchVideoId && watchStartTime) {
      const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
      const title = extractWatchPageTitle();
      const channel = extractWatchPageChannel();
      if (watchDurationSec > 0) {
        queueEvent("watch_end", { videoId: currentWatchVideoId, watchDurationSec, title, channel });
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

  // yt-navigate-start: capture watch_end BEFORE navigation
  window.addEventListener("yt-navigate-start", () => {
    if (currentWatchVideoId && watchStartTime) {
      const watchDurationSec = Math.round((Date.now() - watchStartTime) / 1000);
      const title = extractWatchPageTitle();
      const channel = extractWatchPageChannel();
      if (watchDurationSec > 0) {
        queueEvent("watch_end", { videoId: currentWatchVideoId, watchDurationSec, title, channel });
      }
      previousWatchTitle = title;
      currentWatchVideoId = null;
      watchStartTime = null;
    }
  });

  // yt-navigate-finish
  window.addEventListener("yt-navigate-finish", () => {
    checkUrlChange();
    setTimeout(observeVideoCards, 500);
  });

  // Tab switch
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      checkUrlChange();
      observeVideoCards();
      scanAndRecordVisibleCards();
    }
  });

  // Fallback: title change
  const titleEl = document.querySelector("title");
  if (titleEl) {
    new MutationObserver(() => {
      checkUrlChange();
      setTimeout(observeVideoCards, 1000);
    }).observe(titleEl, { childList: true });
  }

  // Safety net: periodic URL check
  setInterval(checkUrlChange, 2000);

  // If already on a watch page, scan sidebar
  if (window.location.href.includes("/watch")) {
    startSidebarScanning();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
