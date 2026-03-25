"""
Fast YouTube video title extraction via HTML parsing with BeautifulSoup.

Extracts video metadata directly from YouTube's embedded ytInitialData JSON,
bypassing the slow OCR pipeline for real-time display. Runs in <1s vs 5-10s
for OCR + LLM inference.

Used as the fast path in the pipeline: scraper provides instant results while
the C++ OpenCV OCR pipeline processes screen recordings in the background
for accuracy validation and watch-sequence extraction.
"""
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# YouTube embeds page data in a <script> tag as `var ytInitialData = {...};`
_YT_INIT_DATA_RE = re.compile(
    r"var\s+ytInitialData\s*=\s*(\{.+?\});\s*</script>", re.DOTALL
)


def scrape_youtube_homepage(
    cookies: Optional[Dict[str, str]] = None, timeout: float = 5.0
) -> List[Dict]:
    """
    Fetch YouTube homepage HTML and extract video metadata from the embedded
    ytInitialData JSON blob using BeautifulSoup.

    Args:
        cookies: Optional browser cookies for personalized recommendations.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of video dicts: {title, channel, views, duration, posted_ago, video_id, ...}
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    t0 = time.time()
    try:
        resp = requests.get(
            "https://www.youtube.com",
            headers=headers,
            cookies=cookies,
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[SCRAPER] Failed to fetch YouTube homepage: {e}")
        return []

    videos = parse_youtube_html(resp.text)
    elapsed = time.time() - t0
    print(f"[SCRAPER] Extracted {len(videos)} videos in {elapsed:.2f}s")
    return videos


def parse_youtube_html(html: str) -> List[Dict]:
    """
    Parse YouTube HTML using BeautifulSoup to locate the ytInitialData script
    tag, then extract video metadata from the embedded JSON.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: Find script tags containing ytInitialData
    yt_data = None
    for script in soup.find_all("script"):
        text = script.string or ""
        if "ytInitialData" not in text:
            continue
        match = re.search(r"ytInitialData\s*=\s*(\{.+\})\s*;", text, re.DOTALL)
        if match:
            try:
                yt_data = json.loads(match.group(1))
                break
            except json.JSONDecodeError:
                continue

    # Strategy 2: Regex fallback on raw HTML (handles edge cases where
    # BeautifulSoup doesn't parse the script content cleanly)
    if yt_data is None:
        match = _YT_INIT_DATA_RE.search(html)
        if match:
            try:
                yt_data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    if yt_data is None:
        print("[SCRAPER] Could not find ytInitialData in HTML")
        return []

    return _extract_videos_from_yt_data(yt_data)


def _extract_videos_from_yt_data(data: dict) -> List[Dict]:
    """
    Walk the ytInitialData JSON tree to find all videoRenderer objects
    and extract structured video metadata.
    """
    videos: List[Dict] = []
    _find_video_renderers(data, videos)
    return videos


def _find_video_renderers(obj, results: list, depth: int = 0):
    """Recursively locate videoRenderer nodes in the nested JSON."""
    if depth > 25:
        return

    if isinstance(obj, dict):
        if "videoRenderer" in obj:
            video = _parse_video_renderer(obj["videoRenderer"])
            if video:
                results.append(video)
            return
        for v in obj.values():
            _find_video_renderers(v, results, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            _find_video_renderers(item, results, depth + 1)


def _get_text(node: dict, key: str = "simpleText") -> Optional[str]:
    """Extract text from a YouTube JSON text node (simpleText or runs)."""
    if not isinstance(node, dict):
        return None
    if key in node:
        return node[key]
    runs = node.get("runs")
    if runs and isinstance(runs, list):
        return "".join(r.get("text", "") for r in runs)
    return None


def _parse_video_renderer(renderer: dict) -> Optional[Dict]:
    """Extract structured video data from a single videoRenderer object."""
    try:
        # Title (required)
        title = _get_text(renderer.get("title", {}))
        if not title or len(title) < 3:
            return None

        # Channel name
        channel = _get_text(renderer.get("ownerText", {}))
        if not channel:
            channel = _get_text(renderer.get("longBylineText", {}))

        # View count
        views = _get_text(renderer.get("viewCountText", {}))
        if not views:
            views = _get_text(renderer.get("shortViewCountText", {}))

        # Duration
        duration = _get_text(renderer.get("lengthText", {}))

        # Published time
        posted_ago = _get_text(renderer.get("publishedTimeText", {}))

        # Video ID
        video_id = renderer.get("videoId")

        return {
            "title": title,
            "channel": channel,
            "views": views,
            "duration": duration,
            "posted_ago": posted_ago,
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "source": "html_scraper",
        }
    except Exception as e:
        print(f"[SCRAPER] Error parsing videoRenderer: {e}")
        return None
