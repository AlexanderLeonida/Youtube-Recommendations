"""
YouTube Data API v3 client for fetching video metadata.

Replaces OCR/scraping with clean, structured API calls.
Free tier: 10,000 quota units/day (search=100, list=1 per call).
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
API_KEY = os.getenv("YOUTUBE_API_KEY", "")


def _get_key() -> str:
    key = API_KEY or os.getenv("YOUTUBE_API_KEY", "")
    if not key:
        raise ValueError(
            "YOUTUBE_API_KEY not set. Get one at "
            "https://console.cloud.google.com/apis/credentials"
        )
    return key


def _parse_duration(iso: str) -> str:
    """Convert ISO 8601 duration (PT1H2M30S) to human-readable (1:02:30)."""
    import re

    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return ""
    h, mn, s = m.group(1), m.group(2), m.group(3)
    h = int(h) if h else 0
    mn = int(mn) if mn else 0
    s = int(s) if s else 0
    if h > 0:
        return f"{h}:{mn:02d}:{s:02d}"
    return f"{mn}:{s:02d}"


def _format_views(count_str: Optional[str]) -> Optional[str]:
    """Format view count string (e.g. '1234567' -> '1.2M views')."""
    if not count_str:
        return None
    try:
        n = int(count_str)
    except ValueError:
        return count_str
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B views"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M views"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K views"
    return f"{n} views"


def get_trending(
    region_code: str = "US",
    category_id: str = "0",
    max_results: int = 50,
) -> List[Dict]:
    """
    Fetch trending/popular videos.  Costs 1 quota unit per call.
    category_id: 0=all, 10=music, 20=gaming, 24=entertainment, 28=sci&tech
    """
    key = _get_key()
    resp = requests.get(
        f"{API_BASE}/videos",
        params={
            "part": "snippet,contentDetails,statistics",
            "chart": "mostPopular",
            "regionCode": region_code,
            "videoCategoryId": category_id,
            "maxResults": min(max_results, 50),
            "key": key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return _parse_video_items(resp.json().get("items", []))


def search_videos(
    query: str,
    max_results: int = 25,
    order: str = "relevance",
    region_code: str = "US",
) -> List[Dict]:
    """
    Search YouTube videos.  Costs 100 quota units per call.
    order: relevance, date, rating, viewCount, title
    """
    key = _get_key()
    # Step 1: search (returns IDs only)
    resp = requests.get(
        f"{API_BASE}/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results, 50),
            "order": order,
            "regionCode": region_code,
            "key": key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    video_ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]

    if not video_ids:
        return []

    # Step 2: fetch full details (costs 1 unit)
    return get_videos_by_id(video_ids)


def get_videos_by_id(video_ids: List[str]) -> List[Dict]:
    """Fetch full video details by ID.  Costs 1 quota unit per 50 videos."""
    if not video_ids:
        return []
    key = _get_key()
    all_videos: List[Dict] = []

    # API accepts max 50 IDs per call
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = requests.get(
            f"{API_BASE}/videos",
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(batch),
                "key": key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        all_videos.extend(_parse_video_items(resp.json().get("items", [])))

    return all_videos


def get_related_videos(video_id: str, max_results: int = 25) -> List[Dict]:
    """
    Get videos related to a given video.  Costs 100 quota units.
    Useful for building recommendation training pairs.
    """
    key = _get_key()
    resp = requests.get(
        f"{API_BASE}/search",
        params={
            "part": "snippet",
            "relatedToVideoId": video_id,
            "type": "video",
            "maxResults": min(max_results, 50),
            "key": key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    video_ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
    if not video_ids:
        return []
    return get_videos_by_id(video_ids)


def get_channel_videos(
    channel_id: str, max_results: int = 50, order: str = "date"
) -> List[Dict]:
    """Fetch recent uploads from a channel.  Costs 100+1 quota units."""
    key = _get_key()
    resp = requests.get(
        f"{API_BASE}/search",
        params={
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "maxResults": min(max_results, 50),
            "order": order,
            "key": key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    video_ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
    if not video_ids:
        return []
    return get_videos_by_id(video_ids)


def get_category_list(region_code: str = "US") -> List[Dict]:
    """List available video categories.  Costs 1 quota unit."""
    key = _get_key()
    resp = requests.get(
        f"{API_BASE}/videoCategories",
        params={
            "part": "snippet",
            "regionCode": region_code,
            "key": key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return [
        {"id": it["id"], "title": it["snippet"]["title"]}
        for it in resp.json().get("items", [])
    ]


# ── Internal helpers ────────────────────────────────────────────────────────


def _parse_video_items(items: list) -> List[Dict]:
    """Convert YouTube API video items into our standard format."""
    videos: List[Dict] = []
    for item in items:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        videos.append(
            {
                "video_id": item.get("id"),
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "channel_id": snippet.get("channelId"),
                "category_id": snippet.get("categoryId"),
                "description": snippet.get("description", "")[:500],
                "tags": snippet.get("tags", []),
                "published_at": snippet.get("publishedAt"),
                "thumbnail": snippet.get("thumbnails", {})
                .get("high", {})
                .get("url"),
                "duration": _parse_duration(content.get("duration", "")),
                "duration_iso": content.get("duration"),
                "views": _format_views(stats.get("viewCount")),
                "view_count_raw": int(stats.get("viewCount", 0) or 0),
                "likes": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
                "timestamp": datetime.now().isoformat(),
                "source": "youtube_api",
            }
        )
    return videos
