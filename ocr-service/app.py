"""
Flask API server for screen recording and OCR service.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pymysql
from screen_recorder import YouTubeScreenRecorder
from llm_parser import parse_with_llm, check_llm_health
from youtube_scraper import scrape_youtube_homepage
import threading
import time
from datetime import datetime
import random
import base64
import numpy as np
import cv2
from io import BytesIO
from typing import List, Dict

load_dotenv()

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'mysql'),
    'user': os.getenv('DB_USER', 'ytrecs'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'database': os.getenv('DB_NAME', 'ytrecs'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# Global recorder instance
recorder = YouTubeScreenRecorder()
recording_active = False
recording_thread = None
ALLOW_HEADLESS_MODE = os.getenv('ALLOW_HEADLESS_MODE', 'true').lower() == 'true'

PLACEHOLDER_VIDEOS = [
    {
        'title': 'Understanding the YouTube Algorithm (2025)',
        'channel': 'Creator Insider',
        'views': '2.1M views',
        'duration': '14:32'
    },
    {
        'title': '10 Underrated Tech Channels You Should Watch',
        'channel': 'ByteReview',
        'views': '857K views',
        'duration': '11:08'
    },
    {
        'title': 'Can AI Really Pick Your Next Favorite Video?',
        'channel': 'FutureFrames',
        'views': '642K views',
        'duration': '9:47'
    },
    {
        'title': 'React vs Svelte: Real-World Performance',
        'channel': 'CodeCafe',
        'views': '1.4M views',
        'duration': '13:21'
    },
    {
        'title': 'Deep Dive: Transformers Explained Visually',
        'channel': 'NeuralNerd',
        'views': '3.2M views',
        'duration': '18:59'
    }
]


def generate_placeholder_video_data(count: int = 3):
    """Generate placeholder video data when display is unavailable."""
    selected = random.sample(PLACEHOLDER_VIDEOS, k=min(count, len(PLACEHOLDER_VIDEOS)))
    results = []
    for item in selected:
        video = {
            'title': item['title'],
            'channel': item['channel'],
            'views': item['views'],
            'duration': item['duration'],
            'timestamp': datetime.utcnow().isoformat(),
            'raw_text_regions': 0,
            'source': 'headless-mode'
        }
        results.append(video)
    return results


def get_db_connection():
    """Get database connection."""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def extract_videos_from_text_regions(video_data: dict) -> List[dict]:
    """
    Extract structured video data (title, channel, views, duration) from raw
    text_regions by spatially grouping lines into video-card clusters.

    Approach:
      1. Collect all text_regions (with x, y, w, h, text, conf).
      2. Sort by y and cluster into card-groups using vertical gaps.
      3. Within each group, classify lines as title / channel / metadata.
    """
    import re
    from datetime import datetime

    if not isinstance(video_data, dict):
        return []

    # Gather text_regions from single or multi-video payloads
    text_regions = []
    if video_data.get('multi'):
        for v in video_data.get('videos', []):
            text_regions.extend(v.get('text_regions', []))
    else:
        text_regions = video_data.get('text_regions', [])

    if not text_regions:
        return []

    # ---- Patterns ----
    view_pat   = re.compile(r'[\d,\.]+\s*[KMBkmb]?\s*views?', re.I)
    dur_pat    = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')
    ago_pat    = re.compile(r'\d+\s*(?:second|minute|hour|day|week|month|year)s?\s*ago', re.I)
    meta_pat   = re.compile(r'(?:views?|watching|subscribers?|streamed|premiered|ago\b|Try\s*now|featured|PG-13|RATED|Sign\s*in)', re.I)
    channel_at = re.compile(r'@\w+')
    nav_words  = {'home', 'gaming', 'music', 'sports', 'shorts', 'subscriptions',
                  'live', 'trending', 'explore', 'library', 'history'}

    # ---- 1. Filter low-quality / navigation regions ----
    filtered = []
    for r in text_regions:
        text = r.get('text', '').strip()
        conf = r.get('conf', 0)
        if not text or len(text) < 2:
            continue
        if conf < 40:
            continue
        # Skip navigation-bar text (many nav keywords)
        words_lower = text.lower().split()
        if sum(1 for w in words_lower if w in nav_words) >= 3:
            continue
        filtered.append(r)

    if not filtered:
        return []

    # ---- 2. Group into card clusters: COLUMN then ROW ----
    # YouTube's homepage is a grid (typically 3-4 columns).  Regions in the
    # same row but different columns have similar Y but very different X.
    # Step 2a: Assign each region to a column by clustering X positions.
    if not filtered:
        return []

    # Determine column boundaries by looking at X centre-points
    x_centres = sorted(set(r['x'] + r.get('w', 0) // 2 for r in filtered))
    # Simple column clustering: walk through sorted X centres and split when
    # the gap exceeds a threshold (e.g. 25% of image width, or 200px minimum)
    max_x = max(r['x'] + r.get('w', 0) for r in filtered) if filtered else 1
    col_gap = max(200, int(max_x * 0.20))

    col_boundaries = [x_centres[0]]
    for i in range(1, len(x_centres)):
        if x_centres[i] - x_centres[i - 1] > col_gap:
            col_boundaries.append(x_centres[i])

    def _assign_column(r):
        cx = r['x'] + r.get('w', 0) // 2
        best_col = 0
        best_dist = abs(cx - col_boundaries[0])
        for ci, cb in enumerate(col_boundaries):
            d = abs(cx - cb)
            if d < best_dist:
                best_dist = d
                best_col = ci
        return best_col

    # Step 2b: Within each column, group by vertical gaps into card clusters
    from collections import defaultdict
    col_regions = defaultdict(list)
    for r in filtered:
        col_regions[_assign_column(r)].append(r)

    groups: List[List[dict]] = []
    for col_idx in sorted(col_regions.keys()):
        col = sorted(col_regions[col_idx], key=lambda r: r['y'])
        if not col:
            continue
        heights = [r.get('h', 20) for r in col]
        avg_h = sum(heights) / len(heights) if heights else 20
        gap_thresh = max(40, int(avg_h * 2.0))

        current_group = [col[0]]
        for prev_r, curr_r in zip(col, col[1:]):
            gap = curr_r['y'] - (prev_r['y'] + prev_r.get('h', 0))
            if gap > gap_thresh:
                groups.append(current_group)
                current_group = [curr_r]
            else:
                current_group.append(curr_r)
        if current_group:
            groups.append(current_group)

    # ---- 3. Within each group, extract card fields ----
    videos = []
    seen_titles: set = set()

    for grp in groups:
        title = None
        channel = None
        views = None
        duration = None
        title_candidates = []

        for r in sorted(grp, key=lambda r: r['y']):
            text = r.get('text', '').strip()
            if not text or len(text) < 2:
                continue

            # Duration
            dm = dur_pat.search(text)
            if dm and not duration:
                duration = dm.group(0)

            # Views
            vm = view_pat.search(text)
            if vm:
                if not views:
                    views = vm.group(0)
                continue

            # Channel handle
            if channel_at.search(text) and not channel:
                channel = text
                continue

            # Metadata line (views, ago, featured, etc.)
            if meta_pat.search(text) or ago_pat.search(text):
                if not views:
                    vm2 = view_pat.search(text)
                    if vm2:
                        views = vm2.group(0)
                continue

            # Skip very short fragments or garbage
            words = text.split()
            meaningful = [w for w in words if len(w) >= 3 and any(c.isalnum() for c in w)]
            if len(meaningful) < 2:
                continue
            alpha_count = sum(1 for c in text if c.isalpha())
            if len(text) > 0 and alpha_count / len(text) < 0.45:
                continue

            title_candidates.append(r)

        # Pick best title (topmost, merge adjacent lines)
        if title_candidates:
            title_candidates.sort(key=lambda r: r['y'])
            title = title_candidates[0].get('text', '').strip()
            if len(title_candidates) > 1:
                first_bottom = title_candidates[0]['y'] + title_candidates[0].get('h', 20)
                second = title_candidates[1]
                if second['y'] - first_bottom < 30:
                    title = title + ' ' + second.get('text', '').strip()

        # Infer channel from first short text below title
        if not channel and title and title_candidates:
            t_bottom = title_candidates[0]['y'] + title_candidates[0].get('h', 20)
            for r in sorted(grp, key=lambda r: r['y']):
                ry = r.get('y', 0)
                txt = r.get('text', '').strip()
                if ry > t_bottom and ry < t_bottom + 60 and 2 < len(txt) < 50:
                    if not view_pat.search(txt) and not ago_pat.search(txt) and not meta_pat.search(txt):
                        channel = txt
                        break

        if not title:
            continue

        # De-duplicate
        norm = re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9 ]', '', title)).strip().lower()
        if not norm or norm in seen_titles or len(norm) < 8:
            continue
        seen_titles.add(norm)

        video = {
            'title': title,
            'channel': channel,
            'views': views,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'source': 'text_region_extraction'
        }
        videos.append(video)
        print(f"[TEXT_REGION] Extracted: title='{title[:50]}' ch={channel} views={views} dur={duration}")

    return videos[:10]


def is_valid_video_data(video_data: dict) -> bool:
    """Validate if extracted video data looks reasonable."""
    import re
    
    title = video_data.get('title', '')
    if not title or len(title) < 5:
        print(f"[VALIDATION] Rejected: too short - {title[:50]}")
        return False
    
    # Title should have at least 2 actual words (3+ chars)
    words = title.split()
    meaningful_words = [w for w in words if len(w) >= 3 and any(c.isalnum() for c in w)]
    if len(meaningful_words) < 2:
        print(f"[VALIDATION] Rejected: needs 2+ meaningful words - {title[:50]}")
        return False
    
    # CRITICAL: Reject YouTube navigation bar text
    nav_categories = ['home', 'gaming', 'music', 'sports', 'news', 'live', 'shorts', 
                      'subscriptions', 'library', 'history', 'trending', 'explore',
                      'slam dunks', 'golden state', 'italian cuisine', 'speedruns',
                      'gordon ramsay', 'mixes', 'among us', 'warriors', 'slam dunk']
    title_lower = title.lower()
    nav_matches = sum(1 for cat in nav_categories if cat in title_lower)
    if nav_matches >= 3:
        print(f"[VALIDATION] Rejected: looks like nav bar ({nav_matches} category matches) - {title[:50]}")
        return False
    
    # Reject text starting with common nav patterns
    if title_lower.startswith(('home ', 'home gaming', 'gaming music', 'shorts ', 'subscriptions')):
        print(f"[VALIDATION] Rejected: starts with nav pattern - {title[:50]}")
        return False
    
    # Too many special characters
    alpha_count = sum(1 for c in title if c.isalnum())
    special_count = sum(1 for c in title if not c.isalnum() and not c.isspace())
    if alpha_count > 0 and special_count / alpha_count > 0.25:
        print(f"[VALIDATION] Rejected: too many special chars ({special_count}/{alpha_count}) - {title[:50]}")
        return False
    
    # Too many single-char words
    single_char_words = sum(1 for w in words if len(w) == 1)
    if len(words) > 0 and single_char_words / len(words) > 0.3:
        print(f"[VALIDATION] Rejected: too many single-char words - {title[:50]}")
        return False
    
    # Average word length too short (random char soup)
    if len(words) >= 3:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2.5:
            print(f"[VALIDATION] Rejected: avg word len {avg_word_len:.1f} too short - {title[:50]}")
            return False
    
    # Contains timestamp patterns (likely merged OCR)
    if re.search(r'\d+:\d+\s+\d+:\d+', title):
        print(f"[VALIDATION] Rejected: contains timestamp pattern - {title[:50]}")
        return False
    
    # Reject pure random patterns like "ae en y 20" or "TT fi Tsk wD"
    # These have too many 2-char "words"
    short_words = sum(1 for w in words if len(w) <= 2)
    if len(words) >= 4 and short_words / len(words) > 0.5:
        print(f"[VALIDATION] Rejected: too many short words - {title[:50]}")
        return False
    
    # Reject if most characters are non-ASCII or symbols
    ascii_alpha = sum(1 for c in title if c.isascii() and c.isalpha())
    if len(title) > 0 and ascii_alpha / len(title) < 0.4:
        print(f"[VALIDATION] Rejected: low ASCII alpha ratio - {title[:50]}")
        return False
    
    print(f"[VALIDATION] Accepted: {title[:50]}")
    return True


def save_video_data(video_data: dict):
    """Save extracted video data to database."""
    # Validate video data first
    if not is_valid_video_data(video_data):
        print(f"[SAVE] Skipping invalid video data: {video_data.get('title', 'N/A')[:50]}")
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Insert or update video data
        query = """
        INSERT INTO videos (title, channel_name, view_count, duration, extracted_at, raw_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            view_count = VALUES(view_count),
            duration = VALUES(duration),
            extracted_at = VALUES(extracted_at),
            raw_data = VALUES(raw_data)
        """
        
        import json
        cursor.execute(query, (
            video_data.get('title'),
            video_data.get('channel'),
            video_data.get('views'),
            video_data.get('duration'),
            video_data.get('timestamp'),
            json.dumps(video_data)
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving video data: {e}")
        if conn:
            conn.close()
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    display_status = 'available' if recorder.recorder.display_available else 'unavailable'
    llm_info = check_llm_health()
    return jsonify({
        'status': 'ok', 
        'service': 'ocr-screen-recorder',
        'display': display_status,
        'headless_mode': not recorder.recorder.display_available and ALLOW_HEADLESS_MODE,
        'llm': llm_info,
    })


@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    """Start screen recording and OCR extraction."""
    global recording_active, recording_thread
    
    if recording_active:
        return jsonify({'error': 'Recording already in progress'}), 400
    
    data = request.get_json() or {}
    duration = data.get('duration', 30)  # Default 30 seconds
    frame_interval = data.get('frame_interval', 2.0)  # Process every 2 seconds

    # If display is unavailable but headless mode is allowed, return placeholder data
    if not recorder.recorder.display_available and ALLOW_HEADLESS_MODE:
        simulated_results = generate_placeholder_video_data()
        for video_data in simulated_results:
            save_video_data(video_data)
        return jsonify({
            'status': 'headless', 
            'message': 'Display not available. Generated placeholder video data.',
            'videos_generated': len(simulated_results),
            'headless_mode': True
        })
    elif not recorder.recorder.display_available:
        return jsonify({
            'error': 'Display not available',
            'message': 'Screen recording requires display access. In Docker, this may require X11 forwarding or running on the host machine.'
        }), 503
    
    def record_worker():
        global recording_active
        recording_active = True
        try:
            results = recorder.record_and_extract(
                duration_seconds=duration,
                frame_interval=frame_interval
            )
            
            # Save all extracted videos to database
            for video_data in results:
                save_video_data(video_data)
            
            print(f"Recording completed. Extracted {len(results)} videos.")
        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            recording_active = False
    
    recording_thread = threading.Thread(target=record_worker, daemon=True)
    recording_thread.start()
    
    return jsonify({
        'status': 'started',
        'duration': duration,
        'message': 'Screen recording started'
    })


@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    """Stop screen recording."""
    global recording_active, recording_thread
    
    # Always stop the recorder regardless of flag state
    # This allows stopping even if background thread has already finished
    try:
        recorder.recorder.stop_recording()
    except Exception as e:
        print(f"Error stopping recording: {e}")
    
    recording_active = False
    
    return jsonify({'status': 'stopped', 'message': 'Screen recording stopped'})


@app.route('/api/capture-frame', methods=['POST'])
def capture_frame():
    """Capture a single frame and extract video data."""
    try:
        if not recorder.recorder.display_available:
            if ALLOW_HEADLESS_MODE:
                video_data = generate_placeholder_video_data(count=1)[0]
                save_video_data(video_data)
                return jsonify({
                    'status': 'success',
                    'video_data': video_data,
                    'headless_mode': True,
                    'message': 'Display not available. Generated placeholder video data.'
                })
            return jsonify({
                'status': 'error',
                'error': 'Display not available',
                'message': 'Screen capture requires display access. In Docker, this may require X11 forwarding or running on the host machine.'
            }), 503
        
        video_data = recorder.capture_single_frame_and_extract()
        
        if video_data and video_data.get('title'):
            # Save to database
            save_video_data(video_data)
            return jsonify({
                'status': 'success',
                'video_data': video_data
            })
        else:
            return jsonify({
                'status': 'no_data',
                'message': 'No video data extracted from frame'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _llm_background_worker(text_regions):
    """Run LLM structuring in a background thread so upload-frame returns fast."""
    try:
        llm_videos = parse_with_llm(text_regions)
        for v in llm_videos:
            vid = {
                'title': v.get('title'),
                'channel': v.get('channel'),
                'views': v.get('views'),
                'duration': v.get('duration'),
                'posted_ago': v.get('posted_ago'),
                'timestamp': datetime.now().isoformat(),
                'source': 'llm_structured',
            }
            ok = save_video_data(vid)
            print(f"[LLM_BG] save={ok} → {vid.get('title', '')[:50]}")
    except Exception as e:
        print(f"[LLM_BG] Error: {e}")


@app.route('/api/upload-frame', methods=['POST'])
def upload_frame():
    """
    Accept an uploaded frame and extract video data.

    Speed optimisations vs the original implementation:
      - Uses fast=True OCR (single scale) by default — halves Tesseract time.
      - Returns heuristic results immediately instead of waiting for LLM.
      - Kicks off LLM structuring in a background thread (results appear on
        next frontend refresh, not blocking the current request).

    Pass ?detailed=true to force multi-scale OCR + synchronous LLM.
    """
    try:
        # ── Decode image ──────────────────────────────────────────
        if 'file' in request.files:
            file_bytes = request.files['file'].read()
        else:
            data = request.get_json(force=True, silent=True) or {}
            img_b64 = data.get('image')
            if not img_b64:
                return jsonify({'error': 'No image provided'}), 400
            if img_b64.startswith('data:'):
                img_b64 = img_b64.split(',', 1)[1]
            try:
                file_bytes = base64.b64decode(img_b64)
            except Exception:
                return jsonify({'error': 'Invalid base64 image'}), 400

        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        detailed = request.args.get('detailed', 'false').lower() == 'true'
        fast_ocr = not detailed

        # ── Step 1: Run OCR (fast mode by default) ────────────────
        video_data = recorder.extractor.extract_video_data(frame, fast=fast_ocr)
        print(f"[UPLOAD_FRAME] OCR returned type={type(video_data).__name__} fast={fast_ocr}")

        text_regions = []
        if isinstance(video_data, dict):
            if video_data.get('multi'):
                for v in video_data.get('videos', []):
                    text_regions.extend(v.get('text_regions', []))
            else:
                text_regions = video_data.get('text_regions', [])

        # ── Step 2: Heuristic extraction (instant) ────────────────
        extracted_videos = extract_videos_from_text_regions(video_data)
        saved_count = 0
        if extracted_videos:
            for v in extracted_videos:
                if save_video_data(v):
                    saved_count += 1

        # ── Step 3: Kick off LLM structuring in background ────────
        if text_regions:
            if detailed:
                # Synchronous LLM for detailed mode
                llm_videos = parse_with_llm(text_regions)
                for v in llm_videos:
                    vid = {
                        'title': v.get('title'),
                        'channel': v.get('channel'),
                        'views': v.get('views'),
                        'duration': v.get('duration'),
                        'posted_ago': v.get('posted_ago'),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'llm_structured',
                    }
                    if save_video_data(vid):
                        saved_count += 1
            else:
                # Background LLM — don't block the response
                threading.Thread(
                    target=_llm_background_worker,
                    args=(text_regions,),
                    daemon=True,
                ).start()

        if saved_count > 0:
            return jsonify({
                'status': 'success',
                'videos_saved': saved_count,
                'source': 'heuristic',
                'llm_pending': not detailed and bool(text_regions),
                'video_data': {'extracted_from_regions': True, 'count': saved_count},
            })

        # ── Step 4: Single-video fallback ─────────────────────────
        if isinstance(video_data, dict) and video_data.get('title'):
            if save_video_data(video_data):
                return jsonify({'status': 'success', 'source': 'single', 'video_data': video_data})

        return jsonify({
            'status': 'no_data',
            'message': 'No video data extracted from uploaded frame',
            'regions_found': len(text_regions),
            'llm_pending': not detailed and bool(text_regions),
        })
    except Exception as e:
        print(f"Upload frame error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/scrape-youtube', methods=['POST'])
def scrape_youtube():
    """
    Fast path: extract video titles directly from YouTube's HTML DOM using
    BeautifulSoup.  Returns results in <1s vs 5-10s for full OCR + LLM.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        cookies = data.get('cookies')  # optional browser cookies for personalized recs

        videos = scrape_youtube_homepage(cookies=cookies)
        if not videos:
            return jsonify({'status': 'no_data', 'message': 'Could not scrape YouTube homepage'})

        saved_count = 0
        for v in videos:
            vid = {
                'title': v.get('title'),
                'channel': v.get('channel'),
                'views': v.get('views'),
                'duration': v.get('duration'),
                'posted_ago': v.get('posted_ago'),
                'timestamp': v.get('timestamp', datetime.now().isoformat()),
                'source': 'html_scraper',
            }
            if save_video_data(vid):
                saved_count += 1

        return jsonify({
            'status': 'success',
            'videos_saved': saved_count,
            'videos_found': len(videos),
            'source': 'html_scraper',
        })
    except Exception as e:
        print(f"[SCRAPER] Endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-status', methods=['GET'])
def llm_status():
    """Check local Llama model availability."""
    return jsonify(check_llm_health())


@app.route('/api/recording-status', methods=['GET'])
def recording_status():
    """Get current recording status."""
    return jsonify({
        'recording': recording_active,
        'status': 'active' if recording_active else 'inactive',
        'headless_mode': not recorder.recorder.display_available and ALLOW_HEADLESS_MODE
    })


@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get all extracted videos from database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = "SELECT * FROM videos ORDER BY extracted_at DESC LIMIT 100"
        cursor.execute(query)
        videos = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({'videos': videos, 'count': len(videos)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a specific video by ID."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM videos WHERE id = %s", (video_id,))
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        if affected == 0:
            return jsonify({'error': 'Video not found'}), 404
        return jsonify({'status': 'deleted', 'id': video_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting OCR service on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)

