"""
Flask API server for screen recording and OCR service.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pymysql
from screen_recorder import YouTubeScreenRecorder
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
    Extract valid video titles directly from raw text_regions.
    This is a fallback when the main extraction produces garbage.
    """
    import re
    from datetime import datetime
    
    if not isinstance(video_data, dict):
        return []
    
    # Get text_regions from single video or multi-video
    text_regions = []
    if video_data.get('multi'):
        for v in video_data.get('videos', []):
            text_regions.extend(v.get('text_regions', []))
    else:
        text_regions = video_data.get('text_regions', [])
    
    if not text_regions:
        return []
    
    # YouTube UI elements to completely skip
    ui_skip_patterns = [
        r'^watch\s*later\s*\d*$', r'^liked\s*videos?\s*$', r'^your\s*videos?\s*$',
        r'^subscriptions?$', r'^history$', r'^library$', r'^home$',
        r'^\d+\s*(views?|watching)$', r'^\d+[KMB]?\s*views?$',
    ]
    
    # Prefixes to strip from the beginning of text
    ui_prefixes = [
        r'^\d+\)\s*liked\s*videos?\s*[&\s]*',  # "2) Liked videos &"
        r'^\d+\)\s*watch\s*later\s*[&\s]*',    # "1) Watch later &"
        r'^\d+\)\s*your\s*videos?\s*[&\s]*',   # "3) Your videos &"
        r'^watch\s*later\s*\d*\s*[&\s]*',      # "Watch later 49 &"
        r'^liked\s*videos?\s*[&\s]*',          # "Liked videos &"
        r'^your\s*videos?\s*[&\s]*',           # "Your videos &"
        r'^\[\]\s*videos?\s*',                 # "[] videos"
        r'^[\[\]\(\)\{\}<>&,\.\s\d]+(?=[A-Z])', # Leading garbage before capital letter
    ]
    
    # Navigation words
    nav_words = {'home', 'gaming', 'music', 'sports', 'shorts', 'subscriptions', 
                 'live', 'trending', 'explore', 'library', 'history'}
    
    videos = []
    seen_titles = set()
    
    for region in text_regions:
        text = region.get('text', '')
        conf = region.get('conf', 0)
        y = region.get('y', 0)
        
        # Lower confidence threshold to get more videos
        if conf < 75:
            continue
        
        # Skip regions at very top (likely nav)
        if y < 80:
            continue
        
        # Split by colons to separate merged titles
        parts = re.split(r'\s*:\s*', text)
        
        for part in parts:
            part = part.strip()
            
            # Clean up UI prefixes first
            for prefix_pattern in ui_prefixes:
                part = re.sub(prefix_pattern, '', part, flags=re.I)
            part = part.strip()
            
            # Clean up other garbage prefixes
            part = re.sub(r'^[\[\]\(\)\{\}<>&,\.\s]+', '', part)
            part = re.sub(r'^(soc|sug|via|anNYC|Ms)\s*[,\s]+', '', part, flags=re.I)
            part = part.strip()
            
            # Skip too short or too long
            if len(part) < 10 or len(part) > 80:
                continue
            
            # Skip YouTube UI elements (exact matches)
            skip = False
            for pattern in ui_skip_patterns:
                if re.match(pattern, part, re.I):
                    skip = True
                    break
            if skip:
                continue
            
            # Skip if starts with @ (channel name)
            if part.startswith('@'):
                continue
            
            # Skip if looks like view count or timestamp embedded
            if re.search(r'\d+[KMB]?\s*views?\s*\d+\s*(months?|years?|days?|hours?)\s*ago', part, re.I):
                continue
            if re.match(r'^\d+:\d+', part):
                continue
            
            # Skip if contains too many nav words
            words_lower = part.lower().split()
            nav_count = sum(1 for w in words_lower if w in nav_words)
            if nav_count >= 3:  # Relaxed from 2 to 3
                continue
            
            # Skip if mostly channel names (multiple @)
            if part.count('@') >= 2:
                continue
            
            # Skip garbage patterns
            words = part.split()
            if len(words) < 2:  # Relaxed from 3 to 2
                continue
            
            # Count meaningful words (3+ chars, alphanumeric)
            meaningful = [w for w in words if len(w) >= 3 and any(c.isalnum() for c in w)]
            if len(meaningful) < 2:
                continue
            
            # Check for too many single-char words
            single_char = sum(1 for w in words if len(w) == 1)
            if len(words) > 0 and single_char / len(words) > 0.35:  # Relaxed from 0.25
                continue
            
            # Check alpha ratio - must have decent text
            alpha_count = sum(1 for c in part if c.isalpha())
            if len(part) > 0 and alpha_count / len(part) < 0.45:  # Relaxed from 0.55
                continue
            
            # Normalize for dedup
            norm = re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9 ]', '', part)).strip().lower()
            if not norm or norm in seen_titles:
                continue
            if len(norm) < 8:  # Relaxed from 10
                continue
            
            seen_titles.add(norm)
            
            video = {
                'title': part,
                'channel': None,
                'views': None,
                'duration': None,
                'timestamp': datetime.now().isoformat(),
                'source': 'text_region_extraction'
            }
            videos.append(video)
            print(f"[TEXT_REGION] Extracted title: {part[:50]}")
    
    return videos[:10]  # Limit to 10 videos per frame


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
    return jsonify({
        'status': 'ok', 
        'service': 'ocr-screen-recorder',
        'display': display_status,
        'headless_mode': not recorder.recorder.display_available and ALLOW_HEADLESS_MODE
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


@app.route('/api/upload-frame', methods=['POST'])
def upload_frame():
    """Accept an uploaded frame (multipart file) or a base64 image in JSON, run OCR, and save results."""
    try:
        # Accept either multipart/form-data file upload with key 'file'
        # or JSON { "image": "data:image/png;base64,..." }
        if 'file' in request.files:
            file_bytes = request.files['file'].read()
        else:
            data = request.get_json(force=True, silent=True) or {}
            img_b64 = data.get('image')
            if not img_b64:
                return jsonify({'error': 'No image provided'}), 400
            # If data URL prefix is included, strip it
            if img_b64.startswith('data:'):
                img_b64 = img_b64.split(',', 1)[1]
            try:
                file_bytes = base64.b64decode(img_b64)
            except Exception:
                return jsonify({'error': 'Invalid base64 image'}), 400

        # Decode image bytes to OpenCV BGR image
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Use existing extractor to get video data
        video_data = recorder.extractor.extract_video_data(frame)
        print(f"[UPLOAD_FRAME] Extracted video_data: {video_data}")

        # NEW: Try to extract valid titles from raw text_regions if main extraction fails
        extracted_videos = extract_videos_from_text_regions(video_data)
        if extracted_videos:
            saved_any = 0
            for v in extracted_videos:
                ok = save_video_data(v)
                print(f"[UPLOAD_FRAME] save_video_data returned: {ok} for title={v.get('title')}")
                if ok:
                    saved_any += 1
            if saved_any > 0:
                return jsonify({'status': 'success', 'videos_saved': saved_any, 'video_data': {'extracted_from_regions': True, 'count': saved_any}})

        # Support multi-video payloads
        try:
            if isinstance(video_data, dict) and video_data.get('multi'):
                saved_any = 0
                for v in video_data.get('videos', []):
                    ok = save_video_data(v)
                    print(f"[UPLOAD_FRAME] save_video_data returned: {ok} for title={v.get('title')}")
                    if ok:
                        saved_any += 1
                return jsonify({'status': 'success', 'videos_saved': saved_any, 'video_data': video_data})

            if video_data and video_data.get('title'):
                saved = save_video_data(video_data)
                print(f"[UPLOAD_FRAME] save_video_data returned: {saved}")
                if not saved:
                    return jsonify({'status': 'error', 'message': 'Failed to save extracted video data', 'video_data': video_data}), 500
                return jsonify({'status': 'success', 'video_data': video_data})

            print(f"[UPLOAD_FRAME] No title found in extracted data. video_data keys: {list(video_data.keys()) if isinstance(video_data, dict) else 'N/A'}")
            return jsonify({
                'status': 'no_data',
                'video_data': video_data,
                'message': 'No video data extracted from uploaded frame'
            })
        except Exception as e:
            print(f"[UPLOAD_FRAME] Error saving data: {e}")
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"Upload frame error: {e}")
        return jsonify({'error': str(e)}), 500


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


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting OCR service on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)

