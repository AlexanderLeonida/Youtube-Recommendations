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


def save_video_data(video_data: dict):
    """Save extracted video data to database."""
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
    global recording_active
    
    if not recording_active:
        return jsonify({'error': 'No recording in progress'}), 400
    
    recorder.recorder.stop_recording()
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

