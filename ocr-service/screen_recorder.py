"""
Screen recording service with OCR pipeline for extracting YouTube video data.
"""
import cv2
import numpy as np
import pytesseract
from mss import mss
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
import re


class ScreenRecorder:
    def __init__(self, monitor_number: int = 1):
        """
        Initialize screen recorder.
        
        Args:
            monitor_number: Monitor to record (1 = primary monitor)
        """
        try:
            self.sct = mss()
            self.monitor = self.sct.monitors[monitor_number]
            self.is_recording = False
            self.frames = []
            self.display_available = True
        except Exception as e:
            print(f"Warning: Could not initialize screen capture: {e}")
            print("Screen recording may not work in this environment (e.g., Docker without display access)")
            self.sct = None
            self.monitor = None
            self.is_recording = False
            self.frames = []
            self.display_available = False
        
    def start_recording(self):
        """Start recording screen."""
        self.is_recording = True
        self.frames = []
        print("Screen recording started...")
        
    def stop_recording(self):
        """Stop recording screen."""
        self.is_recording = False
        print("Screen recording stopped.")
        
    def capture_frame(self, require_recording: bool = True) -> Optional[np.ndarray]:
        """
        Capture a single frame from the screen.
        
        Args:
            require_recording: If True, only capture when recording is active.
                              If False, capture regardless of recording state.
        """
        if not self.display_available:
            return None
        
        if require_recording and not self.is_recording:
            return None
        
        try:
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def record_frame(self):
        """Record current frame."""
        frame = self.capture_frame()
        if frame is not None:
            self.frames.append(frame)
        return frame


class YouTubeVideoExtractor:
    """Extract video information from YouTube screenshots using OCR."""
    
    def __init__(self):
        # Configure Tesseract (adjust path if needed)
        # For Linux/Docker, tesseract should be in PATH
        # For macOS, might need: pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed grayscale image
        """
        # Simpler pipeline that previously gave better character recognition
        height, width = image.shape[:2]
        # Upscale modestly for small inputs
        if max(height, width) < 1500:
            image = cv2.resize(image, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding to separate text from background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply denoising (fastNlMeans) as earlier version did
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def _preprocess_for_ocr_variants(self, image: np.ndarray) -> list:
        """Return multiple preprocessing variants (normal + inverted) to improve OCR recall."""
        # Create base preprocessing (as single-channel image)
        base = self.preprocess_image(image)
        inv = cv2.bitwise_not(base)

        # Heuristic: detect if image is dark overall (white text on black)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
        except Exception:
            mean_brightness = 255.0

        # If dark background, prefer the inverted preprocessing first
        if mean_brightness < 120:
            return [inv, base]
        return [base, inv]
    
    def extract_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text regions from image using OCR.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of text regions with bounding boxes
        """
        # Run OCR on multiple preprocessing variants to improve recall
        text_regions: List[Dict] = []
        variants = self._preprocess_for_ocr_variants(image)
        seen_boxes = set()

        # Use single reliable PSM that worked best previously (PSM 6)
        psm = 6
        for var in variants:
            try:
                config = f'--oem 3 --psm {psm}'
                ocr_data = pytesseract.image_to_data(var, output_type=pytesseract.Output.DICT, config=config)
            except Exception as e:
                print(f"pytesseract error (psm={psm}): {e}")
                continue

            n_boxes = len(ocr_data.get('text', []))
            for i in range(n_boxes):
                text = (ocr_data.get('text', [''])[i] or '').strip()
                conf_raw = ocr_data.get('conf', [''])[i]
                try:
                    conf = float(conf_raw)
                except Exception:
                    try:
                        conf = float(str(conf_raw).strip())
                    except Exception:
                        conf = -1.0

                if not text:
                    continue

                # Restore higher confidence threshold to avoid garbage
                if conf < 30:
                    continue

                x = int(ocr_data.get('left', [0])[i])
                y = int(ocr_data.get('top', [0])[i])
                w = int(ocr_data.get('width', [0])[i])
                h = int(ocr_data.get('height', [0])[i])

                key = (x, y, w, h, text)
                if key in seen_boxes:
                    continue
                seen_boxes.add(key)

                text_regions.append({
                    'text': text,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'conf': conf
                })

        return text_regions

    def _ocr_region_string(self, image: np.ndarray, region: tuple) -> str:
        """Crop a region and run tesseract OCR returning the combined text."""
        x, y, w, h = region
        h_img, w_img = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            return ''

        # Run on preprocessing variants and combine
        variants = self._preprocess_for_ocr_variants(crop)
        texts = []
        for var in variants:
            try:
                txt = pytesseract.image_to_string(var, config='--oem 3 --psm 6')
                if txt:
                    texts.append(txt.strip())
            except Exception:
                continue

        return '\n'.join(t for t in texts if t)
    
    def extract_video_title(self, text_regions: List[Dict], image: np.ndarray) -> Optional[str]:
        """
        Extract video title from text regions.
        YouTube titles are typically at the top-center of the page.
        
        Args:
            text_regions: List of extracted text regions
            image_shape: (height, width) of the image
            
        Returns:
            Extracted video title or None
        """
        height, width = image.shape[:2]
        
        # First try combining OCR text regions in top-center area
        top_region_y = int(height * 0.35)
        center_x_start = int(width * 0.1)
        center_x_end = int(width * 0.9)

        title_candidates = []
        for region in text_regions:
            if (region['y'] < top_region_y and
                center_x_start < region['x'] < center_x_end and
                len(region['text']) > 3):
                title_candidates.append(region)

        title_candidates.sort(key=lambda x: (x['y'], -x['conf']))

        if title_candidates:
            title_parts = []
            last_y = -1
            for candidate in title_candidates:
                if last_y == -1 or abs(candidate['y'] - last_y) < 60:
                    title_parts.append(candidate['text'])
                    last_y = candidate['y']
            if title_parts:
                return ' '.join(title_parts)

        # Fallback: run OCR on the top area crop (more aggressive)
        top_crop_h = int(height * 0.25)
        top_crop = (int(width * 0.05), 0, int(width * 0.9), top_crop_h)
        top_text = self._ocr_region_string(image, top_crop)
        if top_text:
            # Heuristic: choose the longest reasonable line as title
            lines = [l.strip() for l in top_text.splitlines() if len(l.strip()) > 3]
            if lines:
                lines.sort(key=lambda s: -len(s))
                return lines[0]

        return None
    
    def extract_channel_name(self, text_regions: List[Dict], image: np.ndarray) -> Optional[str]:
        """
        Extract channel name from text regions.
        Channel names are typically below the title.
        
        Args:
            text_regions: List of extracted text regions
            image_shape: (height, width) of the image
            
        Returns:
            Extracted channel name or None
        """
        height, width = image.shape[:2]
        
        # Look for text below title area (20-45% from top)
        channel_region_y_start = int(height * 0.18)
        channel_region_y_end = int(height * 0.5)
        center_x_start = int(width * 0.05)
        center_x_end = int(width * 0.95)

        channel_candidates = []
        for region in text_regions:
            if (channel_region_y_start < region['y'] < channel_region_y_end and
                center_x_start < region['x'] < center_x_end):
                # Skip obvious UI labels
                txt = region['text']
                if any(k.lower() in txt.lower() for k in ['subscribe', 'views', 'watch']):
                    continue
                channel_candidates.append(region)

        channel_candidates.sort(key=lambda x: (x['y'], -x['conf']))

        if channel_candidates:
            return channel_candidates[0]['text']

        # Fallback: try OCR on a region below the title area where channel usually appears
        channel_crop_y = int(height * 0.18)
        channel_crop_h = int(height * 0.15)
        channel_crop = (int(width * 0.05), channel_crop_y, int(width * 0.6), channel_crop_h)
        ch_text = self._ocr_region_string(image, channel_crop)
        if ch_text:
            # pick first non-empty line
            for line in ch_text.splitlines():
                line = line.strip()
                if line and len(line) < 80:
                    if not any(k.lower() in line.lower() for k in ['subscribe', 'views', 'watch']):
                        return line

        return None
    
    def extract_view_count(self, text_regions: List[Dict], image: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Extract view count from text regions.
        
        Args:
            text_regions: List of extracted text regions
            
        Returns:
            Extracted view count or None
        """
        # Look for patterns like "1.2M views", "123K views", "1,234 views"
        view_patterns = [
            r'[\d,\.]+[KMkm]?\s*views?',
            r'[\d,\.]+\s*views?',
        ]
        
        # Search region texts first
        for region in text_regions:
            text = region['text']
            for pattern in view_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(0)

        # Fallback: search raw OCR of center-right area where views usually appear
        if image is not None:
            h, w = image.shape[:2]
            crop = (int(w * 0.6), int(h * 0.15), int(w * 0.35), int(h * 0.15))
            txt = self._ocr_region_string(image, crop)
            for pattern in view_patterns:
                m = re.search(pattern, txt or '', re.IGNORECASE)
                if m:
                    return m.group(0)

        return None
        
        return None
    
    def extract_video_duration(self, text_regions: List[Dict]) -> Optional[str]:
        """
        Extract video duration from text regions.
        
        Args:
            text_regions: List of extracted text regions
            
        Returns:
            Extracted duration (e.g., "10:30") or None
        """
        # Look for time patterns like "10:30", "1:23:45"
        time_pattern = r'\d{1,2}:\d{2}(?::\d{2})?'
        
        for region in text_regions:
            text = region['text']
            match = re.search(time_pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def extract_video_data(self, image: np.ndarray) -> Dict:
        """
        Extract all video data from a YouTube screenshot.
        
        Args:
            image: Screenshot image (BGR format)
            
        Returns:
            Dictionary with extracted video information
        """
        # Preprocess image (kept for possible future use) and extract text regions
        processed = self.preprocess_image(image)

        # Extract text regions from the original image to avoid double preprocessing
        # which can convert the image to a single-channel and break color conversions.
        text_regions = self.extract_text_regions(image)

        # Extract various video information (pass original image for fallbacks)
        video_data = {
            'title': self.extract_video_title(text_regions, image),
            'channel': self.extract_channel_name(text_regions, image),
            'views': self.extract_view_count(text_regions, image),
            'duration': self.extract_video_duration(text_regions),
            'timestamp': datetime.now().isoformat(),
            'raw_text_regions': len(text_regions)
        }
        # Include text regions (limited) to help debugging and downstream processing
        try:
            video_data['text_regions'] = text_regions[:50]
        except Exception:
            video_data['text_regions'] = []
        
        return video_data


class YouTubeScreenRecorder:
    """Main class that combines screen recording and OCR extraction."""
    
    def __init__(self):
        self.recorder = ScreenRecorder()
        self.extractor = YouTubeVideoExtractor()
        self.video_data_history = []
        
    def record_and_extract(self, duration_seconds: int = 5, 
                          frame_interval: float = 1.0) -> List[Dict]:
        """
        Record screen for specified duration and extract video data.
        
        Args:
            duration_seconds: How long to record
            frame_interval: Interval between frames to process (in seconds)
            
        Returns:
            List of extracted video data dictionaries
        """
        self.recorder.start_recording()
        self.video_data_history = []
        
        start_time = time.time()
        last_process_time = 0
        
        print(f"Recording for {duration_seconds} seconds...")
        
        while time.time() - start_time < duration_seconds:
            frame = self.recorder.record_frame()
            
            # Process frame at specified intervals
            current_time = time.time()
            if current_time - last_process_time >= frame_interval:
                if frame is not None:
                    try:
                        video_data = self.extractor.extract_video_data(frame)
                        if video_data['title']:  # Only save if we found a title
                            self.video_data_history.append(video_data)
                            print(f"Extracted: {video_data.get('title', 'N/A')}")
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                
                last_process_time = current_time
            
            time.sleep(0.1)  # Small delay to avoid excessive CPU usage
        
        self.recorder.stop_recording()
        
        # Return unique video data (deduplicate by title)
        unique_videos = {}
        for video in self.video_data_history:
            title = video.get('title')
            if title and title not in unique_videos:
                unique_videos[title] = video
        
        return list(unique_videos.values())
    
    def capture_single_frame_and_extract(self) -> Optional[Dict]:
        """
        Capture a single frame and extract video data.
        This method doesn't require recording to be active.
        
        Returns:
            Extracted video data or None
        """
        frame = self.recorder.capture_frame(require_recording=False)
        if frame is not None:
            try:
                return self.extractor.extract_video_data(frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                return None
        return None


if __name__ == "__main__":
    # Test the screen recorder
    recorder = YouTubeScreenRecorder()
    
    print("Starting 5-second recording test...")
    results = recorder.record_and_extract(duration_seconds=5, frame_interval=1.0)
    
    print(f"\nExtracted {len(results)} unique videos:")
    for i, video in enumerate(results, 1):
        print(f"\n{i}. {video}")

