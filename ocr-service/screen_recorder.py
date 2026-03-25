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
from collections import Counter
import os
import tempfile

try:
    from hybrid_ocr import hybrid_extract_text_regions, _cpp_available
    _HAS_HYBRID = _cpp_available()
except ImportError:
    _HAS_HYBRID = False


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
        # Stopwords to filter out (common UI labels, menu items, etc.)
        self.stopwords = {
            'search', 'create', 'upload', 'extensions', 'settings', 'history',
            'watch', 'subscribe', 'views', 'tab', 'home', 'explore', 'subscriptions',
            'library', 'your', 'channel', 'youtube', 'short', 'downloads', 'show more',
            'like', 'share', 'save', 'report', 'add', 'menu', 'sort', 'filter',
            'play', 'pause', 'fullscreen', 'settings', 'cc', 'audio', 'quality'
        }
        # Frame history for temporal aggregation
        self.frame_history = []
        self.max_history_frames = 5

        # Tunable parameters
        self.ocr_conf_threshold = 30.0  # minimum per-word confidence to consider (lowered for recall)
        self.line_conf_threshold = 35.0  # minimum line-level confidence
        self.top_k_regions = 12
        # Debugging: save annotated images for inspection when True
        self.debug_output = True if os.getenv('OCR_DEBUG', 'false').lower() == 'true' else False
        # Debug save directory
        self.debug_dir = '/tmp'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed grayscale image
        """
        height, width = image.shape[:2]
        # Upscale modestly for small inputs
        if max(height, width) < 1500:
            image = cv2.resize(image, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast BEFORE thresholding — CLAHE on grayscale is meaningful;
        # applying it after binary thresholding (as the old code did) has no effect.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Otsu threshold on contrast-enhanced grayscale
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

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
    
    def _find_text_block_regions(self, image: np.ndarray) -> List[tuple]:
        """
        Detect candidate text block regions using simple contour detection.
        Step 2: Detect video-card regions before OCR.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of regions (x, y, w, h) to prioritize for OCR
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect text blocks using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
            dilated = cv2.dilate(gray, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            h, w = image.shape[:2]
            
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Filter by size: text blocks should be reasonable size (not tiny, not entire image)
                if 30 < cw < w * 0.9 and 15 < ch < h * 0.3:
                    regions.append((x, y, cw, ch))
            
            return sorted(regions, key=lambda r: r[1])  # Sort by y position (top to bottom)
        except Exception as e:
            print(f"Error detecting text blocks: {e}")
            return []
    
    def extract_text_regions(self, image: np.ndarray, fast: bool = False) -> List[Dict]:
        """
        Extract text regions from image using OCR.
        Tries the hybrid C++/Python pipeline first for better accuracy;
        falls back to the pure-Python path if unavailable.

        Args:
            image: Input BGR image.
            fast:  If True, use single-scale OCR only (halves latency).
        """
        # ── Hybrid C++ pipeline (preferred) ──────────────────────
        if _HAS_HYBRID:
            try:
                regions = hybrid_extract_text_regions(image)
                if regions:
                    print(f"[DEBUG] Hybrid pipeline returned {len(regions)} regions")
                    return regions
                print("[DEBUG] Hybrid pipeline returned 0 regions, falling back to Python")
            except Exception as e:
                print(f"[DEBUG] Hybrid pipeline error: {e}, falling back to Python")

        # ── Pure-Python fallback ─────────────────────────────────
        # Multi-scale OCR -> collect word boxes then merge into line-level regions
        h_img, w_img = image.shape[:2]
        # Fast mode uses single scale only (halves OCR time).
        scales = [1.0] if fast else [1.0, 1.5]
        words = []

        for scale in scales:
            try:
                if scale != 1.0:
                    img = cv2.resize(image, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_CUBIC)
                else:
                    img = image.copy()

                proc = self.preprocess_image(img)
                # PSM 11 = sparse text: finds scattered text in grid layouts (YouTube homepage).
                # PSM 6 (uniform block) was wrong here — it merges unrelated text columns.
                config = '--oem 3 --psm 11'
                ocr_data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config=config)
            except Exception as e:
                print(f"pytesseract error (scale={scale}): {e}")
                continue

            n = len(ocr_data.get('text', []))
            print(f"[DEBUG] OCR scale={scale} found {n} boxes")

            for i in range(n):
                txt = (ocr_data.get('text', [''])[i] or '').strip()
                if not txt:
                    continue

                conf_raw = ocr_data.get('conf', [''])[i]
                try:
                    conf = float(conf_raw)
                except Exception:
                    try:
                        conf = float(str(conf_raw).strip())
                    except Exception:
                        conf = -1.0

                if conf < self.ocr_conf_threshold:
                    continue

                left = int(ocr_data.get('left', [0])[i])
                top = int(ocr_data.get('top', [0])[i])
                width = int(ocr_data.get('width', [0])[i])
                height = int(ocr_data.get('height', [0])[i])

                # Map coordinates back to original image space
                if scale != 1.0:
                    left = int(left / scale)
                    top = int(top / scale)
                    width = int(width / scale)
                    height = int(height / scale)

                # Filter by geometry
                if width < 6 or height < 6 or width > w_img * 0.95 or height > h_img * 0.5:
                    continue

                area = width * height
                words.append({
                    'text': txt,
                    'x': left,
                    'y': top,
                    'w': width,
                    'h': height,
                    'conf': conf,
                    'area': area
                })

        if not words:
            return []

        # ---- Deduplicate overlapping word boxes from multi-scale OCR ----
        # Sort by (y, x) so nearby duplicates are adjacent in the list
        words = sorted(words, key=lambda r: (r['y'], r['x']))
        deduped = []
        for w in words:
            merged = False
            for d in deduped:
                # Same word at roughly the same position? Keep higher conf.
                if (abs(w['x'] - d['x']) < max(15, d['w'] * 0.5) and
                    abs(w['y'] - d['y']) < max(10, d['h'] * 0.5) and
                    (w['text'].lower() == d['text'].lower() or
                     w['text'].lower() in d['text'].lower() or
                     d['text'].lower() in w['text'].lower())):
                    if w['conf'] > d['conf']:
                        d.update(w)  # replace with higher-confidence version
                    merged = True
                    break
            if not merged:
                deduped.append(w)
        words = deduped
        print(f"[DEBUG] After word dedup: {len(words)} unique words")

        # Group words into lines by vertical proximity
        words = sorted(words, key=lambda r: r['y'])
        lines = []
        cur_line = [words[0]]
        cur_y = words[0]['y']

        for w in words[1:]:
            # Use 55% of the smaller word height as the vertical grouping threshold.
            # The old average-height threshold was too loose (e.g. 30px for 30px-tall words),
            # causing lines from adjacent video-card rows to merge together.
            thresh = max(8, int(min(w['h'], cur_line[-1]['h']) * 0.55))
            if abs(w['y'] - cur_y) <= thresh:
                cur_line.append(w)
                # update cur_y as average
                cur_y = int(sum(item['y'] for item in cur_line) / len(cur_line))
            else:
                lines.append(cur_line)
                cur_line = [w]
                cur_y = w['y']
        if cur_line:
            lines.append(cur_line)

        # Split lines by large horizontal gaps (prevents merging different video card columns)
        split_lines = []
        for line in lines:
            if len(line) <= 1:
                split_lines.append(line)
                continue
            line_sorted_x = sorted(line, key=lambda r: r['x'])
            avg_w = sum(r['w'] for r in line_sorted_x) / len(line_sorted_x)
            # Use a tight gap threshold: 2x the average word width, minimum 60px.
            # YouTube grid cards are ~300px apart; word gaps within a title are <40px.
            gap_thresh = max(60, avg_w * 2)
            sub = [line_sorted_x[0]]
            for j in range(1, len(line_sorted_x)):
                prev_word = line_sorted_x[j - 1]
                curr_word = line_sorted_x[j]
                gap = curr_word['x'] - (prev_word['x'] + prev_word['w'])
                if gap > gap_thresh:
                    split_lines.append(sub)
                    sub = [curr_word]
                else:
                    sub.append(curr_word)
            split_lines.append(sub)
        lines = split_lines

        # Build line regions (normalize to collapse any remaining adjacent duplicate words)
        text_regions = []
        for line in lines:
            line_sorted = sorted(line, key=lambda r: r['x'])
            line_text = self._normalize_line_text(' '.join([r['text'] for r in line_sorted]))
            x_min = min(r['x'] for r in line_sorted)
            y_min = min(r['y'] for r in line_sorted)
            x_max = max(r['x'] + r['w'] for r in line_sorted)
            y_max = max(r['y'] + r['h'] for r in line_sorted)
            w_box = x_max - x_min
            h_box = y_max - y_min
            total_area = sum(r['area'] for r in line_sorted)
            # weighted confidence by area
            if total_area > 0:
                weighted_conf = sum(r['conf'] * r['area'] for r in line_sorted) / total_area
            else:
                weighted_conf = float(sum(r['conf'] for r in line_sorted) / len(line_sorted))

            region = {
                'text': line_text,
                'x': int(x_min),
                'y': int(y_min),
                'w': int(w_box),
                'h': int(h_box),
                'conf': float(weighted_conf),
                'area': int(w_box * h_box)
            }

            # Basic filtering: reasonable length and confidence
            if len(line_text) >= 2 and (region['conf'] >= self.line_conf_threshold or len(line_text) > 8):
                text_regions.append(region)

        # Sort top-to-bottom
        text_regions = sorted(text_regions, key=lambda r: r['y'])

        # Optionally save debug annotated image
        if self.debug_output:
            try:
                self._draw_debug_image(image, text_regions)
            except Exception as e:
                print(f"[DEBUG] Failed to write debug image: {e}")

        print(f"[DEBUG] extract_text_regions produced {len(text_regions)} line regions")
        if text_regions:
            for reg in text_regions[:8]:
                print(f"[DEBUG] Line region: '{reg['text'][:40]}' conf={reg['conf']:.1f} bbox=({reg['x']},{reg['y']},{reg['w']},{reg['h']})")

        return text_regions

    def _filter_text_by_heuristics(self, text: str, strict: bool = False) -> bool:
        """
        Step 4: Post-filter OCR text with regex + heuristics.
        Check if text is plausible for video title/channel.
        
        Args:
            text: Text to validate
            strict: If True, apply stricter filtering. If False, be more lenient.
            
        Returns:
            True if text passes heuristics, False otherwise
        """
        if not text or len(text) < 2:
            return False
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Filter out pure numbers and tiny text
        if len(text) < 2 or len(text) > 250:
            return False
        
        # Filter out pure emoji or special characters
        if not any(c.isalnum() for c in text):
            return False
        
        # ONLY block EXACT stopword matches in strict mode
        if strict:
            if text.lower() in self.stopwords:
                return False
        
        # Filter out view-count patterns (contains only numbers and units)
        if re.match(r'^[\d,\.]+\s*[KMBkb]?$', text):
            return False
        
        # Filter out common UI-only labels in strict mode
        if strict:
            ui_only_words = {'search', 'create', 'upload', 'subscribe', 'home', 'explore', 'library'}
            if text.lower() in ui_only_words:
                return False
        
        # NEW: Filter out garbage OCR text
        # Too many special characters relative to alphanumeric
        alpha_count = sum(1 for c in text if c.isalnum())
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if alpha_count > 0 and special_count / alpha_count > 0.5:
            return False
        
        # NEW: Filter text with too many single-character "words"
        words = text.split()
        if len(words) > 0:
            single_char_words = sum(1 for w in words if len(w) == 1)
            if single_char_words / len(words) > 0.4:
                return False
        
        # NEW: Filter text that looks like random characters (low avg word length with many words)
        if len(words) >= 4:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 2.0:
                return False
        
        # NEW: Filter text containing timestamp patterns mid-text (likely merged OCR)
        if re.search(r'\d+:\d+\s+\d+:\d+', text):
            return False
        
        # NEW: Filter text that's mostly pipe/bracket/special chars
        if re.match(r'^[\|\[\]{}<>@#\-_\s]+$', text):
            return False
        
        # NEW: Filter obvious garbage patterns
        if re.search(r'[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]', text.lower()):
            # Looks like "e e t T h i e n" - single chars with spaces
            return False
        
        # NEW: Filter YouTube navigation bar text (contains multiple category words)
        nav_categories = ['home', 'gaming', 'music', 'sports', 'news', 'live', 'shorts', 
                          'subscriptions', 'library', 'history', 'trending', 'explore',
                          'slam dunks', 'golden state', 'italian cuisine', 'speedruns',
                          'gordon ramsay', 'mixes', 'among us', 'warriors']
        text_lower = text.lower()
        nav_matches = sum(1 for cat in nav_categories if cat in text_lower)
        if nav_matches >= 3:
            return False
        
        # NEW: Filter text that looks like concatenated category list
        # (many capitalized words with no connecting words)
        if len(words) >= 5:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            if capitalized / len(words) > 0.7:
                # Most words are capitalized - likely category/topic list
                # Check if it lacks articles/prepositions typical in sentences
                connectors = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'is', 'are', 'was', 'were'}
                has_connector = any(w.lower() in connectors for w in words)
                if not has_connector and len(words) >= 6:
                    return False
        
        return True
    
    def _aggregate_frame_results(self, video_data: Dict) -> Dict:
        """
        Step 5: Temporal aggregation - aggregate OCR results across multiple frames.
        Take the most frequent/highest-confidence result for the same field.
        
        Args:
            video_data: Current frame's video data
            
        Returns:
            Aggregated video data from frame history
        """
        self.frame_history.append(video_data)
        if len(self.frame_history) > self.max_history_frames:
            self.frame_history.pop(0)
        
        # Only aggregate if we have multiple frames
        if len(self.frame_history) < 2:
            return video_data
        
        aggregated = {
            'title': self._aggregate_field('title'),
            'channel': self._aggregate_field('channel'),
            'views': self._aggregate_field('views'),
            'duration': self._aggregate_field('duration'),
        }
        
        # Keep current timestamp and raw count
        aggregated['timestamp'] = video_data['timestamp']
        aggregated['raw_text_regions'] = video_data.get('raw_text_regions', 0)
        aggregated['text_regions'] = video_data.get('text_regions', [])
        
        return aggregated
    
    def _extract_card_fields(self, text_regions: List[Dict], image: np.ndarray) -> Dict:
        """
        Extract title, channel, views, and duration from a group of text regions
        representing a single video card.  Uses YouTube's consistent vertical layout:
        title (top, larger text) → channel → metadata (views · age) → duration badge.
        """
        if not text_regions:
            return {}

        sorted_regions = sorted(text_regions, key=lambda r: r['y'])

        view_pat = re.compile(r'[\d,\.]+\s*[KMBkmb]?\s*views?', re.I)
        duration_pat = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')
        channel_at_pat = re.compile(r'@\w+')
        ago_pat = re.compile(r'\d+\s*(?:second|minute|hour|day|week|month|year)s?\s*ago', re.I)
        meta_pat = re.compile(r'(?:views?|watching|subscribers?|streamed|premiered|ago\b)', re.I)

        title = None
        channel = None
        views = None
        duration = None
        title_candidates = []

        for r in sorted_regions:
            text = r.get('text', '').strip()
            if not text or len(text) < 2:
                continue

            # --- Duration ---
            dm = duration_pat.search(text)
            if dm and not duration:
                duration = dm.group(0)

            # --- Views ---
            vm = view_pat.search(text)
            if vm:
                if not views:
                    views = vm.group(0)
                continue  # view lines are never titles

            # --- Channel handle (@name) ---
            if channel_at_pat.search(text) and not channel:
                channel = text
                continue

            # --- Metadata line (views, ago, etc.) ---
            if meta_pat.search(text) or ago_pat.search(text):
                if not views:
                    vm2 = view_pat.search(text)
                    if vm2:
                        views = vm2.group(0)
                continue

            # --- Skip obvious noise ---
            if not self._filter_text_by_heuristics(text, strict=False):
                continue

            # Everything else is a title candidate
            title_candidates.append(r)

        # Pick the best title: topmost candidate(s)
        if title_candidates:
            title_candidates.sort(key=lambda r: r['y'])
            title = title_candidates[0].get('text', '').strip()

            # Multi-line title: merge a second line if it sits directly below
            if len(title_candidates) > 1:
                first_bottom = title_candidates[0]['y'] + title_candidates[0].get('h', 20)
                second = title_candidates[1]
                if second['y'] - first_bottom < 30:
                    title = title + ' ' + second.get('text', '').strip()

        # Infer channel from the first short text below the title region
        if not channel and title and title_candidates:
            title_bottom_y = title_candidates[0]['y'] + title_candidates[0].get('h', 20)
            for r in sorted_regions:
                ry = r.get('y', 0)
                text = r.get('text', '').strip()
                if ry > title_bottom_y and ry < title_bottom_y + 60 and 2 < len(text) < 50:
                    if not view_pat.search(text) and not ago_pat.search(text):
                        channel = text
                        break

        return {
            'title': title,
            'channel': channel,
            'views': views,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'raw_text_regions': len(text_regions),
            'text_regions': text_regions,
        }

    def _aggregate_field(self, field: str) -> Optional[str]:
        """
        Get the most frequent value for a field across frame history,
        or the highest confidence if field appears in only one frame.
        
        Args:
            field: Field name ('title', 'channel', 'views', 'duration')
            
        Returns:
            Most frequent/highest-confidence value or None
        """
        values = [f.get(field) for f in self.frame_history if f.get(field)]
        
        if not values:
            return None
        
        # Return most frequent value
        counter = Counter(values)
        most_common, _ = counter.most_common(1)[0]
        return most_common
    
    def clear_frame_history(self):
        """Clear temporal frame history."""
        self.frame_history = []
    
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

    def _normalize_line_text(self, text: str) -> str:
        """
        Normalize a line by collapsing adjacent duplicate words and trimming extra whitespace.
        Keeps original casing for output but uses a normalized token for duplicate detection.
        """
        if not text:
            return text
        # Split on whitespace, keep punctuation attached for output but normalize for comparison
        tokens = re.split(r'\s+', text.strip())
        out_tokens = []
        prev_norm = None
        for t in tokens:
            # Normalize token for comparison: strip punctuation and lowercase
            norm = re.sub(r'[^A-Za-z0-9]', '', t).lower()
            if norm == '':
                # Keep punctuation-only tokens only if not duplicate
                if prev_norm != t:
                    out_tokens.append(t)
                    prev_norm = t
                continue
            # Only collapse immediate duplicates (common OCR artifact like 'Hi Hi')
            if norm == prev_norm:
                # skip the duplicate token
                continue
            out_tokens.append(t)
            prev_norm = norm

        return ' '.join(out_tokens)

    def _group_lines_by_vertical_gaps(self, lines: List[Dict], gap_multiplier: float = 1.5, min_gap_px: int = 18) -> List[List[Dict]]:
        """
        Group line-level regions into subgroups using vertical gaps.

        Args:
            lines: list of line regions (must contain 'y' and 'h') sorted by y
            gap_multiplier: multiplier on average line height to determine a split
            min_gap_px: minimum pixel gap to consider

        Returns:
            List of grouped line lists
        """
        if not lines:
            return []

        sorted_lines = sorted(lines, key=lambda r: r['y'])
        heights = [r.get('h', 20) for r in sorted_lines]
        avg_h = sum(heights) / len(heights) if heights else 20
        groups = []
        cur = [sorted_lines[0]]

        for prev, curr in zip(sorted_lines, sorted_lines[1:]):
            gap = curr['y'] - (prev['y'] + prev.get('h', 0))
            threshold = max(min_gap_px, int(avg_h * gap_multiplier))
            if gap > threshold:
                groups.append(cur)
                cur = [curr]
            else:
                cur.append(curr)

        if cur:
            groups.append(cur)

        return groups

    def _segment_by_view_lines(self, text_regions: List[Dict], max_above_px: int = 260) -> List[List[Dict]]:
        """
        Segment the page into groups based on detected view-count or duration lines.
        For each view/duration line, collect lines above it within `max_above_px` pixels
        until the previous view/duration line. This helps identify video cards even
        when the contour-based block detection fails.
        Returns list of line groups (each group is a list of regions sorted by y).
        """
        if not text_regions:
            return []

        # Patterns
        view_pat = re.compile(r'[\d,\.]+\s*[KMkm]?\s*views?', re.IGNORECASE)
        duration_pat = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?')

        # Find candidate bottom lines (views or duration)
        bottoms = []
        for r in text_regions:
            t = r.get('text', '')
            if view_pat.search(t) or duration_pat.search(t):
                bottoms.append(r)

        if not bottoms:
            return []

        bottoms_sorted = sorted(bottoms, key=lambda r: r['y'])
        groups = []
        prev_bottom_y = 0

        for b in bottoms_sorted:
            top_limit = max(0, b['y'] - max_above_px)
            group = []
            for r in text_regions:
                ry = r.get('y', 0)
                # include regions between prev_bottom_y and current bottom y, and above top_limit
                if ry >= prev_bottom_y and ry <= b['y'] and ry >= top_limit:
                    group.append(r)
            if group:
                group_sorted = sorted(group, key=lambda r: r['y'])
                groups.append(group_sorted)
            prev_bottom_y = b['y'] + b.get('h', 0)

        return groups

    def _split_possible_titles(self, text: str) -> List[str]:
        """
        Given a possibly-concatenated title string, attempt to split it into multiple
        candidate titles. Strategy:
        - First split on colons that are not part of time patterns (avoid splitting '4:34')
        - If that yields only one part, fall back to splitting on patterns like ': 3' (older heuristic)
        - Clean each part and return those that pass basic heuristics
        """
        if not text:
            return []

        # Split on colon that is not between digits (avoid time patterns)
        parts = re.split(r'(?<!\d):(?!\d)', text)
        if len(parts) <= 1:
            # fallback: split on colon-digit patterns
            parts = re.split(r":\s*\d+\b", text)

        cleaned = []
        for p in parts:
            p = p.strip()
            # strip leading/trailing non-alphanumeric punctuation
            p = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', p).strip()
            if not p:
                continue
            # require at least two words (to avoid splitting on single labels)
            if len(re.findall(r"\w+", p)) < 2:
                continue
            # apply heuristics
            if not self._filter_text_by_heuristics(p, strict=False):
                continue
            cleaned.append(p)

        return cleaned
    
    def extract_video_title(self, text_regions: List[Dict], image: np.ndarray) -> Optional[str]:
        """
        Extract video title from text regions.
        YouTube titles are typically at the top-center of the page.
        Step 4: Apply heuristic filtering to rejected unwanted text.
        
        Args:
            text_regions: List of extracted text regions
            image_shape: (height, width) of the image
            
        Returns:
            Extracted video title or None
        """
        height, width = image.shape[:2]
        # Normalize text regions (collapse duplicated adjacent tokens)
        for r in text_regions:
            r['text'] = self._normalize_line_text(r.get('text', ''))

        # Score candidate lines and pick best by area*conf * content score * position weight
        height, width = image.shape[:2]
        candidates = []
        for r in text_regions:
            txt = r.get('text', '').strip()
            if not txt:
                continue
            area = r.get('area', r.get('w', 0) * r.get('h', 0)) or 1
            conf = float(r.get('conf', 0) or 0)

            # Content heuristics
            words = re.findall(r"\w+", txt)
            word_count = len(words)
            avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0
            alpha_chars = sum(ch.isalnum() for ch in txt)
            frac_alnum = alpha_chars / max(1, len(txt))

            # Penalize lines with lots of punctuation/noise
            noise_penalty = 1.0
            non_alnum_frac = 1.0 - frac_alnum
            if non_alnum_frac > 0.25:
                noise_penalty = 0.6
            if non_alnum_frac > 0.5:
                noise_penalty = 0.2

            # Boost for reasonable sentence-like content
            content_score = 1.0
            if word_count >= 4 and avg_word_len >= 3:
                content_score = 1.6
            elif word_count >= 2 and avg_word_len >= 3:
                content_score = 1.2

            # Position weight: titles often appear near the upper-middle of the content area
            y = r.get('y', 0)
            pos_weight = 1.0
            rel_y = y / max(1, height)
            if 0.15 <= rel_y <= 0.6:
                pos_weight = 1.3
            if 0.25 <= rel_y <= 0.5:
                pos_weight = 1.6

            score = area * conf * content_score * pos_weight * noise_penalty
            candidates.append((score, r))

        # Define reasonable horizontal and vertical bounds for title search
        center_x_start = int(width * 0.05)
        center_x_end = int(width * 0.95)
        top_region_y = int(height * 0.6)

        # Collect candidate regions in the top/center area (common title zone)
        # But filter out regions at the very top (navigation bar) and those that span too wide
        title_candidates = []
        for region in text_regions:
            y = region.get('y', 0)
            x = region.get('x', 0)
            w = region.get('w', 0)
            text = region.get('text', '')
            
            # Skip regions at the very top (likely nav bar) - y < 5% of height
            if y < height * 0.05:
                continue
            
            # Skip regions that span more than 70% of the width (likely nav bar or banner)
            if w > width * 0.7:
                continue
            
            # Skip if text contains multiple YouTube nav categories
            nav_cats = ['home', 'gaming', 'music', 'shorts', 'subscriptions', 'live', 'sports']
            text_lower = text.lower()
            if sum(1 for cat in nav_cats if cat in text_lower) >= 2:
                continue
            
            if (y < top_region_y and
                center_x_start <= x <= center_x_end and
                len(text) > 1):
                title_candidates.append(region)

        title_candidates.sort(key=lambda x: (x['y'], x['x']))  # Sort top-to-bottom, left-to-right

        if title_candidates:
            # Build first-line string from top-most line
            first_line_y = title_candidates[0]['y']
            first_line_parts = [r['text'] for r in title_candidates if abs(r['y'] - first_line_y) < 30]
            combined_first = ' '.join(first_line_parts).strip()

            # Look for a second line immediately below that likely is part of the same title
            second_line_text = None
            for r in title_candidates:
                if r['y'] > first_line_y + 18:
                    sec_y = r['y']
                    second_line_parts = [rr['text'] for rr in title_candidates if abs(rr['y'] - sec_y) < 30]
                    second_line_text = ' '.join(second_line_parts).strip()
                    break

            # If second line exists, prefer the combined two-line title when plausible
            if second_line_text:
                combined_two = (combined_first + ' ' + second_line_text).strip()
                if self._filter_text_by_heuristics(combined_two, strict=False):
                    print(f"[DEBUG] Title from regions (2-line): '{combined_two}'")
                    return combined_two

            # Otherwise, fallback to the single-line first-line candidate
            if combined_first:
                if self._filter_text_by_heuristics(combined_first, strict=True):
                    print(f"[DEBUG] Title from regions: '{combined_first}'")
                    return combined_first
                if self._filter_text_by_heuristics(combined_first, strict=False):
                    print(f"[DEBUG] Title from regions accepted leniently: '{combined_first}'")
                    return combined_first
                print(f"[DEBUG] Combined title filtered out: '{combined_first}'")

        # Fallback: run OCR on the top area crop (more aggressive)
        top_crop_h = int(height * 0.25)
        top_crop = (int(width * 0.05), 0, int(width * 0.9), top_crop_h)
        top_text = self._ocr_region_string(image, top_crop)
        if top_text:
            # Heuristic: choose the longest reasonable line as title
            lines = [l.strip() for l in top_text.splitlines() if len(l.strip()) > 3]
            if lines:
                lines.sort(key=lambda s: -len(s))
                candidate_title = lines[0]
                passed = self._filter_text_by_heuristics(candidate_title, strict=False)
                print(f"[DEBUG] Fallback candidate: '{candidate_title}' passed_filter={passed}")
                if passed:
                    print(f"[DEBUG] Title from fallback: '{candidate_title}'")
                    return candidate_title
                else:
                    print(f"[DEBUG] Fallback title filtered: '{candidate_title}'")

        return None
    
    def extract_channel_name(self, text_regions: List[Dict], image: np.ndarray) -> Optional[str]:
        """
        Extract channel name from text regions.
        Channel names are typically below the title.
        Step 4: Apply heuristic filtering.
        
        Args:
            text_regions: List of extracted text regions
            image_shape: (height, width) of the image
            
        Returns:
            Extracted channel name or None
        """
        height, width = image.shape[:2]
        
        # Look for text below title area (15-40% from top)
        channel_region_y_start = int(height * 0.15)
        channel_region_y_end = int(height * 0.45)
        center_x_start = int(width * 0.05)
        center_x_end = int(width * 0.95)

        channel_candidates = []
        for region in text_regions:
            if (channel_region_y_start < region['y'] < channel_region_y_end and
                center_x_start < region['x'] < center_x_end):
                # Don't filter individual words - combine first, filter later
                txt = region['text']
                channel_candidates.append(region)

        channel_candidates.sort(key=lambda x: (x['y'], x['x']))  # Sort top-to-bottom, left-to-right

        if channel_candidates:
            # Try to get the first likely channel name (usually short and on one line)
            for candidate in channel_candidates:
                txt = candidate['text']
                # Apply filter to individual candidates
                if self._filter_text_by_heuristics(txt, strict=False) and len(txt) < 80:
                    print(f"[DEBUG] Channel from regions: '{txt}'")
                    return txt

        # Fallback: try OCR on a region below the title area where channel usually appears
        channel_crop_y = int(height * 0.15)
        channel_crop_h = int(height * 0.15)
        channel_crop = (int(width * 0.05), channel_crop_y, int(width * 0.6), channel_crop_h)
        ch_text = self._ocr_region_string(image, channel_crop)
        if ch_text:
            # pick first non-empty line
            for line in ch_text.splitlines():
                line = line.strip()
                if line and len(line) < 80 and self._filter_text_by_heuristics(line, strict=False):
                    print(f"[DEBUG] Channel from fallback: '{line}'")
                    return line
            if ch_text.strip():
                print(f"[DEBUG] Channel fallback found text but filtered: {ch_text.splitlines()[:2]}")

        return None

    def get_largest_text_regions(self, text_regions: List[Dict], top_k: int = 12, by: str = 'area') -> List[Dict]:
        """
        Return the largest text regions by area or height.

        Args:
            text_regions: List of OCR text region dicts
            top_k: Number of top regions to return
            by: Metric to rank by: 'area' or 'height'

        Returns:
            List of top-k regions sorted by the chosen metric (descending)
        """
        if not text_regions:
            return []

        def metric(r):
            # Score lines by area * confidence to prefer large, high-confidence lines
            area = r.get('area', r.get('w', 0) * r.get('h', 0))
            conf = float(r.get('conf', 0) or 0)
            if by == 'height':
                return r.get('h', 0) * conf
            return area * conf

        sorted_regions = sorted(text_regions, key=lambda r: metric(r), reverse=True)
        return sorted_regions[:top_k]

    def _draw_debug_image(self, image: np.ndarray, text_regions: List[Dict], top_n: int = 12) -> None:
        """
        Draw bounding boxes and candidate labels onto a copy of the image and save to disk.
        """
        try:
            vis = image.copy()
            h, w = vis.shape[:2]
            # Choose top regions to draw
            regions = self.get_largest_text_regions(text_regions, top_k=top_n)
            for i, r in enumerate(regions):
                x, y, ww, hh = r['x'], r['y'], r['w'], r['h']
                color = (0, 200, 0) if i == 0 else (0, 120, 255)
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), color, 2)
                label = f"{i+1}:{r['text'][:30]} ({r.get('conf',0):.0f})"
                cv2.putText(vis, label, (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # Save to temp file
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(self.debug_dir, f"ocr_debug_{ts}.png")
            cv2.imwrite(fname, vis)
            print(f"[DEBUG] Wrote annotated OCR debug image: {fname}")
        except Exception as e:
            print(f"[DEBUG] Error drawing debug image: {e}")
    
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
    
    def extract_video_data(self, image: np.ndarray, enable_temporal_aggregation: bool = True, fast: bool = False) -> Dict:
        """
        Extract raw OCR text from a YouTube screenshot.

        The heavy lifting of *structuring* text into individual videos is now
        handled downstream by the Llama LLM parser (see llm_parser.py).
        This method focuses on:
          1. Cropping browser chrome.
          2. Running OCR (hybrid C++/Python pipeline).
          3. Returning all text_regions so the LLM can reason over them.

        Args:
            image: BGR input image.
            enable_temporal_aggregation: Unused (kept for API compat).
            fast: If True, use single-scale OCR for lower latency.

        Returns:
            Dictionary with 'text_regions' list and basic metadata.
        """
        # Step 1: Ignore top N pixels (browser chrome)
        N = 120
        height, width = image.shape[:2]
        if height > N:
            cropped_image = image[N:, :]
        else:
            cropped_image = image

        text_regions = self.extract_text_regions(cropped_image, fast=fast)

        print(f"[extract_video_data] Extracted {len(text_regions)} text regions")
        if text_regions:
            for r in text_regions[:8]:
                print(f"[extract_video_data]   '{r['text'][:50]}' conf={r.get('conf',0):.0f} @ ({r['x']},{r['y']})")

        # Return raw regions — the LLM pipeline in app.py will structure them
        return {
            'text_regions': text_regions,
            'timestamp': datetime.now().isoformat(),
            'raw_text_regions': len(text_regions),
        }


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
                        # Support multi-video results
                        if isinstance(video_data, dict) and video_data.get('multi'):
                            for v in video_data.get('videos', []):
                                if v.get('title'):
                                    self.video_data_history.append(v)
                                    print(f"Extracted: {v.get('title', 'N/A')}")
                        else:
                            if video_data and video_data.get('title'):
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

