"""
Hybrid C++/Python OCR pipeline.

Flow:
  1. Save the input image to a temp file.
  2. Call the C++ `ocr_preprocess` binary which uses MSER + morphology
     to detect text regions, clean/binarise each crop, and write them
     as numbered PNGs alongside a manifest.json.
  3. Run pytesseract on each cleaned crop to get text + confidence.
  4. Return structured text-region dicts compatible with the rest of
     the pipeline.
"""

import json
import os
import shutil
import subprocess
import tempfile
from typing import List, Dict, Optional

import cv2
import numpy as np
import pytesseract

# Path to the C++ binary (installed into /usr/local/bin by the Dockerfile)
CPP_BINARY = os.getenv("OCR_PREPROCESS_BIN", "/usr/local/bin/ocr_preprocess")


def _cpp_available() -> bool:
    """Check whether the C++ preprocessor binary exists."""
    return os.path.isfile(CPP_BINARY) and os.access(CPP_BINARY, os.X_OK)


def hybrid_extract_text_regions(image: np.ndarray) -> List[Dict]:
    """
    Run the hybrid C++ → Python OCR pipeline on *image* (BGR numpy array).

    Returns a list of dicts with keys:
        text, x, y, w, h, conf, area
    Compatible with the existing YouTubeVideoExtractor pipeline.
    """
    if not _cpp_available():
        print("[hybrid] C++ binary not found, falling back to pure-Python pipeline")
        return []

    tmpdir = tempfile.mkdtemp(prefix="ocr_")
    input_path = os.path.join(tmpdir, "input.png")

    try:
        # 1. Write the image to disk
        cv2.imwrite(input_path, image)

        # 2. Call the C++ preprocessor
        result = subprocess.run(
            [CPP_BINARY, input_path, tmpdir],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"[hybrid] C++ preprocessor failed: {result.stderr.strip()}")
            return []

        # The binary prints stderr diagnostics and stdout = manifest path
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                print(f"[hybrid] {line}")

        manifest_path = os.path.join(tmpdir, "manifest.json")
        if not os.path.exists(manifest_path):
            print("[hybrid] No manifest.json produced")
            return []

        with open(manifest_path) as f:
            manifest = json.load(f)

        if not manifest:
            print("[hybrid] Manifest is empty — no regions detected")
            return []

        print(f"[hybrid] Processing {len(manifest)} regions through Tesseract")

        # 3. OCR each cleaned crop
        text_regions: List[Dict] = []
        for entry in manifest:
            crop_path = os.path.join(tmpdir, entry["file"])
            if not os.path.exists(crop_path):
                continue

            crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
            if crop is None:
                continue

            # Run Tesseract with word-level data
            try:
                ocr_data = pytesseract.image_to_data(
                    crop,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6",
                )
            except Exception as e:
                print(f"[hybrid] Tesseract error on {entry['file']}: {e}")
                continue

            # Build line text from all detected words
            words = []
            confs = []
            n = len(ocr_data.get("text", []))
            for i in range(n):
                txt = (ocr_data["text"][i] or "").strip()
                if not txt:
                    continue
                try:
                    c = float(ocr_data["conf"][i])
                except (ValueError, TypeError):
                    c = 0.0
                if c < 20:
                    continue
                words.append(txt)
                confs.append(c)

            if not words:
                continue

            line_text = " ".join(words)

            # Collapse immediate duplicate words (OCR artefact)
            deduped_tokens = []
            prev = None
            for tok in line_text.split():
                norm = tok.lower().strip(".,!?;:")
                if norm and norm == prev:
                    continue
                deduped_tokens.append(tok)
                prev = norm
            line_text = " ".join(deduped_tokens)

            if len(line_text) < 2:
                continue

            avg_conf = sum(confs) / len(confs) if confs else 0.0

            text_regions.append({
                "text": line_text,
                "x": entry["x"],
                "y": entry["y"],
                "w": entry["w"],
                "h": entry["h"],
                "conf": avg_conf,
                "area": entry["w"] * entry["h"],
            })

        # --- Merge horizontally adjacent fragments into single lines ---
        # With reduced C++ dilation, one title may produce multiple small
        # boxes.  Merge them if they are on the same Y-row AND close in X.
        text_regions.sort(key=lambda r: (r["y"], r["x"]))
        merged: List[Dict] = []
        for r in text_regions:
            if merged:
                prev = merged[-1]
                same_row = abs(r["y"] - prev["y"]) < max(20, prev["h"] * 0.8)
                x_gap = r["x"] - (prev["x"] + prev["w"])
                close_x = x_gap < 60  # tight: only merge within-title gaps
                if same_row and close_x:
                    # Merge into prev
                    prev["text"] = prev["text"] + " " + r["text"]
                    new_x = min(prev["x"], r["x"])
                    new_right = max(prev["x"] + prev["w"], r["x"] + r["w"])
                    new_top = min(prev["y"], r["y"])
                    new_bot = max(prev["y"] + prev["h"], r["y"] + r["h"])
                    prev["x"] = new_x
                    prev["y"] = new_top
                    prev["w"] = new_right - new_x
                    prev["h"] = new_bot - new_top
                    prev["area"] = prev["w"] * prev["h"]
                    prev["conf"] = (prev["conf"] + r["conf"]) / 2
                    continue
            merged.append(dict(r))
        text_regions = merged

        # --- Filter garbage regions ---
        # Low confidence + mostly non-alpha = thumbnail OCR noise
        clean_regions: List[Dict] = []
        for r in text_regions:
            text = r["text"]
            alpha_count = sum(1 for c in text if c.isalpha())
            if len(text) > 0 and alpha_count / len(text) < 0.35:
                continue  # too much garbage
            if r["conf"] < 35 and len(text) > 30:
                continue  # low confidence long text = noise
            clean_regions.append(r)
        text_regions = clean_regions

        # Sort top-to-bottom, left-to-right
        text_regions.sort(key=lambda r: (r["y"], r["x"]))

        print(f"[hybrid] Produced {len(text_regions)} text regions")
        for r in text_regions[:10]:
            print(f"  [{r['conf']:.0f}] ({r['x']},{r['y']},{r['w']},{r['h']}) {r['text'][:60]}")

        return text_regions

    except subprocess.TimeoutExpired:
        print("[hybrid] C++ preprocessor timed out")
        return []
    except Exception as e:
        print(f"[hybrid] Unexpected error: {e}")
        return []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
