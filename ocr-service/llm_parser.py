"""
RAG pipeline for structuring raw OCR text into YouTube video metadata.

Architecture (Retrieval-Augmented Generation):
  1. Embed OCR text regions using a lightweight neural model (fastembed / BGE).
  2. Retrieve the most relevant regions via cosine similarity against
     video-metadata queries  (the **retrieval** in RAG).
  3. Feed retrieved context to a local Llama model (llama-cpp-python)
     for structured JSON extraction  (the **generation** in RAG).

No external LLM service (e.g. Ollama) is required — everything runs
locally inside this container.
"""

import json
import os
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# ── Configuration ────────────────────────────────────────────────────
MODELS_DIR       = os.getenv("MODELS_DIR", "/app/models")
LLAMA_HF_REPO    = os.getenv("LLAMA_HF_REPO", "bartowski/Llama-3.2-1B-Instruct-GGUF")
LLAMA_HF_FILE    = os.getenv("LLAMA_HF_FILENAME", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
N_CTX            = int(os.getenv("LLAMA_N_CTX", "2048"))
N_THREADS        = int(os.getenv("LLAMA_N_THREADS", "4"))
MAX_TOKENS       = int(os.getenv("LLAMA_MAX_TOKENS", "2048"))
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "30"))

# ── Lazy-loaded singletons ──────────────────────────────────────────
_llm = None
_embedder = None

# ── Retrieval queries (what we look for in the OCR text) ────────────
_RAG_QUERIES = [
    "YouTube video title name",
    "channel creator uploader name",
    "view count views million thousand",
    "video duration length minutes seconds",
    "posted uploaded days weeks months ago",
    "thumbnail description",
]

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a structured-data extraction assistant.
You will receive raw OCR text scraped from a YouTube homepage screenshot.
The text is noisy and may contain navigation elements, ads, and OCR artifacts.

Your task: identify every distinct YouTube video visible on the page and
return ONLY a JSON array. Each element must have these fields:
  - "title"      : the video title (string, required)
  - "channel"    : the channel or creator name (string or null)
  - "views"      : view count as shown, e.g. "1.2M views" (string or null)
  - "posted_ago" : how long ago posted, e.g. "3 days ago" (string or null)
  - "duration"   : video length, e.g. "12:34" (string or null)

Rules:
1. Return ONLY the JSON array — no markdown fences, no commentary.
2. Each video should appear exactly once.
3. Ignore navigation bar text (Home, Shorts, Subscriptions, Gaming, Music…).
4. Ignore ads, sign-in prompts, and sidebar text.
5. If you cannot determine a field, set it to null.
6. Clean up OCR noise: fix obvious misspellings only when confident.
7. Do NOT invent videos that aren't in the text.
"""

USER_PROMPT_TEMPLATE = """\
Here is the raw OCR text from a YouTube screenshot. Each line is prefixed with
[x,y] pixel coordinates indicating where on the screen that text appears.
Text regions with similar x values are in the same column; close y values are
on the same row. Use this spatial info to group text into individual video cards.

---
{ocr_text}
---

Return the JSON array of videos you can identify.
"""


# ═════════════════════════════════════════════════════════════════════
#  Model loading
# ═════════════════════════════════════════════════════════════════════

def _ensure_gguf_model() -> str:
    """Return path to the GGUF model file, downloading from HuggingFace if needed."""
    model_path = Path(MODELS_DIR) / LLAMA_HF_FILE
    if model_path.exists():
        print(f"[RAG] Model already present at {model_path}")
        return str(model_path)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[RAG] Downloading {LLAMA_HF_REPO}/{LLAMA_HF_FILE} …")
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    path = hf_hub_download(
        repo_id=LLAMA_HF_REPO,
        filename=LLAMA_HF_FILE,
        local_dir=MODELS_DIR,
    )
    print(f"[RAG] Model downloaded in {time.time() - t0:.1f}s → {path}")
    return path


def _get_llm():
    """Lazy-load the local Llama model."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        model_path = _ensure_gguf_model()
        print(f"[RAG] Loading Llama model ({model_path}) …")
        t0 = time.time()
        _llm = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False,
            chat_format="llama-3",
        )
        print(f"[RAG] Llama model loaded in {time.time() - t0:.1f}s")
    return _llm


def _get_embedder():
    """Lazy-load the fastembed text-embedding model (ONNX — no PyTorch)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        print(f"[RAG] Loading embedding model: {EMBED_MODEL_NAME}")
        t0 = time.time()
        _embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
        print(f"[RAG] Embedding model loaded in {time.time() - t0:.1f}s")
    return _embedder


# ═════════════════════════════════════════════════════════════════════
#  RAG retrieval
# ═════════════════════════════════════════════════════════════════════

def _rag_retrieve(text_regions: List[Dict], top_k: int = RAG_TOP_K) -> List[Dict]:
    """
    RAG retrieval step.

    Embed all OCR text regions and the predefined metadata queries, then
    return the *top_k* most relevant regions by cosine similarity.
    """
    if len(text_regions) <= top_k:
        return text_regions  # all regions fit — skip retrieval

    embedder = _get_embedder()

    # Extract text from regions
    texts: list[str] = []
    valid_indices: list[int] = []
    for i, r in enumerate(text_regions):
        t = r.get("text", "").strip()
        if t and len(t) >= 2:
            texts.append(t)
            valid_indices.append(i)

    if not texts:
        return text_regions

    # Embed regions and queries  (fastembed returns generators)
    region_embeds = np.array(list(embedder.embed(texts)))        # (N, D)
    query_embeds  = np.array(list(embedder.embed(_RAG_QUERIES))) # (Q, D)

    # Normalise for cosine similarity
    region_norms = region_embeds / (np.linalg.norm(region_embeds, axis=1, keepdims=True) + 1e-9)
    query_norms  = query_embeds  / (np.linalg.norm(query_embeds,  axis=1, keepdims=True) + 1e-9)

    # Max similarity of each region across all queries → (N,)
    sims = (region_norms @ query_norms.T).max(axis=1)

    # Pick top-K indices
    top_idx = np.argsort(sims)[-top_k:]
    top_idx_sorted = sorted(top_idx)  # preserve reading order

    return [text_regions[valid_indices[i]] for i in top_idx_sorted if i < len(valid_indices)]


# ═════════════════════════════════════════════════════════════════════
#  Llama generation
# ═════════════════════════════════════════════════════════════════════

def _build_ocr_text_blob(text_regions: List[Dict]) -> str:
    """
    Flatten text regions into a readable blob sorted top→bottom, left→right.
    Each line is prefixed with [col,row] pixel coordinates so the LLM can
    spatially group text into video-card clusters (title / channel / metadata).
    """
    if not text_regions:
        return ""
    sorted_regions = sorted(
        text_regions, key=lambda r: (r.get("y", 0), r.get("x", 0))
    )
    lines = []
    for r in sorted_regions:
        t = r.get("text", "").strip()
        if len(t) < 2:
            continue
        x, y = r.get("x", 0), r.get("y", 0)
        lines.append(f"[{x},{y}] {t}")
    return "\n".join(lines)


def _call_llama(prompt: str, system: str = SYSTEM_PROMPT) -> Optional[str]:
    """Run inference on the local Llama model and return the response text."""
    try:
        llm = _get_llm()
    except Exception as e:
        print(f"[RAG] Failed to load Llama model: {e}")
        return None

    try:
        print("[RAG] Running Llama inference …")
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - t0
        text = resp["choices"][0]["message"]["content"]
        print(f"[RAG] Llama responded in {elapsed:.1f}s  ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"[RAG] Llama inference error: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════
#  JSON parsing
# ═════════════════════════════════════════════════════════════════════

def _parse_llm_response(raw: str) -> List[Dict]:
    """
    Parse the LLM's JSON response into a list of video dicts.
    Handles common issues: markdown fences, trailing commas, partial JSON.
    """
    if not raw:
        return []

    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Locate the JSON array in the response
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        cleaned = cleaned[bracket_start : bracket_end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"[RAG] Failed to parse JSON: {e}")
            print(f"[RAG] Raw response (first 500 chars): {raw[:500]}")
            return []

    if not isinstance(parsed, list):
        parsed = [parsed]

    # Validate and normalise each entry
    videos: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title or len(title) < 3:
            continue

        videos.append({
            "title":      title,
            "channel":    (item.get("channel") or "").strip() or None,
            "views":      (item.get("views") or "").strip() or None,
            "posted_ago": (item.get("posted_ago") or "").strip() or None,
            "duration":   (item.get("duration") or "").strip() or None,
        })

    return videos


# ═════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════

def parse_with_llm(text_regions: List[Dict]) -> List[Dict]:
    """
    Main entry point.  RAG pipeline:
      1. Retrieve most relevant OCR text regions (embedding similarity).
      2. Build a focused prompt from retrieved context.
      3. Generate structured JSON via local Llama model.

    Args:
        text_regions: list of dicts with at least 'text', 'x', 'y'.

    Returns:
        List of video dicts (title, channel, views, posted_ago, duration).
    """
    if not text_regions:
        return []

    # ── Step 1: RAG retrieval ────────────────────────────────────
    print(f"[RAG] Starting pipeline with {len(text_regions)} text regions")
    retrieved = _rag_retrieve(text_regions, top_k=RAG_TOP_K)
    print(f"[RAG] Retrieved {len(retrieved)} relevant regions (from {len(text_regions)})")

    # ── Step 2: Build prompt from retrieved context ──────────────
    ocr_blob = _build_ocr_text_blob(retrieved)
    if not ocr_blob or len(ocr_blob) < 20:
        print("[RAG] Retrieved OCR text too short, skipping")
        return []

    print(f"[RAG] Sending {len(ocr_blob)} chars to Llama")
    prompt = USER_PROMPT_TEMPLATE.format(ocr_text=ocr_blob)

    # ── Step 3: Generate structured output ───────────────────────
    raw_response = _call_llama(prompt)
    if raw_response is None:
        return []

    videos = _parse_llm_response(raw_response)
    print(f"[RAG] Parsed {len(videos)} videos from Llama response")
    for v in videos:
        print(f"[RAG]   → {v['title'][:60]}  |  ch={v['channel']}  |  views={v['views']}")

    return videos


def check_llm_health() -> Dict:
    """Check whether the local Llama model is available and loadable."""
    model_path = Path(MODELS_DIR) / LLAMA_HF_FILE
    model_exists = model_path.exists()

    return {
        "status": "ok" if _llm is not None else ("ready" if model_exists else "model_not_downloaded"),
        "backend": "llama-cpp-python (local RAG)",
        "model_repo": LLAMA_HF_REPO,
        "model_file": LLAMA_HF_FILE,
        "model_downloaded": model_exists,
        "model_loaded": _llm is not None,
        "embed_model": EMBED_MODEL_NAME,
        "rag_top_k": RAG_TOP_K,
    }
