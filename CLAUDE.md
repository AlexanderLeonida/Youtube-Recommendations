# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

TwinTube Vector is a YouTube recommendation system that:
1. Records the user's screen while browsing YouTube
2. Extracts video titles/metadata via OCR (Tesseract), LLM parsing (Llama GGUF), and HTML scraping
3. Stores extracted videos in MySQL
4. (Optional) Generates personalized recommendations via a two-tower PyTorch/JAX model with FAISS vector search

## Running the System

```bash
# Start all services (builds images, runs in background, streams logs)
./run.sh

# Or manually:
docker-compose up --build -d
docker-compose logs -f

# Stop everything
docker-compose down

# Start optional C++ OpenCV OCR preprocessor
docker-compose --profile tools up opencv-ocr
```

## Development Commands

**Frontend** (React + TypeScript, port 3000):
```bash
cd frontend
npm install
npm run dev      # Vite dev server
npm run build
npm run lint
```

**Backend** (Node.js/Express, port 4000):
```bash
cd backend
npm install
node src/index.js
```

**OCR Service** (Python/Flask, port 5001):
```bash
cd ocr-service
pip install -r requirements.txt
python app.py
```

**ML Service** (PyTorch/FastAPI, port 8000):
```bash
cd ml
pip install -r requirements.txt
python server.py
```

## Architecture

Six Docker services communicate as follows:

```
Browser (DisplayMedia API)
    ↓ Base64 PNG frames
Frontend :3000
    ↓ API calls
Backend :4000  (Express — proxies to OCR, queries MySQL)
    ↓                    ↓
OCR Service :5001     MySQL :3306
(Tesseract + Llama    (table: videos)
 + BeautifulSoup)
    ↓ (optional)
ML Service :8000      Redis :6379
(two-tower model,     (embedding cache,
 FAISS index)          LRU + TTL=1h)
```

**Video extraction pipeline** (OCR service, `ocr-service/app.py`):
1. Fast path: YouTube HTML scraper via BeautifulSoup (`/api/scrape-youtube`)
2. Medium path: Tesseract OCR + spatial text clustering (heuristic)
3. Thorough path: Llama GGUF LLM parsing (background thread)

All paths save results to MySQL. Frontend polls `/api/videos` every 10s to display results.

**ML recommendation pipeline** (`ml/inference.py`, `ml/model.py`):
- Two-tower DNN: user tower (attention-based) + video tower (multi-modal)
- 256-dim embedding space, Recall@100 target: 68%
- INT8 quantization (70% latency reduction target)
- FAISS IVF-PQ index with GPU acceleration
- P50 <8ms, P99 <12ms latency targets

## Key Files

| File | Role |
|------|------|
| `backend/src/index.js` | Express API — proxies OCR calls, handles MySQL CRUD |
| `frontend/src/pages/MainApp.tsx` | Main UI — screen capture, frame sending, video display |
| `frontend/src/services/api.ts` | Axios client; note OCR calls go direct to `:5001`, backend calls to `:4000` |
| `ocr-service/app.py` | Flask OCR service — recording, extraction, DB writes |
| `ocr-service/screen_recorder.py` | Screen capture worker |
| `ocr-service/llm_parser.py` | Llama GGUF-based title extraction |
| `ocr-service/youtube_scraper.py` | BeautifulSoup HTML scraper |
| `ml/inference.py` | Production inference engine (quantization, caching, FAISS) |
| `ml/model.py` | Two-tower PyTorch model definition |
| `ml/server.py` | FastAPI server wrapping the inference engine |

## Environment Variables

Backend (`.env`): `PORT`, `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

OCR service (docker-compose): `LLAMA_HF_REPO`, `EMBED_MODEL`, context/thread settings

ML service (docker-compose): `USE_QUANTIZATION`, `USE_GPU_INDEX`, `REDIS_URL`

## Headless / No-Display Mode

The OCR service detects when no display is available and falls back to returning placeholder video data. This allows backend/frontend development without a running screen recorder.
