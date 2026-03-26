# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

TwinTube Vector is a YouTube recommendation system that:
1. Collects YouTube browsing data via a Chrome extension (impressions, clicks, watch time) **and/or** screen recording with OCR
2. Stores video metadata and engagement events in MySQL
3. Trains a two-tower PyTorch/JAX recommendation model on the collected data
4. Serves personalized recommendations via FAISS vector search

## Running the System

```bash
./run.sh                  # Build + start all services, stream logs
docker-compose down       # Stop everything
./rebuild.sh              # Full rebuild: down -v, build --no-cache, up
./fast.sh                 # Quick: rebuild backend only, then up

# Optional C++ OpenCV OCR preprocessor
docker-compose --profile tools up opencv-ocr
```

## Development Commands

**Frontend** (React + TypeScript, port 3000 — uses Create React App, NOT Vite):
```bash
cd frontend && npm install
npm start                 # CRA dev server (react-scripts start)
npm run build
npm test                  # react-scripts test (Jest)
```

**Backend** (Node.js/Express with ES modules, port 4000):
```bash
cd backend && npm install
npm run dev               # nodemon (auto-restart on changes)
npm start                 # plain node src/index.js
```

**OCR Service** (Python/Flask, port 5001 externally → 5000 internally):
```bash
cd ocr-service && pip install -r requirements.txt
python app.py
```

**ML Service** (PyTorch/FastAPI, port 8000):
```bash
cd ml && pip install -r requirements.txt
python server.py                          # FastAPI inference server
python train_from_events.py               # Train from Chrome extension data
TRAIN_BACKEND=jax python train.py         # JAX/Flax training alternative
python test_recall_latency.py             # Recall@K and latency benchmarks
```

**Chrome Extension** (`extension/`):
Load as unpacked extension in Chrome. Injects `content.js` on YouTube pages, sends impression/click/watch_end events to backend `POST /api/events`.

## Architecture

Two parallel data collection paths feed into the same MySQL database:

```
Path A: Chrome Extension                   Path B: Screen Recording + OCR
extension/content.js                       Frontend :3000 (DisplayMedia API)
    ↓ POST /api/events                         ↓ API calls
Backend :4000 ──────────────────────────► MySQL :3306
    ↓ proxy                                    (tables: videos, browse_events,
OCR Service :5001                                sessions, video_views)
    ↓ (optional)
ML Service :8000 ◄─────────────────────► Redis :6379
(two-tower model, FAISS index)            (embedding cache, LRU + TTL=1h)
```

**Backend** (`backend/src/index.js`) is the single entry file — no router modules. It proxies to OCR and ML services, handles MySQL CRUD, and upserts `videos` rows from Chrome extension events.

**Video extraction pipeline** (OCR service, `ocr-service/app.py`):
1. Fast path: YouTube HTML scraper via BeautifulSoup (`/api/scrape-youtube`)
2. Medium path: Tesseract OCR + spatial text clustering
3. Thorough path: Llama GGUF LLM parsing (background thread)
4. YouTube Data API path (`ocr-service/youtube_api.py`): trending, search, related videos

**ML recommendation pipeline**:
- Two-tower DNN: user tower (attention-based) + video tower (multi-modal) — `ml/model.py` (PyTorch), `ml/jax_model.py` (JAX/Flax)
- Multi-stage ranker with re-ranking — `ml/multi_stage_ranker.py`
- `ml/train_from_events.py`: trains from `browse_events` table data (the Chrome extension path)
- 256-dim embedding space, FAISS IVF-PQ index, INT8 quantization
- P50 <8ms, P99 <12ms latency targets

## Key Files

| File | Role |
|------|------|
| `backend/src/index.js` | Entire Express API — proxy to OCR/ML, MySQL CRUD, Chrome extension event ingestion |
| `frontend/src/pages/MainApp.tsx` | Main UI — screen capture, frame sending, video display |
| `frontend/src/services/api.ts` | Axios client — all calls go through backend `:4000` |
| `frontend/src/pages/AdminPage.tsx` | Admin/debug dashboard |
| `extension/content.js` | Chrome extension — scrapes YouTube DOM, sends browse events to backend |
| `ocr-service/app.py` | Flask OCR service — recording, extraction, DB writes |
| `ocr-service/youtube_api.py` | YouTube Data API wrapper (trending, search, related) |
| `ml/model.py` | Two-tower PyTorch model definition |
| `ml/jax_model.py` | Two-tower JAX/Flax model definition |
| `ml/multi_stage_ranker.py` | Multi-stage retrieval + re-ranking pipeline |
| `ml/train_from_events.py` | Training pipeline using Chrome extension browse_events |
| `ml/inference.py` | Production inference engine (quantization, caching, FAISS) |
| `ml/server.py` | FastAPI server wrapping the inference engine |
| `mysql-init/schema.sql` | Database schema — `videos`, `sessions`, `video_views`, `browse_events` tables |

## Database Schema

The `browse_events` table is the primary ML training data source. Each row is an impression, click, or watch_end event from the Chrome extension. The backend's `GET /api/training-data` endpoint aggregates these into sessions with impressions/clicks/watch_times for model training.

The `videos` table is populated both by OCR extraction and by upserts from Chrome extension click/impression events.

## Environment Variables

Backend (`.env`): `PORT`, `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT`

OCR service (docker-compose): `LLAMA_HF_REPO`, `LLAMA_HF_FILENAME`, `EMBED_MODEL`, `LLAMA_N_CTX`, `LLAMA_N_THREADS`, `YOUTUBE_API_KEY`

ML service (docker-compose): `USE_QUANTIZATION`, `USE_GPU_INDEX`, `USE_REDIS_CACHE`, `REDIS_HOST`, `REDIS_PORT`, `BACKEND_URL`

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `ci.yaml`: Runs on push/PR to `main`, `develop`, `feature/*`. Jobs: backend (npm lint + tsc + test), frontend (npm lint + tsc + test + build), ML (flake8 + mypy + pytest), OpenCV C++ (cmake build + benchmark), Docker image builds to GHCR, Trivy security scan, K8s manifest validation.
- `cd.yaml`: Deployment pipeline.

## Headless / No-Display Mode

The OCR service detects when no display is available and falls back to returning placeholder video data. This allows backend/frontend development without a running screen recorder.
