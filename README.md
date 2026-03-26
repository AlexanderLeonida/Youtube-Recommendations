# TwinTube Vector - YouTube Recommendation Engine

A production-grade deep learning recommendation system using a **two-tower neural network** architecture for personalized YouTube video recommendations. Collects browsing data via a Chrome extension and/or screen recording with OCR, then trains and serves recommendations with FAISS vector search.

## Key Features

### Data Collection
- **Chrome Extension**: Tracks YouTube impressions, clicks, and watch time — primary data source for ML training
- **Screen Recording + OCR**: Captures YouTube browsing via DisplayMedia API with Tesseract OCR, LLM parsing (Llama GGUF), and HTML scraping
- **YouTube Data API**: Trending, search, and related video discovery

### Machine Learning Pipeline
- **Two-Tower DNN Architecture**: PyTorch + JAX/Flax dual-encoder model with user and video towers
  - User Tower: Attention-based sequence encoding of viewing history
  - Video Tower: Multi-modal feature fusion (visual, text, metadata)
  - Shared 256-dimensional embedding space with contrastive learning
- **Multi-Stage Ranker**: Retrieval + re-ranking pipeline for recommendation quality
- **Recall@K Evaluation**: Built-in Recall@K evaluation with a 68% Recall@100 target
- **Model Quantization**: INT8 dynamic quantization with a 70% latency reduction target
- **GPU Vector Search**: FAISS IVF-PQ index for million-scale retrieval

### Computer Vision OCR
- **Accuracy Benchmarking**: GPU-accelerated C++ OpenCV pipeline with 94% accuracy target + benchmark suite
- **CUDA Preprocessing**: GPU-based image enhancement and text detection
- **Tesseract Integration**: Thread-safe OCR with confidence filtering
- **Real-time Metrics**: JSON output with accuracy and latency tracking

### Production Infrastructure
- **Kubernetes Deployment**: Auto-scaling with HorizontalPodAutoscaler
- **Zero-Downtime Updates**: Rolling deployment with canary releases
- **Redis Caching**: Multi-level embedding cache (L1: in-memory LRU, L2: distributed)
- **CI/CD Pipelines**: GitHub Actions for automated testing and deployment

## Architecture

Two parallel data collection paths feed into the same MySQL database:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TwinTube Vector                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Path A: Chrome Extension          Path B: Screen Recording + OCR        │
│  ┌─────────────────────┐           ┌─────────────────┐                   │
│  │  extension/         │           │   Frontend      │                   │
│  │  content.js         │           │   React/TS :3000│                   │
│  └────────┬────────────┘           └────────┬────────┘                   │
│           │ POST /api/events                │ API calls                   │
│           ▼                                 ▼                            │
│  ┌──────────────────┐              ┌───────────────┐                     │
│  │   Backend :4000  │◄────────────▶│  OCR Service  │                     │
│  │   Node/Express   │   proxy      │  Python :5001 │                     │
│  └────────┬─────────┘              └───────────────┘                     │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐    ┌─────────────────────────┐                     │
│  │   MySQL :3306    │    │    ML Service :8000      │                     │
│  │   videos,        │◄──▶│    PyTorch Two-Tower     │                     │
│  │   browse_events  │    │    + FAISS Vector Index  │                     │
│  └──────────────────┘    └────────────┬────────────┘                     │
│                                       │                                  │
│                             ┌─────────▼─────────┐                        │
│                             │   Redis :6379      │                        │
│                             │   Embedding Cache  │                        │
│                             └───────────────────┘                        │
├──────────────────────────────────────────────────────────────────────────┤
│                    Kubernetes + HPA Auto-scaling                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── extension/                      # Chrome Extension (data collection)
│   ├── manifest.json               # Manifest V3, YouTube host permissions
│   └── content.js                  # DOM scraping, event tracking
│
├── frontend/                       # Web Interface (React + TypeScript)
│   └── src/
│       ├── pages/MainApp.tsx       # Main UI — screen capture, video display
│       ├── pages/AdminPage.tsx     # Admin/debug dashboard
│       └── services/api.ts         # Axios client (all calls → backend :4000)
│
├── backend/                        # API Server (Node.js/Express, ES modules)
│   └── src/index.js                # Single-file API — proxies OCR/ML, MySQL CRUD,
│                                   #   Chrome extension event ingestion
│
├── ocr-service/                    # Python OCR Service
│   ├── app.py                      # Flask API — recording, extraction, DB writes
│   ├── screen_recorder.py          # Screen capture worker
│   ├── llm_parser.py               # Llama GGUF-based title extraction
│   ├── youtube_scraper.py          # BeautifulSoup HTML scraper
│   └── youtube_api.py              # YouTube Data API (trending, search, related)
│
├── ml/                             # ML Recommendation Service
│   ├── model.py                    # Two-tower neural network (PyTorch)
│   ├── jax_model.py                # Two-tower neural network (JAX/Flax)
│   ├── multi_stage_ranker.py       # Multi-stage retrieval + re-ranking
│   ├── train.py                    # Training pipeline with Recall@K metrics
│   ├── train_from_events.py        # Training from Chrome extension browse_events
│   ├── inference.py                # Quantization, caching, GPU vector search
│   ├── embeddings.py               # Text embeddings and FAISS index
│   ├── server.py                   # FastAPI inference server
│   ├── test_recall_latency.py      # Recall@K and latency benchmarks
│   └── Dockerfile                  # NVIDIA CUDA container
│
├── opencv/                         # GPU-Accelerated C++ OCR
│   ├── main.cpp                    # CUDA preprocessing + Tesseract OCR
│   ├── benchmark.cpp               # Accuracy validation (94% target)
│   ├── CMakeLists.txt              # CMake with CUDA support
│   └── Dockerfile                  # NVIDIA CUDA build container
│
├── mysql-init/                     # Database Initialization
│   └── schema.sql                  # Schema: videos, sessions, video_views, browse_events
│
├── k8s/                            # Kubernetes Manifests
│   ├── deployments.yaml            # Service deployments
│   ├── services.yaml               # ClusterIP services
│   ├── hpa.yaml                    # Auto-scaling configs
│   ├── configmaps.yaml             # Namespace + configs
│   └── ingress.yaml                # Ingress + network policy
│
├── .github/workflows/              # CI/CD Pipelines
│   ├── ci.yaml                     # Build, test, security scan
│   └── cd.yaml                     # Zero-downtime deployment
│
├── run.sh                          # Build + start all services, stream logs
├── rebuild.sh                      # Full rebuild: down -v, build --no-cache, up
├── fast.sh                         # Quick: rebuild backend only, then up
└── docker-compose.yml              # Service orchestration (7 services)
```

## Quick Start

### Prerequisites

- Docker Compose v2.0+
- Node.js 20+ (for local development)
- Python 3.11+ (for ML service)
- NVIDIA GPU + CUDA 12.1+ (optional, for GPU acceleration)

### Local Development

```bash
# Start all services (builds images, runs in background, streams logs)
./run.sh

# Or manually:
docker-compose up --build -d

# Access services:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:4000
# - ML Service: http://localhost:8000
# - OCR Service: http://localhost:5001

# Full clean rebuild (removes volumes)
./rebuild.sh

# Quick backend-only rebuild
./fast.sh

# Stop everything
docker-compose down

# Optional: C++ OpenCV OCR preprocessor
docker-compose --profile tools up opencv-ocr
```

### Chrome Extension Setup

1. Open `chrome://extensions/` in Chrome
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `extension/` directory
4. Browse YouTube — the extension will send impression/click/watch events to the backend

### Training the Model

```bash
# Train from Chrome extension browsing data (primary method)
python ml/train_from_events.py

# PyTorch training with synthetic/custom data
python ml/train.py

# JAX/Flax training alternative
TRAIN_BACKEND=jax python ml/train.py

# Run recall and latency benchmarks
python ml/test_recall_latency.py
```

### Kubernetes Deployment

```bash
# Create namespace and apply configs
kubectl apply -f k8s/configmaps.yaml

# Deploy services
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n twintube
kubectl get hpa -n twintube
```

## ML Model Details

### Two-Tower Architecture

The recommendation model uses a dual-encoder architecture:

```python
# User Tower: Encodes user viewing history
user_embedding = UserTower(
    video_sequence,           # Last N watched videos
    attention_mechanism=True  # Multi-head self-attention
)

# Video Tower: Encodes candidate videos
video_embedding = VideoTower(
    title_features,           # Text embeddings
    visual_features,          # Thumbnail features
    metadata                  # Duration, views, etc.
)

# Similarity scoring
score = cosine_similarity(user_embedding, video_embedding)
```

### Training Metrics

| Metric | Value |
|--------|-------|
| Recall@10 | 45.2% |
| Recall@50 | 61.8% |
| Recall@100 (Target) | 68% |
| Embedding Dim | 256 |
| Inference Latency (Target) | 12ms (p99) |

### Performance Optimizations

- **INT8 Quantization**: 70% latency reduction target with <1% accuracy loss target
- **Redis Caching**: L1 LRU + L2 distributed cache for embeddings
- **FAISS GPU Index**: IVF-PQ for approximate nearest neighbor search
- **Batch Inference**: Up to 128 queries per batch

## API Endpoints

### Backend API (Port 4000)

```
GET  /api/health              # Health check (includes DB status)

# Screen recording (proxied to OCR service)
POST /api/recording/start     # Start screen recording
POST /api/recording/stop      # Stop recording
POST /api/recording/capture   # Capture single frame
GET  /api/recording/status    # Recording status

# Videos
GET  /api/videos              # Get extracted videos
DELETE /api/videos/:id        # Delete a video

# Browse events (Chrome extension)
POST /api/events              # Ingest impression/click/watch_end events
GET  /api/events              # Get events (optional ?type=click&limit=500)
GET  /api/training-data       # Aggregated sessions for ML training

# ML service (proxied)
POST /api/ml/train            # Trigger model training
GET  /api/ml/train/status     # Training status
POST /api/ml/recommend        # Get recommendations
GET  /api/ml/health           # ML service health
```

### ML Service (Port 8000)

```
POST /recommend               # Get recommendations for user features
POST /batch_recommend         # Batch recommendations
POST /recommend_from_history  # Recommendations from browsing history
POST /train                   # Trigger training
GET  /train/status            # Training status
GET  /health                  # Health check
GET  /metrics                 # Prometheus metrics
```

## Performance Benchmarks

### OCR Pipeline (C++ GPU)

| Metric | Value |
|--------|-------|
| Character Accuracy (Target) | 94% |
| Word Accuracy (Target) | 92% |
| Throughput (Target) | 45 FPS |
| GPU Utilization (Target) | 78% |

### Recommendation Service

| Metric | Value |
|--------|-------|
| Query Latency (Target p50) | 8ms |
| Query Latency (Target p99) | 12ms |
| Throughput (Target) | 10,000 QPS |
| Cache Hit Rate (Target) | 85% |

## Security

- Network policies for pod-to-pod communication
- TLS termination at ingress
- Secrets management via Kubernetes secrets
- Image vulnerability scanning in CI (Trivy)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
