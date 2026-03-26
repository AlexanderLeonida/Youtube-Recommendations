"""
TwinTube Vector: FastAPI Inference Server.

Production-ready API server for video recommendations with:
- Low-latency inference (<50ms P99)
- Async request handling
- Request batching
- Health monitoring
- Prometheus metrics
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from inference import RecommendationEngine, InferenceConfig
from multi_stage_ranker import MultiStageRecommender, PipelineConfig, ReRankConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[RecommendationEngine] = None
multi_stage: Optional[MultiStageRecommender] = None


class UserFeatures(BaseModel):
    """User behavioral features for recommendation."""
    watch_history: List[int] = Field(..., description="List of recently watched video IDs")
    watch_times: List[float] = Field(..., description="Watch duration for each video (seconds)")
    engagement: List[List[float]] = Field(..., description="Engagement signals [likes, comments, shares]")


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_features: UserFeatures
    top_k: int = Field(default=100, ge=1, le=1000, description="Number of recommendations")
    use_cache: bool = Field(default=True, description="Whether to use embedding cache")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    video_ids: List[int]
    scores: List[float]
    latency_ms: float
    cache_hit: bool


class StageLatency(BaseModel):
    stage1_ms: float
    stage2_ms: float
    stage3_ms: float
    total_ms: float


class MultiStageResponse(BaseModel):
    """Response from the three-stage pipeline."""
    video_ids: List[int]
    scores: List[float]
    latency: StageLatency


class MultiStageRequest(BaseModel):
    """Request for multi-stage recommendations."""
    user_features: UserFeatures
    top_k: int = Field(default=20, ge=1, le=200, description="Final result size")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    index_size: int
    cache_hit_rate: float
    latency_p50_ms: float
    latency_p99_ms: float


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_requests: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cache_hit_rate: float
    index_size: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global engine, multi_stage

    logger.info("Starting TwinTube Vector inference server...")
    
    # Initialize config
    config = InferenceConfig(
        model_path=os.getenv('MODEL_PATH', './checkpoints/best_model.pt'),
        index_path=os.getenv('INDEX_PATH', './index/video_embeddings.faiss'),
        use_quantization=os.getenv('USE_QUANTIZATION', 'true').lower() == 'true',
        use_gpu_index=os.getenv('USE_GPU_INDEX', 'true').lower() == 'true',
        redis_host=os.getenv('REDIS_HOST', 'localhost'),
        redis_port=int(os.getenv('REDIS_PORT', '6379')),
        use_redis_cache=os.getenv('USE_REDIS_CACHE', 'false').lower() == 'true'
    )
    
    # Initialize engine
    engine = RecommendationEngine(config)
    
    # Load or build index
    if os.path.exists(config.index_path):
        engine.index.load(config.index_path)
        logger.info(f"Loaded index from {config.index_path}")
    else:
        logger.warning("No pre-built index found. Index must be built before serving.")

    # Initialize multi-stage pipeline
    multi_stage = MultiStageRecommender(
        PipelineConfig(
            stage1_top_k=int(os.getenv('STAGE1_TOP_K', '1000')),
            stage2_top_k=int(os.getenv('STAGE2_TOP_K', '100')),
            final_top_k=int(os.getenv('FINAL_TOP_K', '20')),
            scorer_embedding_dim=config.embedding_dim,
            scorer_device=config.device,
            rerank_config=ReRankConfig(
                diversity_lambda=float(os.getenv('DIVERSITY_LAMBDA', '0.3')),
            ),
        ),
        engine,
    )
    logger.info("Multi-stage pipeline initialized")

    logger.info("Server ready to serve requests")
    
    yield
    
    # Cleanup
    logger.info("Shutting down server...")


app = FastAPI(
    title="TwinTube Vector Recommendation API",
    description="Low-latency video recommendation service using two-tower neural network",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and key metrics.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    latency_stats = engine.get_latency_stats()
    cache_stats = engine.cache.get_stats()
    
    return HealthResponse(
        status="healthy",
        model_loaded=engine.model is not None,
        index_size=len(engine.index.video_ids) if engine.index.video_ids is not None else 0,
        cache_hit_rate=cache_stats.get('hit_rate', 0.0),
        latency_p50_ms=latency_stats.get('p50_ms', 0.0),
        latency_p99_ms=latency_stats.get('p99_ms', 0.0)
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get detailed metrics.
    
    Returns performance statistics for monitoring.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    latency_stats = engine.get_latency_stats()
    cache_stats = engine.cache.get_stats()
    
    return MetricsResponse(
        total_requests=latency_stats.get('num_requests', 0),
        latency_p50_ms=latency_stats.get('p50_ms', 0.0),
        latency_p95_ms=latency_stats.get('p95_ms', 0.0),
        latency_p99_ms=latency_stats.get('p99_ms', 0.0),
        cache_hit_rate=cache_stats.get('hit_rate', 0.0),
        index_size=len(engine.index.video_ids) if engine.index.video_ids is not None else 0
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized video recommendations.
    
    Takes user behavioral features and returns top-k recommended video IDs
    with similarity scores.
    
    Performance targets:
    - P50 latency: <10ms
    - P99 latency: <50ms
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Convert to dict for engine
        user_features = {
            'watch_history': request.user_features.watch_history,
            'watch_times': request.user_features.watch_times,
            'engagement': request.user_features.engagement
        }
        
        # Get recommendations
        result = engine.get_recommendations(
            user_features=user_features,
            top_k=request.top_k,
            use_cache=request.use_cache
        )
        
        return RecommendationResponse(
            video_ids=result['video_ids'],
            scores=result['scores'],
            latency_ms=result['latency_ms'],
            cache_hit=result['cache_hit']
        )
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/multi_stage", response_model=MultiStageResponse)
async def get_multi_stage_recommendations(request: MultiStageRequest):
    """
    Three-stage recommendation pipeline:
      1. Candidate generation (two-tower + FAISS ANN) → ~1000
      2. Cross-feature scoring → ~100
      3. Diversity re-ranking → final top-k

    Performance targets: P50 <8ms, P99 <12ms
    """
    if multi_stage is None:
        raise HTTPException(status_code=503, detail="Multi-stage pipeline not initialized")

    try:
        user_features = {
            "watch_history": request.user_features.watch_history,
            "watch_times": request.user_features.watch_times,
            "engagement": request.user_features.engagement,
        }

        result = multi_stage.recommend(user_features, top_k=request.top_k)

        return MultiStageResponse(
            video_ids=result["video_ids"],
            scores=result["scores"],
            latency=StageLatency(**result["latency"]),
        )
    except Exception as e:
        logger.error(f"Multi-stage recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/multi_stage/stats")
async def multi_stage_stats():
    """Per-stage latency percentiles for the multi-stage pipeline."""
    if multi_stage is None:
        raise HTTPException(status_code=503, detail="Multi-stage pipeline not initialized")
    return multi_stage.get_latency_stats()


@app.post("/batch_recommend")
async def batch_recommendations(requests: List[RecommendationRequest]):
    """
    Batch recommendation endpoint for multiple users.
    
    More efficient for bulk processing.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    results = []
    for request in requests:
        user_features = {
            'watch_history': request.user_features.watch_history,
            'watch_times': request.user_features.watch_times,
            'engagement': request.user_features.engagement
        }
        
        result = engine.get_recommendations(
            user_features=user_features,
            top_k=request.top_k,
            use_cache=request.use_cache
        )
        results.append(result)
    
    return {"results": results}


@app.post("/index/build")
async def build_index(background_tasks: BackgroundTasks, videos: List[Dict]):
    """
    Build or rebuild the video embedding index.
    
    This is a long-running operation that runs in the background.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    def _build():
        engine.build_video_index(videos)
        logger.info("Index build complete")
    
    background_tasks.add_task(_build)
    
    return {"status": "building", "message": "Index build started in background"}


@app.get("/index/status")
async def index_status():
    """Get index status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "index_loaded": engine.index.video_ids is not None,
        "num_videos": len(engine.index.video_ids) if engine.index.video_ids is not None else 0,
        "embedding_dim": engine.config.embedding_dim
    }


# ── Training from browse events ─────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=2, le=512)
    learning_rate: float = Field(default=1e-3, gt=0)


@app.post("/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train the two-tower model from Chrome extension browse events.
    Runs in background so the request returns immediately.
    """
    from train_from_events import train_from_events

    def _run():
        result = train_from_events(
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.learning_rate,
            backend_url=os.getenv("BACKEND_URL", "http://backend:4000"),
        )
        logger.info(f"Training complete: {result}")
        # Reload model after training
        if result.get("status") == "success" and engine is not None:
            try:
                engine._load_model()
                index_path = os.getenv("INDEX_PATH", "./index/video_embeddings.faiss")
                if os.path.exists(index_path):
                    engine.index.load(index_path)
                logger.info("Reloaded model and index after training")
            except Exception as e:
                logger.error(f"Failed to reload after training: {e}")

    background_tasks.add_task(_run)
    return {"status": "training_started", "epochs": req.epochs}


@app.get("/train/status")
async def train_status():
    """Check if a trained model exists and return CTR stats if available."""
    model_path = os.getenv("MODEL_PATH", "./checkpoints/best_model.pt")
    mapper_path = "./checkpoints/id_mapper.json"
    index_path = os.getenv("INDEX_PATH", "./index/video_embeddings.faiss")
    ctr_stats_path = "./checkpoints/ctr_stats.json"

    result = {
        "model_exists": os.path.exists(model_path),
        "id_mapper_exists": os.path.exists(mapper_path),
        "index_exists": os.path.exists(index_path),
    }

    if os.path.exists(ctr_stats_path):
        import json as _json
        with open(ctr_stats_path) as f:
            result["ctr_stats"] = _json.load(f)

    return result


class EvaluateRequest(BaseModel):
    """Request for model evaluation metrics."""
    k_values: List[int] = Field(default=[5, 10, 20, 50], description="K values for Recall/NDCG/HitRate")
    latency_runs: int = Field(default=100, ge=10, le=1000, description="Number of inference runs for latency benchmarking")


@app.post("/evaluate")
async def evaluate_model(req: EvaluateRequest):
    """
    Comprehensive model evaluation using leave-one-out on real browse data.

    Metrics computed (standard RecSys evaluation):
    - Recall@K: fraction of relevant items found in top-K
    - NDCG@K: normalized discounted cumulative gain (ranking quality)
    - MRR: mean reciprocal rank of first relevant item
    - Hit Rate@K: fraction of users with at least one hit in top-K
    - Coverage: fraction of catalog appearing in recommendations
    - Latency: P50, P95, P99 inference latency
    - Training loss curve and CTR stats from saved artifacts
    """
    import requests as http_requests
    import json as _json
    import math

    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    mapper_path = "./checkpoints/id_mapper.json"
    if not os.path.exists(mapper_path):
        raise HTTPException(status_code=400, detail="No trained model found. Train first.")

    if not engine.index.is_ready:
        index_path = os.getenv("INDEX_PATH", "./index/video_embeddings.faiss")
        if os.path.exists(index_path):
            try:
                engine._load_model()
                engine.index.load(index_path)
            except Exception:
                pass
        if not engine.index.is_ready:
            raise HTTPException(status_code=400, detail="Model not trained yet.")

    # Load ID mapper
    with open(mapper_path) as f:
        mapper_data = _json.load(f)
    video_to_int = mapper_data["video_to_int"]
    int_to_video = {int(k): v for k, v in mapper_data["int_to_video"].items()}
    catalog_size = len(video_to_int)

    # Fetch sessions from backend
    backend_url = os.getenv("BACKEND_URL", "http://backend:4000")
    try:
        resp = http_requests.get(
            f"{backend_url}/api/training-data", params={"limit": 5000}, timeout=15
        )
        resp.raise_for_status()
        sessions = resp.json().get("sessions", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch training data: {e}")

    if not sessions:
        raise HTTPException(status_code=400, detail="No browse sessions available for evaluation.")

    # ── Build per-user click sequences ──
    user_sequences = []
    for sess in sessions:
        raw_clicks = sess.get("clicks")
        raw_watch = sess.get("watch_times")
        if not raw_clicks:
            continue
        clicks = [c.strip() for c in raw_clicks.split(",") if c.strip()]
        mapped = [video_to_int[c] for c in clicks if c in video_to_int]
        if len(mapped) < 2:
            continue

        watch_map = {}
        if raw_watch:
            for entry in raw_watch.split(","):
                entry = entry.strip()
                if ":" in entry:
                    parts = entry.rsplit(":", 1)
                    if len(parts) == 2:
                        try:
                            watch_map[parts[0].strip()] = float(parts[1])
                        except ValueError:
                            pass

        watch_times = []
        for c in clicks:
            if c in video_to_int:
                watch_times.append(watch_map.get(c, 60.0))

        user_sequences.append({"history": mapped, "watch_times": watch_times})

    if not user_sequences:
        raise HTTPException(
            status_code=400,
            detail="Not enough click sequences (need >=2 clicks per session) for evaluation.",
        )

    # ── Leave-one-out evaluation ──
    max_k = max(req.k_values)
    recall_sums = {k: 0.0 for k in req.k_values}
    ndcg_sums = {k: 0.0 for k in req.k_values}
    mrr_sum = 0.0
    hit_sums = {k: 0 for k in req.k_values}
    all_recommended: set = set()
    n_eval = 0
    latencies: List[float] = []

    for seq in user_sequences:
        history = seq["history"]
        watch_times = seq["watch_times"]
        # Hold out the last click as ground truth
        ground_truth = history[-1]
        input_history = history[:-1]
        input_watch = watch_times[:-1]

        # Pad/truncate to 50
        max_len = 50
        if len(input_history) > max_len:
            input_history = input_history[-max_len:]
            input_watch = input_watch[-max_len:]

        engagement = [[0.0, 0.0, 0.0]] * len(input_history)
        user_features = {
            "watch_history": input_history,
            "watch_times": input_watch,
            "engagement": engagement,
        }

        try:
            t0 = time.time()
            result = engine.get_recommendations(user_features, top_k=max_k, use_cache=False)
            latencies.append((time.time() - t0) * 1000)
        except Exception:
            continue

        retrieved = result["video_ids"]
        all_recommended.update(retrieved)
        n_eval += 1

        # Find rank of ground truth (1-indexed)
        rank = None
        for i, vid in enumerate(retrieved):
            if vid == ground_truth:
                rank = i + 1
                break

        # MRR
        if rank is not None:
            mrr_sum += 1.0 / rank

        # Per-K metrics
        for k in req.k_values:
            top_k_set = set(retrieved[:k])
            # Recall@K (binary — 1 relevant item)
            if ground_truth in top_k_set:
                recall_sums[k] += 1.0
                hit_sums[k] += 1
                # NDCG@K — for single relevant item, NDCG = 1/log2(rank+1) if rank <= K
                if rank is not None and rank <= k:
                    ndcg_sums[k] += 1.0 / math.log2(rank + 1)

    if n_eval == 0:
        raise HTTPException(status_code=400, detail="No valid evaluation sequences.")

    # ── Compile ranking metrics ──
    recall_at_k = {f"recall@{k}": round(recall_sums[k] / n_eval, 4) for k in req.k_values}
    ndcg_at_k = {f"ndcg@{k}": round(ndcg_sums[k] / n_eval, 4) for k in req.k_values}
    hit_rate_at_k = {f"hit_rate@{k}": round(hit_sums[k] / n_eval, 4) for k in req.k_values}
    mrr = round(mrr_sum / n_eval, 4)
    coverage = round(len(all_recommended) / max(catalog_size, 1), 4)

    # ── Latency benchmarks (additional runs for stability) ──
    if len(user_sequences) > 0 and len(latencies) < req.latency_runs:
        extra_runs = req.latency_runs - len(latencies)
        for i in range(extra_runs):
            seq = user_sequences[i % len(user_sequences)]
            hist = seq["history"][:-1]
            wt = seq["watch_times"][:-1]
            if len(hist) > 50:
                hist = hist[-50:]
                wt = wt[-50:]
            uf = {
                "watch_history": hist,
                "watch_times": wt,
                "engagement": [[0.0, 0.0, 0.0]] * len(hist),
            }
            try:
                t0 = time.time()
                engine.get_recommendations(uf, top_k=20, use_cache=False)
                latencies.append((time.time() - t0) * 1000)
            except Exception:
                pass

    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    latency_stats = {
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
        "mean_ms": round(float(lat_arr.mean()), 3),
        "num_runs": len(latencies),
    }

    # ── Training loss curve ──
    loss_history = []
    loss_path = "./checkpoints/loss_history.json"
    if os.path.exists(loss_path):
        with open(loss_path) as f:
            loss_history = _json.load(f)

    # ── CTR stats ──
    ctr_stats = {}
    ctr_path = "./checkpoints/ctr_stats.json"
    if os.path.exists(ctr_path):
        with open(ctr_path) as f:
            raw = _json.load(f)
        ctr_stats = {
            "overall_ctr": raw.get("overall_ctr", 0),
            "total_impressions": raw.get("total_impressions", 0),
            "total_clicks": raw.get("total_clicks", 0),
            "unique_videos": raw.get("unique_videos", 0),
        }

    return {
        "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k,
        "hit_rate_at_k": hit_rate_at_k,
        "mrr": mrr,
        "coverage": coverage,
        "catalog_size": catalog_size,
        "num_eval_sessions": n_eval,
        "latency": latency_stats,
        "loss_history": loss_history,
        "ctr_stats": ctr_stats,
    }


class BrowseRecommendRequest(BaseModel):
    """Recommend videos based on the user's recent browse history."""
    session_id: Optional[str] = None  # If None, uses all sessions
    top_k: int = Field(default=20, ge=1, le=100)


@app.post("/recommend_from_history")
async def recommend_from_history(req: BrowseRecommendRequest):
    """
    Get recommendations using the user's actual browse history from the extension.
    Fetches click history from the backend, encodes it, and searches the index.
    """
    import requests as http_requests

    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    mapper_path = "./checkpoints/id_mapper.json"
    if not os.path.exists(mapper_path):
        raise HTTPException(status_code=400, detail="No trained model. Click 'Train Model' first.")

    if not engine.index.is_ready:
        # Try to reload index in case training just finished
        index_path = os.getenv("INDEX_PATH", "./index/video_embeddings.faiss")
        if os.path.exists(index_path):
            try:
                engine._load_model()
                engine.index.load(index_path)
                logger.info("Hot-reloaded model and index")
            except Exception as e:
                logger.error(f"Failed to reload index: {e}")
        if not engine.index.is_ready:
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Click 'Train Model' to train on your browse data first.",
            )

    # Load ID mapper
    import json
    with open(mapper_path) as f:
        mapper_data = json.load(f)
    video_to_int = mapper_data["video_to_int"]
    int_to_video = {int(k): v for k, v in mapper_data["int_to_video"].items()}
    int_to_channel = {int(k): v for k, v in mapper_data.get("int_to_channel", {}).items()}

    # Fetch user's click history
    backend_url = os.getenv("BACKEND_URL", "http://backend:4000")
    try:
        params = {"type": "click", "limit": 100}
        resp = http_requests.get(f"{backend_url}/api/events", params=params, timeout=10)
        resp.raise_for_status()
        clicks = resp.json().get("events", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch browse history: {e}")

    # Fetch ALL clicked video IDs for filtering (no limit)
    try:
        excl_resp = http_requests.get(f"{backend_url}/api/clicked-video-ids", timeout=10)
        excl_resp.raise_for_status()
        all_clicked_ids = set(excl_resp.json().get("video_ids", []))
    except Exception:
        all_clicked_ids = None  # Fall back to limited set below

    # Fetch impressions for engagement-aware re-scoring
    try:
        imp_params = {"type": "impression", "limit": 500}
        imp_resp = http_requests.get(f"{backend_url}/api/events", params=imp_params, timeout=10)
        imp_resp.raise_for_status()
        impressions = imp_resp.json().get("events", [])
    except Exception:
        impressions = []

    if not clicks:
        raise HTTPException(status_code=400, detail="No click history found. Browse YouTube first.")

    # Filter by session if specified
    if req.session_id:
        clicks = [c for c in clicks if c["session_id"] == req.session_id]
        impressions = [i for i in impressions if i["session_id"] == req.session_id]

    # Track clicked video IDs and impression counts for re-scoring
    # Use the full set from dedicated endpoint if available, otherwise fall back to limited clicks
    if all_clicked_ids is not None:
        clicked_video_ids = all_clicked_ids
    else:
        clicked_video_ids = {c.get("video_id") for c in clicks if c.get("video_id")}
    impression_counts: Dict[str, int] = {}
    for imp in impressions:
        vid = imp.get("video_id", "")
        if vid and vid not in clicked_video_ids:
            impression_counts[vid] = impression_counts.get(vid, 0) + 1

    # Map to int IDs
    watch_history = []
    watch_times = []
    for c in clicks:
        vid = c.get("video_id", "")
        if vid in video_to_int:
            watch_history.append(video_to_int[vid])
            watch_times.append(float(c.get("watch_duration_sec") or 60))

    if not watch_history:
        raise HTTPException(
            status_code=400,
            detail="None of your watched videos are in the trained model. Retrain after more browsing.",
        )

    # Pad/truncate to 50
    max_len = 50
    if len(watch_history) > max_len:
        watch_history = watch_history[-max_len:]
        watch_times = watch_times[-max_len:]

    engagement = [[0.0, 0.0, 0.0]] * len(watch_history)

    # Get recommendations from engine
    user_features = {
        "watch_history": watch_history,
        "watch_times": watch_times,
        "engagement": engagement,
    }

    try:
        # Request extra candidates to compensate for post-filtering
        fetch_k = req.top_k + len(clicked_video_ids) + len(impression_counts)
        result = engine.get_recommendations(user_features, top_k=fetch_k, use_cache=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

    # Load CTR and engagement stats
    ctr_stats_path = "./checkpoints/ctr_stats.json"
    per_video_ctr = {}
    per_video_impressions = {}
    overall_ctr = 0.0
    if os.path.exists(ctr_stats_path):
        with open(ctr_stats_path) as f:
            ctr_data = json.load(f)
        per_video_ctr = ctr_data.get("per_video_ctr", {})
        per_video_impressions = ctr_data.get("per_video_impressions", {})
        overall_ctr = ctr_data.get("overall_ctr", 0.0)

    # Build the set of ALL video IDs the user has ever been shown (clicked + impressed)
    all_seen_ids = set(clicked_video_ids)
    for imp in impressions:
        vid = imp.get("video_id", "")
        if vid:
            all_seen_ids.add(vid)

    # Map int IDs back to YouTube IDs with engagement-aware re-scoring
    recommendations = []
    stale_dropped = 0
    STALE_IMPRESSION_THRESHOLD = 5  # Drop after this many unclicked impressions
    for int_id, score in zip(result["video_ids"], result["scores"]):
        yt_id = int_to_video.get(int(int_id), "unknown")
        if yt_id == "unknown":
            continue

        # Skip videos the user has already watched
        if yt_id in clicked_video_ids:
            continue

        video_ctr = per_video_ctr.get(yt_id, None)
        adjusted_score = float(score)

        # Count total times shown without a click
        live_imp_count = impression_counts.get(yt_id, 0)
        hist_imp_count = int(per_video_impressions.get(yt_id, 0))
        total_ignores = live_imp_count + hist_imp_count

        # Drop stale videos entirely — shown too many times with no interest
        if total_ignores >= STALE_IMPRESSION_THRESHOLD and (video_ctr is None or video_ctr == 0):
            stale_dropped += 1
            continue

        # Mild penalty for fewer unclicked impressions
        if total_ignores > 0 and (video_ctr is None or video_ctr == 0):
            penalty = min(total_ignores * 0.15, 0.8)
            adjusted_score *= (1.0 - penalty)
        elif video_ctr is not None and video_ctr < overall_ctr * 0.5:
            penalty = min((1.0 - video_ctr / max(overall_ctr, 0.01)) * 0.3, 0.5)
            adjusted_score *= (1.0 - penalty)

        recommendations.append({
            "video_id": yt_id,
            "score": adjusted_score,
            "raw_score": float(score),
            "ctr": video_ctr,
            "impressions_ignored": live_imp_count,
            "youtube_url": f"https://www.youtube.com/watch?v={yt_id}",
            "source": "model",
        })

    # Re-sort by engagement-adjusted score and trim to requested size
    recommendations.sort(key=lambda r: r["score"], reverse=True)
    recommendations = recommendations[:req.top_k]

    # ── Discovery: fill slots freed by stale videos with fresh YouTube API results ──
    discovery_slots = min(stale_dropped, max(req.top_k // 3, 2))
    open_slots = req.top_k - len(recommendations)
    discovery_slots = max(discovery_slots, open_slots)
    discovery_videos: list = []

    if discovery_slots > 0:
        # Build search queries from the user's top watched channels
        channel_counts: Dict[str, int] = {}
        for c in clicks:
            ch = c.get("channel_name") or c.get("channel") or ""
            if ch:
                channel_counts[ch] = channel_counts.get(ch, 0) + 1
        top_channels = sorted(channel_counts, key=channel_counts.get, reverse=True)[:3]

        ocr_url = os.getenv("OCR_SERVICE_URL", "http://ocr-service:5000")
        fetched: List[Dict] = []

        # Strategy 1: search by top channels
        for ch in top_channels:
            if len(fetched) >= discovery_slots * 2:
                break
            try:
                sr = http_requests.get(
                    f"{ocr_url}/api/youtube/search",
                    params={"q": ch, "max_results": 10},
                    timeout=8,
                )
                if sr.ok:
                    for v in sr.json().get("videos", []):
                        if v.get("video_id") and v["video_id"] not in all_seen_ids:
                            fetched.append(v)
            except Exception as e:
                logger.warning(f"YouTube discovery search failed for '{ch}': {e}")

        # Strategy 2: related videos from recent clicks (if not enough)
        if len(fetched) < discovery_slots:
            recent_click_ids = [c.get("video_id") for c in clicks[:3] if c.get("video_id")]
            for vid in recent_click_ids:
                if len(fetched) >= discovery_slots * 2:
                    break
                try:
                    rr = http_requests.get(
                        f"{ocr_url}/api/youtube/related",
                        params={"video_id": vid, "max_results": 10},
                        timeout=8,
                    )
                    if rr.ok:
                        for v in rr.json().get("videos", []):
                            if v.get("video_id") and v["video_id"] not in all_seen_ids:
                                fetched.append(v)
                except Exception as e:
                    logger.warning(f"YouTube related-videos failed for '{vid}': {e}")

        # Deduplicate and pick top discovery_slots
        seen_disc: set = set()
        for v in fetched:
            vid = v["video_id"]
            if vid in seen_disc or vid in all_seen_ids or vid in clicked_video_ids:
                continue
            seen_disc.add(vid)
            # Score by popularity (view count) — higher is better for discovery
            raw_views = v.get("view_count_raw", 0)
            disc_score = min(raw_views / 1_000_000, 1.0) * 0.3  # Normalize
            discovery_videos.append({
                "video_id": vid,
                "score": disc_score,
                "raw_score": 0.0,
                "ctr": None,
                "impressions_ignored": 0,
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "source": "discovery",
                "title": v.get("title"),
                "channel": v.get("channel"),
                "views": v.get("views"),
                "duration": v.get("duration"),
                "thumbnail": v.get("thumbnail"),
            })
            if len(discovery_videos) >= discovery_slots:
                break

    # Merge: model recs first, then discovery fills remaining slots
    final = recommendations[:req.top_k - len(discovery_videos)] + discovery_videos

    return {
        "recommendations": final,
        "history_size": len(watch_history),
        "latency_ms": result["latency_ms"],
        "overall_ctr": overall_ctr,
        "impression_suppressed_count": sum(
            1 for r in final if r.get("impressions_ignored", 0) > 0
        ),
        "stale_dropped": stale_dropped,
        "discovery_count": len(discovery_videos),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
