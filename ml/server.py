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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[RecommendationEngine] = None


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
    global engine
    
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
        global engine
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

    if not clicks:
        raise HTTPException(status_code=400, detail="No click history found. Browse YouTube first.")

    # Filter by session if specified
    if req.session_id:
        clicks = [c for c in clicks if c["session_id"] == req.session_id]

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
        result = engine.get_recommendations(user_features, top_k=req.top_k, use_cache=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

    # Load CTR stats if available
    ctr_stats_path = "./checkpoints/ctr_stats.json"
    per_video_ctr = {}
    overall_ctr = 0.0
    if os.path.exists(ctr_stats_path):
        with open(ctr_stats_path) as f:
            ctr_data = json.load(f)
        per_video_ctr = ctr_data.get("per_video_ctr", {})
        overall_ctr = ctr_data.get("overall_ctr", 0.0)

    # Map int IDs back to YouTube IDs and titles
    recommendations = []
    for int_id, score in zip(result["video_ids"], result["scores"]):
        yt_id = int_to_video.get(int(int_id), "unknown")
        video_ctr = per_video_ctr.get(yt_id, None)
        recommendations.append({
            "video_id": yt_id,
            "score": float(score),
            "ctr": video_ctr,
            "youtube_url": f"https://www.youtube.com/watch?v={yt_id}" if yt_id != "unknown" else None,
        })

    return {
        "recommendations": recommendations,
        "history_size": len(watch_history),
        "latency_ms": result["latency_ms"],
        "overall_ctr": overall_ctr,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
