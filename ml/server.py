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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
