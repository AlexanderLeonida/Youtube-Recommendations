"""
Multi-Stage Recommendation Pipeline for TwinTube Vector.

Three-stage architecture that progressively narrows candidates while
increasing scoring precision at each stage:

  Stage 1 — Candidate Generation (two-tower + FAISS ANN)
      Input:  full corpus (~1M videos)
      Output: ~1000 candidates
      Budget: P50 < 5ms

  Stage 2 — Scoring (cross-feature interaction network)
      Input:  ~1000 candidates + user embedding
      Output: ~100 scored candidates
      Budget: P50 < 3ms

  Stage 3 — Re-ranking (diversity, freshness, business rules)
      Input:  ~100 scored candidates
      Output: final top-k (default 20)
      Budget: P50 < 1ms

Total pipeline target: P50 < 8ms, P99 < 12ms
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2: Cross-feature Interaction Scorer
# ---------------------------------------------------------------------------

class CrossFeatureScorer(nn.Module):
    """
    Lightweight scoring network that refines candidate rankings using
    cross-feature interactions between user and video embeddings.

    Unlike the two-tower retrieval (dot-product only), this model learns
    non-linear interactions via element-wise product, difference, and
    concatenation — giving it much higher discriminative power over the
    ~1000 candidates surfaced by Stage 1.

    Architecture:
        [user_emb; video_emb; user_emb * video_emb; |user_emb - video_emb|]
          → FC(4d, 256) → ReLU → Dropout
          → FC(256, 128) → ReLU → Dropout
          → FC(128, 1) → score
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # 4 interaction features: concat, element product, abs diff, each of dim embedding_dim
        input_dim = embedding_dim * 4
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score each candidate against the user.

        Args:
            user_embedding:      (1, d) or (d,) single user vector
            candidate_embeddings: (n, d) candidate video vectors

        Returns:
            scores: (n,) relevance score per candidate
        """
        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)
        # Broadcast user to match candidates
        user_expanded = user_embedding.expand_as(candidate_embeddings)  # (n, d)

        interaction = torch.cat(
            [
                user_expanded,
                candidate_embeddings,
                user_expanded * candidate_embeddings,
                (user_expanded - candidate_embeddings).abs(),
            ],
            dim=-1,
        )  # (n, 4d)

        scores = self.scorer(interaction).squeeze(-1)  # (n,)
        return scores


# ---------------------------------------------------------------------------
# Stage 3: Re-ranker (diversity + freshness + business rules)
# ---------------------------------------------------------------------------

@dataclass
class ReRankConfig:
    """Knobs for the re-ranking stage."""

    # Maximum Marginal Relevance (MMR) for diversity
    diversity_lambda: float = 0.3  # 0 = pure relevance, 1 = pure diversity

    # Freshness boost: score += freshness_weight * recency_factor
    freshness_weight: float = 0.05

    # Category diversity: max fraction of results from one category
    max_category_fraction: float = 0.4

    # Channel diversity: max results from same channel
    max_per_channel: int = 3

    # Minimum score threshold (drop very low-scoring candidates)
    min_score_threshold: float = -float("inf")

    # Engagement-based scoring: boost high-CTR videos, penalize ignored ones
    engagement_weight: float = 0.15

    # Impression suppression: penalize videos shown many times without clicks
    impression_penalty_weight: float = 0.2


class DiversityReRanker:
    """
    Re-ranks scored candidates to balance relevance with diversity
    and freshness using Maximum Marginal Relevance (MMR).

    MMR iteratively selects items that are both relevant to the user
    and dissimilar to already-selected items:

        MMR(i) = λ · score(i) − (1−λ) · max_{j ∈ S} sim(i, j)

    Additional business-rule constraints:
    - Per-channel cap (avoid one creator dominating)
    - Category diversity floor
    - Freshness boost for recent uploads
    """

    def __init__(self, config: ReRankConfig = ReRankConfig()):
        self.config = config

    def rerank(
        self,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 20,
        metadata: Optional[List[Dict]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MMR re-ranking with business-rule constraints.

        Args:
            candidate_ids: (n,) video IDs
            scores:        (n,) relevance scores from Stage 2
            embeddings:    (n, d) candidate embeddings for diversity calc
            top_k:         final number of results
            metadata:      optional per-candidate dicts with keys like
                           'channel_id', 'category_id', 'upload_age_days'

        Returns:
            selected_ids:   (top_k,) re-ranked video IDs
            selected_scores: (top_k,) final scores
        """
        n = len(candidate_ids)
        if n == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        top_k = min(top_k, n)
        cfg = self.config

        # Filter by minimum score
        mask = scores >= cfg.min_score_threshold
        if not mask.any():
            mask[:] = True  # fallback: keep everything
        candidate_ids = candidate_ids[mask]
        scores = scores[mask]
        embeddings = embeddings[mask]
        if metadata:
            metadata = [metadata[i] for i, m in enumerate(mask) if m]
        n = len(candidate_ids)
        top_k = min(top_k, n)

        # Normalize scores to [0, 1] for combining with diversity
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)

        # Apply freshness and engagement boosts if metadata available
        if metadata:
            for i, meta in enumerate(metadata):
                age_days = meta.get("upload_age_days", 365)
                recency = max(0.0, 1.0 - age_days / 365.0)
                norm_scores[i] += cfg.freshness_weight * recency

                # Engagement boost: reward high-CTR videos
                video_ctr = meta.get("video_ctr")
                if video_ctr is not None:
                    norm_scores[i] += cfg.engagement_weight * video_ctr

                # Impression suppression: penalize heavily-shown but unclicked
                impressions = meta.get("impression_count", 0)
                clicks = meta.get("click_count", 0)
                if impressions > 0 and clicks == 0:
                    penalty = min(impressions * 0.1, 1.0)
                    norm_scores[i] -= cfg.impression_penalty_weight * penalty
                elif impressions > 0 and clicks > 0:
                    # Low CTR relative to impressions — mild penalty
                    vid_ctr = clicks / impressions
                    if vid_ctr < 0.05:
                        norm_scores[i] -= cfg.impression_penalty_weight * (1.0 - vid_ctr / 0.05) * 0.5

        # Pre-compute normalized embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
        normed = embeddings / norms

        # Vectorized MMR: maintain a running max-similarity vector that is
        # updated after each selection, avoiding the O(n*k) inner loop.
        #   max_sim_to_selected[i] = max_{j in S} sim(i, j)
        max_sim_to_selected = np.full(n, -np.inf, dtype=np.float64)
        eligible = np.ones(n, dtype=bool)  # mask for candidates still in play

        # Pre-compute channel IDs for fast cap checking
        channel_ids = None
        if metadata:
            channel_ids = np.array(
                [m.get("channel_id", -1) for m in metadata], dtype=np.int64
            )
        channel_counts: Dict[int, int] = {}

        lam = cfg.diversity_lambda
        selected_indices: List[int] = []

        for step in range(top_k):
            # On the first step, max_sim is 0 (no items selected yet)
            if step == 0:
                mmr_scores = lam * norm_scores
            else:
                mmr_scores = lam * norm_scores - (1 - lam) * max_sim_to_selected

            # Mask out ineligible candidates
            mmr_scores[~eligible] = -np.inf

            # Per-channel cap (vectorized mask update)
            if channel_ids is not None:
                for ch, cnt in channel_counts.items():
                    if cnt >= cfg.max_per_channel:
                        mmr_scores[channel_ids == ch] = -np.inf

            best_idx = int(np.argmax(mmr_scores))
            if mmr_scores[best_idx] == -np.inf:
                break  # no eligible candidates left

            selected_indices.append(best_idx)
            eligible[best_idx] = False

            # Update max-similarity vector: one dot product (1, d) @ (n, d).T
            sim_to_new = normed @ normed[best_idx]  # (n,)
            np.maximum(max_sim_to_selected, sim_to_new, out=max_sim_to_selected)

            if channel_ids is not None:
                ch = int(channel_ids[best_idx])
                if ch >= 0:
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1

        selected_indices = np.array(selected_indices, dtype=np.int64)
        return candidate_ids[selected_indices], scores[selected_indices]


# ---------------------------------------------------------------------------
# Full Multi-Stage Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration."""

    # Stage 1: candidate generation
    stage1_top_k: int = 1000  # candidates from ANN search

    # Stage 2: scoring
    stage2_top_k: int = 100  # candidates after scoring
    scorer_embedding_dim: int = 256
    scorer_device: str = "cpu"

    # Stage 3: re-ranking
    final_top_k: int = 20
    rerank_config: ReRankConfig = field(default_factory=ReRankConfig)


class MultiStageRecommender:
    """
    Orchestrates the full three-stage recommendation pipeline.

    Usage:
        from inference import RecommendationEngine, InferenceConfig
        from multi_stage_ranker import MultiStageRecommender, PipelineConfig

        engine = RecommendationEngine(InferenceConfig(...))
        engine.build_video_index(video_features)

        pipeline = MultiStageRecommender(PipelineConfig(), engine)
        result = pipeline.recommend(user_features)
    """

    def __init__(self, config: PipelineConfig, engine):
        """
        Args:
            config: pipeline tuning knobs
            engine: a RecommendationEngine with a built FAISS index
        """
        self.config = config
        self.engine = engine

        # Stage 2 scorer
        self.scorer = CrossFeatureScorer(
            embedding_dim=config.scorer_embedding_dim,
        )
        self.scorer.to(config.scorer_device)
        self.scorer.eval()

        # Stage 3 re-ranker
        self.reranker = DiversityReRanker(config.rerank_config)

        # Latency tracking per stage
        self._stage_latencies: Dict[str, List[float]] = {
            "stage1_ms": [],
            "stage2_ms": [],
            "stage3_ms": [],
            "total_ms": [],
        }

    # ── public API ────────────────────────────────────────────────────────

    @torch.no_grad()
    def recommend(
        self,
        user_features: Dict,
        top_k: Optional[int] = None,
        candidate_metadata: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run the full three-stage pipeline.

        Args:
            user_features: dict consumed by RecommendationEngine._encode_user
            top_k: final result size (overrides config.final_top_k)
            candidate_metadata: optional per-candidate metadata for re-ranking
                                (channel_id, category_id, upload_age_days)

        Returns:
            {
              "video_ids": [...],
              "scores": [...],
              "latency": {"stage1_ms": ..., "stage2_ms": ..., "stage3_ms": ..., "total_ms": ...}
            }
        """
        final_k = top_k or self.config.final_top_k
        total_start = time.perf_counter()

        # ── Stage 1: Candidate generation ─────────────────────────────
        t0 = time.perf_counter()
        user_embedding = self.engine._encode_user(user_features)
        query = user_embedding.reshape(1, -1)

        candidate_ids, stage1_scores = self.engine.index.search(
            query, self.config.stage1_top_k
        )
        candidate_ids = candidate_ids[0]  # (stage1_top_k,)
        stage1_scores = stage1_scores[0]
        stage1_ms = (time.perf_counter() - t0) * 1000

        # Retrieve candidate embeddings for stages 2 & 3
        candidate_embeddings = self._lookup_embeddings(candidate_ids)

        # ── Stage 2: Cross-feature scoring ────────────────────────────
        t1 = time.perf_counter()
        user_tensor = torch.from_numpy(user_embedding).float().to(self.config.scorer_device)
        cand_tensor = torch.from_numpy(candidate_embeddings).float().to(self.config.scorer_device)

        refined_scores = self.scorer(user_tensor, cand_tensor).cpu().numpy()

        # Keep top stage2_top_k
        stage2_k = min(self.config.stage2_top_k, len(refined_scores))
        top2_indices = np.argpartition(-refined_scores, stage2_k)[:stage2_k]
        top2_indices = top2_indices[np.argsort(-refined_scores[top2_indices])]

        candidate_ids = candidate_ids[top2_indices]
        refined_scores = refined_scores[top2_indices]
        candidate_embeddings = candidate_embeddings[top2_indices]
        stage2_ms = (time.perf_counter() - t1) * 1000

        # Subset metadata to match Stage 2 output
        stage2_meta = None
        if candidate_metadata:
            stage2_meta = [candidate_metadata[i] for i in top2_indices]

        # ── Stage 3: Diversity re-ranking ─────────────────────────────
        t2 = time.perf_counter()
        final_ids, final_scores = self.reranker.rerank(
            candidate_ids,
            refined_scores,
            candidate_embeddings,
            top_k=final_k,
            metadata=stage2_meta,
        )
        stage3_ms = (time.perf_counter() - t2) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # Record latencies
        latency = {
            "stage1_ms": round(stage1_ms, 3),
            "stage2_ms": round(stage2_ms, 3),
            "stage3_ms": round(stage3_ms, 3),
            "total_ms": round(total_ms, 3),
        }
        for k, v in latency.items():
            self._stage_latencies[k].append(v)

        return {
            "video_ids": final_ids.tolist(),
            "scores": final_scores.tolist(),
            "latency": latency,
        }

    def get_latency_stats(self) -> Dict:
        """Percentile latency breakdown across all stages."""
        stats = {}
        for key, samples in self._stage_latencies.items():
            if not samples:
                continue
            arr = np.array(samples)
            stats[key] = {
                "p50": round(float(np.percentile(arr, 50)), 3),
                "p95": round(float(np.percentile(arr, 95)), 3),
                "p99": round(float(np.percentile(arr, 99)), 3),
                "mean": round(float(arr.mean()), 3),
                "count": len(samples),
            }
        return stats

    # ── internal helpers ──────────────────────────────────────────────────

    def _lookup_embeddings(self, video_ids: np.ndarray) -> np.ndarray:
        """
        Retrieve pre-computed embeddings for a set of video IDs.

        Falls back to reconstructing from the FAISS index if available,
        otherwise uses the brute-force embedding matrix.
        """
        idx = self.engine.index
        try:
            if hasattr(idx, "index") and idx.index is not None:
                import faiss as _faiss

                # Map video IDs to FAISS internal IDs (same order for flat/IVF)
                id_to_pos = {vid: pos for pos, vid in enumerate(idx.video_ids)}
                positions = np.array(
                    [id_to_pos.get(int(v), 0) for v in video_ids], dtype=np.int64
                )
                # Reconstruct from index
                inner = idx.index
                if hasattr(inner, "reconstruct_n"):
                    # For flat indices, reconstruct in batch
                    all_emb = inner.reconstruct_n(0, inner.ntotal)
                    return all_emb[positions]
                # Fallback: reconstruct one-by-one
                dim = self.config.scorer_embedding_dim
                out = np.zeros((len(positions), dim), dtype=np.float32)
                for i, pos in enumerate(positions):
                    out[i] = inner.reconstruct(int(pos))
                return out
        except Exception:
            pass

        # Numpy brute-force fallback
        if hasattr(idx, "embeddings") and idx.embeddings is not None:
            id_to_pos = {vid: pos for pos, vid in enumerate(idx.video_ids)}
            positions = np.array(
                [id_to_pos.get(int(v), 0) for v in video_ids], dtype=np.int64
            )
            return idx.embeddings[positions]

        # Last resort: zeros (scorer still works, just less precise)
        logger.warning("Could not look up candidate embeddings; using zeros.")
        return np.zeros(
            (len(video_ids), self.config.scorer_embedding_dim), dtype=np.float32
        )
