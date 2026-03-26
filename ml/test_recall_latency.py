"""
TwinTube Vector: Recall@100 & Latency Benchmark Suite.

Measures the trade-off between retrieval quality (Recall@K) and query
latency across multiple FAISS index configurations.  Ground truth is
computed via brute-force inner product, and each ANN index is scored
against it.

Sweep dimensions:
  - Index type: Flat, IVFFlat, IVFPQ, HNSW
  - nprobe:     1, 4, 16, 64, 128  (IVF variants)
  - HNSW efSearch: 32, 64, 128, 256
  - Corpus size: configurable (default 100k)

Targets (from CLAUDE.md):
  - Recall@100 >= 68%
  - P50 latency < 8ms, P99 < 12ms

Usage:
    python test_recall_latency.py                  # quick run (100k corpus)
    python test_recall_latency.py --num-videos 1000000 --num-queries 5000
"""

import argparse
import time
import sys
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure ml/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from model import create_model
from inference import InferenceConfig, GPUVectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_corpus(num_videos: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate L2-normalized random embeddings simulating a video corpus."""
    rng = np.random.RandomState(seed)
    embeddings = rng.randn(num_videos, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    return embeddings / norms


def generate_queries(num_queries: int, dim: int, seed: int = 99) -> np.ndarray:
    """Generate L2-normalized random query embeddings simulating users."""
    rng = np.random.RandomState(seed)
    queries = rng.randn(num_queries, dim).astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-8)
    return queries / norms


# ---------------------------------------------------------------------------
# Ground truth via brute-force
# ---------------------------------------------------------------------------

def brute_force_topk(
    queries: np.ndarray, corpus: np.ndarray, k: int
) -> np.ndarray:
    """
    Exact top-k retrieval via matrix multiplication.

    Returns:
        (num_queries, k) array of corpus indices
    """
    # Batch to avoid OOM for large corpora
    batch_size = 256
    n = len(queries)
    all_topk = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = queries[start:end] @ corpus.T  # (batch, num_videos)
        # argpartition is O(n) vs O(n log n) for argsort
        topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        # Sort the top-k by score descending
        for i in range(end - start):
            order = np.argsort(-sims[i, topk_idx[i]])
            topk_idx[i] = topk_idx[i][order]
        all_topk[start:end] = topk_idx

    return all_topk


# ---------------------------------------------------------------------------
# Recall@K computation
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    retrieved: np.ndarray, ground_truth: np.ndarray, k_values: List[int]
) -> Dict[str, float]:
    """
    Compute Recall@K for multiple K values.

    Args:
        retrieved:    (n, max_k) indices from ANN search
        ground_truth: (n, gt_k) indices from brute-force
        k_values:     list of K values to evaluate

    Returns:
        {"recall@10": 0.85, "recall@100": 0.72, ...}
    """
    results = {}
    n = len(retrieved)
    for k in k_values:
        hits = 0
        for i in range(n):
            gt_set = set(ground_truth[i].tolist())
            ret_set = set(retrieved[i, :k].tolist())
            hits += len(gt_set & ret_set)
        # Recall = hits / (n * min(k, gt_k))
        denom = n * min(k, ground_truth.shape[1])
        results[f"recall@{k}"] = hits / denom if denom > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# Index configurations to sweep
# ---------------------------------------------------------------------------

@dataclass
class IndexVariant:
    name: str
    index_type: str
    nprobe: int = 64
    ivf_nlist: int = 0
    pq_m: int = 32
    pq_nbits: int = 8
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128


def get_sweep_configs(num_videos: int) -> List[IndexVariant]:
    """Return a list of index configurations to benchmark."""
    configs = [
        IndexVariant(name="Flat (exact)", index_type="flat"),
    ]

    if not FAISS_AVAILABLE:
        return configs

    # IVFFlat with varying nprobe
    for nprobe in [1, 4, 16, 64, 128]:
        configs.append(IndexVariant(
            name=f"IVFFlat nprobe={nprobe}",
            index_type="ivfflat",
            nprobe=nprobe,
        ))

    # IVFPQ with varying nprobe
    for nprobe in [1, 4, 16, 64, 128]:
        configs.append(IndexVariant(
            name=f"IVFPQ nprobe={nprobe}",
            index_type="ivfpq",
            nprobe=nprobe,
        ))

    # HNSW with varying efSearch
    for ef in [32, 64, 128, 256]:
        configs.append(IndexVariant(
            name=f"HNSW efSearch={ef}",
            index_type="hnsw",
            hnsw_ef_search=ef,
        ))

    return configs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_index(
    variant: IndexVariant,
    corpus: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    warmup: int = 50,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build an index from `variant`, search `queries`, and return timing stats.

    Returns:
        (retrieved_ids, latency_stats)
    """
    num_videos, dim = corpus.shape

    config = InferenceConfig(
        embedding_dim=dim,
        num_videos=num_videos,
        nprobe=variant.nprobe,
        index_type=variant.index_type,
        ivf_nlist=variant.ivf_nlist,
        pq_m=variant.pq_m,
        pq_nbits=variant.pq_nbits,
        hnsw_m=variant.hnsw_m,
        hnsw_ef_construction=variant.hnsw_ef_construction,
        hnsw_ef_search=variant.hnsw_ef_search,
        use_gpu_index=False,  # benchmark on CPU for reproducibility
        use_quantization=False,
        use_redis_cache=False,
    )

    index = GPUVectorIndex(config)

    # The corpus is already normalized; build_index will re-normalize (idempotent
    # for unit vectors) so pass a copy to avoid mutating the shared array.
    video_ids = np.arange(num_videos, dtype=np.int64)
    index.build_index(corpus.copy(), video_ids)

    # Warmup
    for i in range(min(warmup, len(queries))):
        index.search(queries[i : i + 1], top_k)

    # Timed queries
    latencies = []
    all_retrieved = np.empty((len(queries), top_k), dtype=np.int64)

    for i in range(len(queries)):
        q = queries[i : i + 1]
        t0 = time.perf_counter()
        ids, _ = index.search(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_retrieved[i] = ids[0]

    lat = np.array(latencies)
    stats = {
        "p50_ms": round(float(np.percentile(lat, 50)), 4),
        "p95_ms": round(float(np.percentile(lat, 95)), 4),
        "p99_ms": round(float(np.percentile(lat, 99)), 4),
        "mean_ms": round(float(lat.mean()), 4),
    }

    return all_retrieved, stats


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    num_videos: int = 100_000,
    num_queries: int = 1_000,
    dim: int = 256,
    top_k: int = 100,
    k_values: List[int] = [10, 50, 100],
):
    """
    Run the complete recall-vs-latency benchmark.

    Prints a table of results and checks whether the 68% Recall@100 target
    is achievable within the P50 <8ms budget.
    """
    logger.info(f"Corpus: {num_videos:,} videos, dim={dim}")
    logger.info(f"Queries: {num_queries:,}, top_k={top_k}")

    # Generate data
    logger.info("Generating corpus...")
    corpus = generate_corpus(num_videos, dim)
    queries = generate_queries(num_queries, dim)

    # Ground truth
    logger.info("Computing brute-force ground truth...")
    t0 = time.perf_counter()
    gt = brute_force_topk(queries, corpus, top_k)
    gt_time = time.perf_counter() - t0
    logger.info(f"Ground truth computed in {gt_time:.2f}s")

    # Sweep
    configs = get_sweep_configs(num_videos)
    results = []

    header = f"{'Index':<30} {'P50 ms':>8} {'P99 ms':>8}"
    for k in k_values:
        header += f" {'R@' + str(k):>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for variant in configs:
        logger.info(f"Benchmarking: {variant.name}")
        try:
            retrieved, latency = benchmark_index(
                variant, corpus, queries, top_k
            )
            recall = compute_recall_at_k(retrieved, gt, k_values)
        except Exception as e:
            logger.warning(f"  SKIP {variant.name}: {e}")
            continue

        row = {
            "name": variant.name,
            **latency,
            **recall,
        }
        results.append(row)

        line = f"{variant.name:<30} {latency['p50_ms']:>8.3f} {latency['p99_ms']:>8.3f}"
        for k in k_values:
            line += f" {recall[f'recall@{k}']:>8.1%}"
        print(line)

    print("=" * len(header))

    # Pareto analysis: which configs meet BOTH targets?
    print("\n--- Pareto-optimal configs (Recall@100 >= 68% AND P50 < 8ms) ---")
    pareto = [
        r for r in results
        if r.get("recall@100", 0) >= 0.68 and r["p50_ms"] < 8.0
    ]
    if pareto:
        # Sort by recall descending
        pareto.sort(key=lambda r: -r["recall@100"])
        for r in pareto:
            print(f"  {r['name']:<30}  R@100={r['recall@100']:.1%}  P50={r['p50_ms']:.3f}ms")
    else:
        print("  (none — try increasing nprobe or switching index type)")

    # Best recall@100 overall
    if results:
        best = max(results, key=lambda r: r.get("recall@100", 0))
        print(f"\nBest Recall@100: {best['recall@100']:.1%} ({best['name']})")
        fastest = min(results, key=lambda r: r["p50_ms"])
        print(f"Fastest P50:     {fastest['p50_ms']:.3f}ms ({fastest['name']})")

    return results


# ---------------------------------------------------------------------------
# Multi-stage pipeline end-to-end test
# ---------------------------------------------------------------------------

def test_multi_stage_pipeline(num_videos: int = 50_000, num_queries: int = 200):
    """
    End-to-end test of the three-stage pipeline:
      Stage 1 (FAISS) → Stage 2 (scorer) → Stage 3 (re-ranker)

    Verifies latency targets and prints per-stage breakdown.
    """
    from inference import RecommendationEngine
    from multi_stage_ranker import MultiStageRecommender, PipelineConfig

    logger.info(f"Testing multi-stage pipeline ({num_videos:,} videos, {num_queries} queries)")

    config = InferenceConfig(
        num_videos=num_videos,
        use_quantization=False,
        use_redis_cache=False,
        use_gpu_index=False,
        index_type="ivfpq",
        nprobe=32,
    )
    engine = RecommendationEngine(config)

    # Build synthetic index
    video_features = []
    for i in range(1, num_videos + 1):
        video_features.append({
            "video_id": i,
            "channel_id": i % 5000 + 1,
            "category_id": i % 50 + 1,
            "title_embedding": np.random.randn(384).tolist(),
            "numerical": np.random.rand(5).tolist(),
        })
    engine.build_video_index(video_features)

    pipeline = MultiStageRecommender(
        PipelineConfig(
            stage1_top_k=500,
            stage2_top_k=100,
            final_top_k=20,
        ),
        engine,
    )

    # Warmup
    for _ in range(20):
        pipeline.recommend({
            "watch_history": np.random.randint(1, num_videos, size=20).tolist(),
            "watch_times": np.random.exponential(300, size=20).tolist(),
            "engagement": np.random.rand(20, 3).tolist(),
        })

    # Benchmark
    for _ in range(num_queries):
        pipeline.recommend({
            "watch_history": np.random.randint(1, num_videos, size=20).tolist(),
            "watch_times": np.random.exponential(300, size=20).tolist(),
            "engagement": np.random.rand(20, 3).tolist(),
        })

    stats = pipeline.get_latency_stats()

    print("\n" + "=" * 60)
    print("MULTI-STAGE PIPELINE LATENCY BREAKDOWN")
    print("=" * 60)
    for stage, st in stats.items():
        print(f"  {stage:<12}  P50={st['p50']:.3f}ms  P95={st['p95']:.3f}ms  P99={st['p99']:.3f}ms")
    print("=" * 60)

    total = stats.get("total_ms", {})
    p50 = total.get("p50", float("inf"))
    p99 = total.get("p99", float("inf"))
    print(f"\nTarget P50 <8ms:  {'PASS' if p50 < 8 else 'FAIL'} ({p50:.3f}ms)")
    print(f"Target P99 <12ms: {'PASS' if p99 < 12 else 'FAIL'} ({p99:.3f}ms)")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TwinTube Recall@100 & Latency Benchmark")
    parser.add_argument("--num-videos", type=int, default=100_000, help="Corpus size")
    parser.add_argument("--num-queries", type=int, default=1_000, help="Number of queries")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K for recall")
    parser.add_argument("--multi-stage", action="store_true", help="Also run multi-stage pipeline test")
    args = parser.parse_args()

    results = run_benchmark(
        num_videos=args.num_videos,
        num_queries=args.num_queries,
        dim=args.dim,
        top_k=args.top_k,
    )

    if args.multi_stage:
        test_multi_stage_pipeline(
            num_videos=min(args.num_videos, 50_000),
            num_queries=min(args.num_queries, 200),
        )


if __name__ == "__main__":
    main()
