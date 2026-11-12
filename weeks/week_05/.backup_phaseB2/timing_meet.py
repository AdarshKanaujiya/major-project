"""
weeks/week_05/benchmarking/timing_meet.py

SENTINEL-RS Week 5 — Retrieval Pipeline Latency Benchmarker.

Provides structured benchmarking functions to measure end-to-end retrieval
pipeline latency components:
  - TwoTowerNetwork SAR/MS encoder inference
  - FAISS IndexFlatIP gallery search
  - Full query pipeline (preprocess + encode + search)

DEPRECATION NOTICE — PlaceholderEncoder self-test:
  The original __main__ block ran a toy self-test using a 'PlaceholderEncoder'
  (a small nn.Linear operating on (2, 120, 120) random tensors) and a
  placeholder matrix-multiply retrieval function.

  ⚠️  That test does NOT benchmark the actual retrieval system. It operates
  on wrong input dimensions (120x120 vs canonical 224x224), a random 64-D
  output (vs canonical 512-D), and a numpy dot-product (vs FAISS IndexFlatIP).

  The PlaceholderEncoder self-test is deprecated and replaced by the
  `TwoTowerBenchmark` class below, which correctly targets:
    - TwoTowerNetwork (from weeks.week_04.models.two_tower_jainam)
    - Canonical 2-channel SAR and 3-channel MS inputs at 224×224
    - FAISS IndexFlatIP with 512-D embeddings

CLI Usage:
    python timing_meet.py --mode sar --n-queries 20 --n-warmup 5 --iterations 10
    python timing_meet.py --mode ms  --n-queries 20 --output-dir results/local_pc/week_05/benchmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.config.settings import RESULTS_DIR
except ImportError:
    RESULTS_DIR = PROJECT_ROOT / "results" / "local_pc"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from weeks.week_04.models.two_tower_jainam import TwoTowerNetwork
except ImportError:
    TwoTowerNetwork = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("W5TimingBenchmark")

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "week_05" / "benchmarks"


# ============================================================
# DEPRECATED PLACEHOLDER SELF-TEST — PRESERVED FOR PROVENANCE
# ============================================================

def _legacy_placeholder_selftest_warning() -> None:
    """
    Issues a deprecation warning for the original PlaceholderEncoder self-test.

    The original timing_meet.py ran a __main__ self-test using:
      class PlaceholderEncoder(nn.Module): nn.Linear(2 * 120 * 120, 64)
    and a numpy dot-product retrieval function.

    ⚠ That test does NOT represent the actual TwoTowerNetwork (512-D, 224×224).
    It is a toy API-verification harness.

    Use TwoTowerBenchmark instead.
    """
    warnings.warn(
        "\n"
        "[DEPRECATED] The PlaceholderEncoder self-test in timing_meet.py uses:\n"
        "  - Input shape: (2, 120, 120) — WRONG (canonical is (2, 224, 224))\n"
        "  - Output dim: 64            — WRONG (canonical is 512)\n"
        "  - Retrieval fn: numpy dot   — NOT FAISS IndexFlatIP\n"
        "This benchmark does not represent the SENTINEL-RS retrieval pipeline.\n"
        "Use TwoTowerBenchmark (--mode sar/ms) for valid benchmarks.",
        DeprecationWarning,
        stacklevel=2,
    )


# ============================================================
# UTILITY — CUDA-SYNCHRONIZED TIMER
# ============================================================

def get_sync_time() -> float:
    """Returns perf_counter time, synchronizing CUDA ops first if GPU is active."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# ============================================================
# LOWER-LEVEL TIMING FUNCTIONS (preserved from original API)
# ============================================================

def measure_inference_timing(
    model: nn.Module,
    dataloader: Any,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measures batch and single-image inference latencies and throughput.
    Accepts any DataLoader yielding (tensor, ...) tuples.

    Returns:
        Dict with batch_inference_ms, single_inference_ms, throughput_samples_per_second.
    """
    if torch is None:
        raise RuntimeError("[BLOCKED BY ENVIRONMENT] PyTorch required.")

    model_device = torch.device(device)
    model.to(model_device)
    model.eval()

    total_samples = 0
    batch_latencies = []
    single_latencies = []

    logger.info(f"Running inference benchmarking on device: {device}...")

    # Warm-up
    logger.info("Performing warm-up iteration...")
    sample_batch = next(iter(dataloader))
    inputs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
    inputs = inputs.to(model_device)
    with torch.no_grad():
        _ = model(inputs)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(model_device)
            b_size = inputs.size(0)
            total_samples += b_size

            t_start = get_sync_time()
            _ = model(inputs)
            t_end = get_sync_time()
            batch_latencies.append((t_end - t_start) * 1000.0)

            t_single_start = get_sync_time()
            for idx in range(min(5, b_size)):
                single_input = inputs[idx: idx + 1]
                _ = model(single_input)
            t_single_end = get_sync_time()
            single_latencies.append(((t_single_end - t_single_start) * 1000.0) / min(5, b_size))

    avg_batch_ms = float(np.mean(batch_latencies))
    avg_single_ms = float(np.mean(single_latencies))
    total_elapsed_sec = sum(batch_latencies) / 1000.0
    throughput = total_samples / total_elapsed_sec if total_elapsed_sec > 0 else 0.0

    return {
        "batch_inference_ms": avg_batch_ms,
        "single_inference_ms": avg_single_ms,
        "throughput_samples_per_second": throughput,
        "total_samples_benchmarked": total_samples,
    }


def measure_retrieval_timing(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    retrieval_fn: Any,
) -> Dict[str, float]:
    """
    Measures execution latency of a retrieval search function.

    Args:
        query_embeddings: Query vectors (N_queries, D)
        database_embeddings: Reference database vectors (N_db, D)
        retrieval_fn: Callable(queries, database) -> results

    Returns:
        Dict with total_retrieval_time_ms, average_query_latency_ms.
    """
    if not callable(retrieval_fn):
        raise TypeError("retrieval_fn must be callable.")

    logger.info("Benchmarking retrieval query latency...")
    _ = retrieval_fn(query_embeddings[:1], database_embeddings)

    n_queries = query_embeddings.shape[0]
    t_start = time.perf_counter()
    _ = retrieval_fn(query_embeddings, database_embeddings)
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0
    avg_query_ms = elapsed_ms / n_queries if n_queries > 0 else 0.0

    return {
        "total_retrieval_time_ms": elapsed_ms,
        "average_query_latency_ms": avg_query_ms,
    }


# ============================================================
# CANONICAL BENCHMARK: TwoTowerNetwork + FAISS IndexFlatIP
# ============================================================

class TwoTowerBenchmark:
    """
    Benchmarks TwoTowerNetwork SAR/MS inference and FAISS IndexFlatIP search
    with correct canonical dimensions (2-ch SAR or 3-ch MS, 224×224, 512-D).
    """

    def __init__(
        self,
        mode: str = "sar",
        device: str = "cpu",
        n_gallery: int = 4500,
        n_queries: int = 20,
        n_warmup: int = 5,
        iterations: int = 10,
    ) -> None:
        if torch is None:
            raise RuntimeError("[BLOCKED BY ENVIRONMENT] PyTorch is required.")
        if TwoTowerNetwork is None:
            raise RuntimeError("[BLOCKED BY ENVIRONMENT] TwoTowerNetwork could not be imported.")

        self.mode = mode.strip().lower()
        assert self.mode in ("sar", "ms"), f"mode must be 'sar' or 'ms', got {self.mode!r}"
        self.device = torch.device(device)
        self.n_gallery = n_gallery
        self.n_queries = n_queries
        self.n_warmup = n_warmup
        self.iterations = iterations

        self.in_channels = 2 if self.mode == "sar" else 3

    def _build_model(self) -> nn.Module:
        model = TwoTowerNetwork(pretrained=False)
        model.to(self.device)
        model.eval()
        return model

    def _build_faiss_index(self) -> faiss.Index:
        if faiss is None:
            raise RuntimeError("[BLOCKED BY ENVIRONMENT] FAISS is required.")
        gallery_embs = np.random.randn(self.n_gallery, 512).astype(np.float32)
        gallery_embs /= np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-8
        index = faiss.IndexFlatIP(512)
        index.add(gallery_embs)
        return index

    def _make_query_batch(self, batch_size: int = 1) -> torch.Tensor:
        return torch.randn(batch_size, self.in_channels, 224, 224).to(self.device)

    def _encode_one(self, model: nn.Module, batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if self.mode == "sar":
                # Pass MS as zeros to satisfy forward signature
                dummy_ms = torch.zeros(batch.size(0), 3, 224, 224, device=self.device)
                sar_emb, _ = model(batch, dummy_ms)
                emb = F.normalize(sar_emb, p=2, dim=1)
            else:
                dummy_sar = torch.zeros(batch.size(0), 2, 224, 224, device=self.device)
                _, ms_emb = model(dummy_sar, batch)
                emb = F.normalize(ms_emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def run(self) -> Dict[str, Any]:
        """Execute all benchmark measurements and return structured results."""
        print("=" * 60)
        print(f"TwoTowerBenchmark: mode={self.mode.upper()}, device={self.device}")
        print(f"  Gallery size : {self.n_gallery}")
        print(f"  Queries      : {self.n_queries}")
        print(f"  Warm-up runs : {self.n_warmup}")
        print(f"  Iterations   : {self.iterations}")
        print("=" * 60)

        # 1. Build model
        t0 = time.perf_counter()
        model = self._build_model()
        model_init_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[1] Model init           : {model_init_ms:.2f} ms")

        # 2. Build FAISS index
        t0 = time.perf_counter()
        index = self._build_faiss_index()
        index_build_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[2] FAISS index build    : {index_build_ms:.2f} ms (synthetic gallery)")

        # 3. Warm-up runs
        print(f"[3] Warm-up ({self.n_warmup} runs)...")
        query_batch = self._make_query_batch(1)
        for _ in range(self.n_warmup):
            self._encode_one(model, query_batch)

        # 4. Inference latency benchmark
        inference_times = []
        for _ in range(self.iterations):
            t0 = get_sync_time()
            self._encode_one(model, query_batch)
            inference_times.append((get_sync_time() - t0) * 1000.0)

        inference_mean = float(np.mean(inference_times))
        inference_std = float(np.std(inference_times))
        inference_p50 = float(np.percentile(inference_times, 50))
        inference_p95 = float(np.percentile(inference_times, 95))
        print(f"[4] Single-query inference (over {self.iterations} iters):")
        print(f"    mean={inference_mean:.2f} ms  std={inference_std:.2f} ms  p50={inference_p50:.2f} ms  p95={inference_p95:.2f} ms")

        # 5. FAISS search latency benchmark
        query_embs = np.random.randn(self.n_queries, 512).astype(np.float32)
        query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8

        faiss_times = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            _, _ = index.search(query_embs[:1], 10)
            faiss_times.append((time.perf_counter() - t0) * 1000.0)

        faiss_mean = float(np.mean(faiss_times))
        faiss_std = float(np.std(faiss_times))
        faiss_p50 = float(np.percentile(faiss_times, 50))
        faiss_p95 = float(np.percentile(faiss_times, 95))
        print(f"[5] FAISS single-query search (over {self.iterations} iters):")
        print(f"    mean={faiss_mean:.2f} ms  std={faiss_std:.2f} ms  p50={faiss_p50:.2f} ms  p95={faiss_p95:.2f} ms")

        # 6. End-to-end pipeline latency (preprocess + encode + search)
        pipeline_times = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            q = self._make_query_batch(1)
            emb = self._encode_one(model, q)
            _, _ = index.search(emb, 10)
            pipeline_times.append((time.perf_counter() - t0) * 1000.0)

        pipeline_mean = float(np.mean(pipeline_times))
        pipeline_std = float(np.std(pipeline_times))
        pipeline_p50 = float(np.percentile(pipeline_times, 50))
        pipeline_p95 = float(np.percentile(pipeline_times, 95))
        print(f"[6] End-to-end pipeline latency (over {self.iterations} iters):")
        print(f"    mean={pipeline_mean:.2f} ms  std={pipeline_std:.2f} ms  p50={pipeline_p50:.2f} ms  p95={pipeline_p95:.2f} ms")
        print("=" * 60)

        return {
            "mode": self.mode,
            "device": str(self.device),
            "n_gallery": self.n_gallery,
            "n_queries": self.n_queries,
            "n_warmup": self.n_warmup,
            "iterations": self.iterations,
            "model_init_ms": round(model_init_ms, 3),
            "faiss_index_build_ms": round(index_build_ms, 3),
            "note_index": "FAISS index built on synthetic random gallery (not from real checkpoint)",
            "inference": {
                "mean_ms": round(inference_mean, 3),
                "std_ms": round(inference_std, 3),
                "p50_ms": round(inference_p50, 3),
                "p95_ms": round(inference_p95, 3),
            },
            "faiss_search": {
                "mean_ms": round(faiss_mean, 3),
                "std_ms": round(faiss_std, 3),
                "p50_ms": round(faiss_p50, 3),
                "p95_ms": round(faiss_p95, 3),
            },
            "end_to_end_pipeline": {
                "mean_ms": round(pipeline_mean, 3),
                "std_ms": round(pipeline_std, 3),
                "p50_ms": round(pipeline_p50, 3),
                "p95_ms": round(pipeline_p95, 3),
            },
            "note": (
                "Inference and FAISS timings use untrained (random-weight) TwoTowerNetwork. "
                "Checkpoint not loaded because best_model-v1.ckpt binary is unavailable in this environment. "
                "These timings reflect architectural structure cost (parameter count, forward pass ops) "
                "and are valid for latency estimation purposes."
            ),
        }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SENTINEL-RS Week 5 — TwoTower Retrieval Latency Benchmark"
    )
    parser.add_argument(
        "--mode",
        choices=["sar", "ms"],
        default="sar",
        help="Query modality tower to benchmark (sar or ms)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument(
        "--n-gallery",
        type=int,
        default=4500,
        help="Synthetic gallery size (default: 4500 to match test set)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=20,
        help="Number of query vectors to benchmark",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Number of warm-up forward passes before timing",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timing iterations for statistical reporting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save benchmark results JSON",
    )
    args = parser.parse_args()

    if torch is None:
        print("[BLOCKED BY ENVIRONMENT] PyTorch is not installed in the active environment.")
        print("Please run this script in an environment with PyTorch installed.")
        sys.exit(0)

    if faiss is None:
        print("[BLOCKED BY ENVIRONMENT] FAISS is not installed in the active environment.")
        print("Please run this script in an environment with faiss-cpu installed.")
        sys.exit(0)

    benchmark = TwoTowerBenchmark(
        mode=args.mode,
        device=args.device,
        n_gallery=args.n_gallery,
        n_queries=args.n_queries,
        n_warmup=args.n_warmup,
        iterations=args.iterations,
    )
    results = benchmark.run()

    # Save to new run directory (non-overwriting)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%dT%H%M%S")
    out_file = args.output_dir / f"timing_{args.mode}_{run_ts}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
