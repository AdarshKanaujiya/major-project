"""
Week 5 - Trained Cross-Modal Retrieval Evaluation (Scientific Revision).

Evaluates:
    1. SAR -> MS
    2. MS -> SAR

Metrics:
    Recall@1
    Recall@5
    Recall@10
    Average retrieval time per query

Ground truth:
    Paired patch IDs from test_split.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.config.settings import RESULTS_DIR, SPLITS_DIR
except ImportError:
    RESULTS_DIR = PROJECT_ROOT / "results" / "local_pc"
    SPLITS_DIR = RESULTS_DIR / "week_02" / "splits"

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_TEST_CSV = SPLITS_DIR / "test_split.csv"
DEFAULT_EMBEDDINGS_DIR = RESULTS_DIR / "week_05" / "trained_embeddings"
DEFAULT_INDEX_DIR = RESULTS_DIR / "week_05" / "faiss_indices"
DEFAULT_REPORT_DIR = RESULTS_DIR / "week_05" / "evaluation" / "runs"

DEFAULT_SAR_EMBEDDINGS = DEFAULT_EMBEDDINGS_DIR / "trained_test_sar_embeddings.npy"
DEFAULT_MS_EMBEDDINGS = DEFAULT_EMBEDDINGS_DIR / "trained_test_ms_embeddings.npy"
DEFAULT_SAR_INDEX = DEFAULT_INDEX_DIR / "sar_gallery.index"
DEFAULT_MS_INDEX = DEFAULT_INDEX_DIR / "ms_gallery.index"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "trained_cross_modal_results.csv"

TOP_K = 10
EXPECTED_SAMPLES = 4500


# ============================================================
# LOAD TEST PATCH IDS
# ============================================================

def load_test_patch_ids(csv_path: Path | None = None) -> list[str]:
    """Load test patch IDs in the same order as the embeddings."""
    target_csv = Path(csv_path) if csv_path else DEFAULT_TEST_CSV

    if not target_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{target_csv}"
        )

    df = pd.read_csv(target_csv)

    if "patch_id" not in df.columns:
        raise ValueError(
            "test_split.csv does not contain 'patch_id' column."
        )

    patch_ids = (
        df["patch_id"]
        .astype(str)
        .tolist()
    )

    if len(patch_ids) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} test patch IDs, "
            f"found {len(patch_ids)}."
        )

    if len(set(patch_ids)) != EXPECTED_SAMPLES:
        raise ValueError(
            "Test patch IDs are not unique."
        )

    return patch_ids


# ============================================================
# VALIDATION
# ============================================================

def validate_embeddings(
    embeddings: np.ndarray,
    name: str,
) -> None:

    if embeddings.shape != (
        EXPECTED_SAMPLES,
        512,
    ):
        raise ValueError(
            f"{name} shape is {embeddings.shape}; "
            f"expected ({EXPECTED_SAMPLES}, 512)."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


# ============================================================
# CROSS-MODAL EVALUATION
# ============================================================

def evaluate_direction(
    query_embeddings: np.ndarray,
    gallery_index: faiss.Index,
    query_patch_ids: list[str],
    gallery_patch_ids: list[str],
    direction_name: str,
) -> dict[str, float]:

    print()
    print("=" * 70)
    print(direction_name)
    print("=" * 70)

    start_time = time.perf_counter()

    distances, indices = gallery_index.search(
        query_embeddings,
        TOP_K,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    total_queries = len(query_patch_ids)

    recall_counts = {
        1: 0,
        5: 0,
        10: 0,
    }

    for query_index in range(total_queries):

        query_patch_id = query_patch_ids[
            query_index
        ]

        retrieved_indices = indices[
            query_index
        ]

        retrieved_patch_ids = [
            gallery_patch_ids[index]
            for index in retrieved_indices
            if index >= 0
        ]

        for k in recall_counts:

            top_k_ids = retrieved_patch_ids[:k]

            if query_patch_id in top_k_ids:
                recall_counts[k] += 1

    results = {
        "Mode": direction_name,
        "Recall@1": (
            recall_counts[1] / total_queries
        ) * 100.0,
        "Recall@5": (
            recall_counts[5] / total_queries
        ) * 100.0,
        "Recall@10": (
            recall_counts[10] / total_queries
        ) * 100.0,
        "Avg Retrieval Time (ms)": (
            elapsed_seconds / total_queries
        ) * 1000.0,
    }

    print(
        f"Queries             : {total_queries:,}"
    )

    print(
        f"Recall@1            : "
        f"{results['Recall@1']:.4f}%"
    )

    print(
        f"Recall@5            : "
        f"{results['Recall@5']:.4f}%"
    )

    print(
        f"Recall@10           : "
        f"{results['Recall@10']:.4f}%"
    )

    print(
        f"Avg retrieval time  : "
        f"{results['Avg Retrieval Time (ms)']:.4f} ms/query"
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 5 Cross-Modal Retrieval Evaluation (Recall@1, 5, 10)"
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help="Path to test_split.csv",
    )
    parser.add_argument(
        "--sar-embeddings",
        type=Path,
        default=DEFAULT_SAR_EMBEDDINGS,
        help="Path to trained test SAR embeddings (.npy)",
    )
    parser.add_argument(
        "--ms-embeddings",
        type=Path,
        default=DEFAULT_MS_EMBEDDINGS,
        help="Path to trained test MS embeddings (.npy)",
    )
    parser.add_argument(
        "--sar-index",
        type=Path,
        default=DEFAULT_SAR_INDEX,
        help="Path to SAR FAISS gallery index",
    )
    parser.add_argument(
        "--ms-index",
        type=Path,
        default=DEFAULT_MS_INDEX,
        help="Path to MS FAISS gallery index",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (defaults to a separate run directory to protect historical records)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WEEK 5 — TRAINED CROSS-MODAL RETRIEVAL (SCIENTIFIC EVALUATION)")
    print("=" * 70)
    print()

    if faiss is None:
        raise ImportError(
            "faiss is not installed in the active Python environment.\n"
            "Please install faiss-cpu to execute retrieval evaluation:\n"
            "    pip install faiss-cpu"
        )

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    print("Loading trained test embeddings...")

    sar_embeddings = np.load(
        args.sar_embeddings
    ).astype(np.float32)

    ms_embeddings = np.load(
        args.ms_embeddings
    ).astype(np.float32)

    validate_embeddings(
        sar_embeddings,
        "SAR embeddings",
    )

    validate_embeddings(
        ms_embeddings,
        "MS embeddings",
    )

    print(
        f"[OK] SAR embeddings: "
        f"{sar_embeddings.shape}"
    )

    print(
        f"[OK] MS embeddings: "
        f"{ms_embeddings.shape}"
    )

    # --------------------------------------------------------
    # LOAD PATCH IDS
    # --------------------------------------------------------

    patch_ids = load_test_patch_ids(args.test_csv)

    print(
        f"[OK] Test patch IDs: "
        f"{len(patch_ids):,}"
    )

    # --------------------------------------------------------
    # LOAD FAISS INDICES
    # --------------------------------------------------------

    if not args.sar_index.exists():
        raise FileNotFoundError(
            f"SAR FAISS index not found:\n"
            f"{args.sar_index}"
        )

    if not args.ms_index.exists():
        raise FileNotFoundError(
            f"MS FAISS index not found:\n"
            f"{args.ms_index}"
        )

    sar_index = faiss.read_index(
        str(args.sar_index)
    )

    ms_index = faiss.read_index(
        str(args.ms_index)
    )

    if sar_index.ntotal != EXPECTED_SAMPLES:
        raise ValueError(
            f"SAR index does not contain {EXPECTED_SAMPLES} vectors: got {sar_index.ntotal}"
        )

    if ms_index.ntotal != EXPECTED_SAMPLES:
        raise ValueError(
            f"MS index does not contain {EXPECTED_SAMPLES} vectors: got {ms_index.ntotal}"
        )

    print(
        f"[OK] SAR gallery index: "
        f"{sar_index.ntotal:,} vectors"
    )

    print(
        f"[OK] MS gallery index: "
        f"{ms_index.ntotal:,} vectors"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Test CSV order must match embedding order.
    # --------------------------------------------------------

    # SAR -> MS
    sar_to_ms = evaluate_direction(
        query_embeddings=sar_embeddings,
        gallery_index=ms_index,
        query_patch_ids=patch_ids,
        gallery_patch_ids=patch_ids,
        direction_name="SAR -> MS",
    )

    # MS -> SAR
    ms_to_sar = evaluate_direction(
        query_embeddings=ms_embeddings,
        gallery_index=sar_index,
        query_patch_ids=patch_ids,
        gallery_patch_ids=patch_ids,
        direction_name="MS -> SAR",
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            sar_to_ms,
            ms_to_sar,
        ]
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        args.output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINED CROSS-MODAL EVALUATION COMPLETE")
    print("=" * 70)
    print()

    print(results_df.to_string(index=False))

    print()
    print(
        f"[OK] Results saved: {args.output_csv}"
    )


if __name__ == "__main__":
    main()