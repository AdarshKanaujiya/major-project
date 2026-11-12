"""
Week 5 - Trained Same-Modal Retrieval Evaluation (Scientific Revision).

Evaluates:
    1. SAR -> SAR
    2. MS -> MS

Metrics (Reconciled Phase A.1):
    HitRate@1, HitRate@5, HitRate@10:
        Fraction of queries where at least one retrieved gallery patch
        shares >= 1 land-cover label with the query (formerly reported as Recall@K).
    mAP_ret@10:
        Retrieved-normalized mean average precision at K=10, where precision
        is normalized by the count of relevant items retrieved in top-K
        (sum_{k=1}^K rel(k)). Formerly reported as mAP.
    Average retrieval time per query (ms)

Same-modal relevance:
    A gallery item is relevant when it shares at least one
    land-cover label with the query.

The query item itself is excluded from retrieval (K+1 candidate search).
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
# DEFAULT PATHS (Canonical Integration)
# ============================================================

DEFAULT_TEST_CSV = SPLITS_DIR / "test_split.csv"
DEFAULT_EMBEDDINGS_DIR = RESULTS_DIR / "week_05" / "trained_embeddings"
DEFAULT_INDEX_DIR = RESULTS_DIR / "week_05" / "faiss_indices"
DEFAULT_REPORT_DIR = RESULTS_DIR / "week_05" / "evaluation" / "runs"

DEFAULT_SAR_EMBEDDINGS = DEFAULT_EMBEDDINGS_DIR / "trained_test_sar_embeddings.npy"
DEFAULT_MS_EMBEDDINGS = DEFAULT_EMBEDDINGS_DIR / "trained_test_ms_embeddings.npy"
DEFAULT_SAR_INDEX = DEFAULT_INDEX_DIR / "sar_gallery.index"
DEFAULT_MS_INDEX = DEFAULT_INDEX_DIR / "ms_gallery.index"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "trained_same_modal_results.csv"

TOP_K = 10
EXPECTED_SAMPLES = 4500
EXPECTED_DIMENSION = 512


# ============================================================
# LOAD TEST METADATA
# ============================================================

def load_test_metadata(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load test metadata and validate required columns."""
    target_csv = Path(csv_path) if csv_path else DEFAULT_TEST_CSV

    if not target_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{target_csv}"
        )

    df = pd.read_csv(target_csv)

    required_columns = {
        "patch_id",
        "labels",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} test samples, "
            f"found {len(df)}."
        )

    df["patch_id"] = (
        df["patch_id"]
        .astype(str)
    )

    return df


# ============================================================
# LABEL PARSING
# ============================================================

def parse_labels(value) -> set[str]:
    """
    Convert the labels column into a set of labels.

    Supports common representations such as:
        "a;b;c"
        "['a', 'b', 'c']"
        "a, b, c"
    """

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    # Handle Python-list-like representation.
    if (
        text.startswith("[")
        and text.endswith("]")
    ):
        text = text[1:-1]

    text = text.replace(
        "'",
        "",
    ).replace(
        '"',
        "",
    )

    if ";" in text:
        parts = text.split(";")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = text.split()

    return {
        part.strip()
        for part in parts
        if part.strip()
    }


# ============================================================
# RELEVANCE
# ============================================================

def build_label_sets(
    metadata: pd.DataFrame,
) -> list[set[str]]:
    """Build a label set for every test patch."""

    return [
        parse_labels(value)
        for value in metadata["labels"]
    ]


def is_relevant(
    query_labels: set[str],
    gallery_labels: set[str],
) -> bool:
    """Return True when query and gallery share >= 1 label."""

    return bool(
        query_labels.intersection(
            gallery_labels
        )
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(
    path: Path,
    name: str,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} embeddings not found:\n{path}"
        )

    embeddings = np.load(path)

    if embeddings.shape != (
        EXPECTED_SAMPLES,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            f"{name} embeddings have shape "
            f"{embeddings.shape}; expected "
            f"({EXPECTED_SAMPLES}, {EXPECTED_DIMENSION})."
        )

    embeddings = embeddings.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name} embeddings contain NaN or Inf."
        )

    return embeddings


# ============================================================
# SAME-MODAL EVALUATION
# ============================================================

def evaluate_same_modal(
    embeddings: np.ndarray,
    index: faiss.Index,
    label_sets: list[set[str]],
    mode_name: str,
) -> dict[str, float]:

    print()
    print("=" * 70)
    print(mode_name)
    print("=" * 70)

    total_queries = len(embeddings)

    # We need one extra result because the query itself
    # occupies one gallery position and must be removed.
    search_k = TOP_K + 1

    start_time = time.perf_counter()

    distances, indices = index.search(
        embeddings,
        search_k,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    recall_counts = {
        1: 0,
        5: 0,
        10: 0,
    }

    average_precisions = []

    for query_index in range(total_queries):

        query_labels = label_sets[
            query_index
        ]

        retrieved_indices = []

        for gallery_index in indices[
            query_index
        ]:

            if gallery_index < 0:
                continue

            # Exclude the query itself.
            if gallery_index == query_index:
                continue

            retrieved_indices.append(
                gallery_index
            )

            if len(retrieved_indices) >= TOP_K:
                break

        # ----------------------------------------------------
        # Build relevance list
        # ----------------------------------------------------

        relevance = []

        for gallery_index in retrieved_indices:

            relevant = is_relevant(
                query_labels,
                label_sets[gallery_index],
            )

            relevance.append(
                1 if relevant else 0
            )

        # ----------------------------------------------------
        # Recall@K
        # ----------------------------------------------------

        for k in recall_counts:

            top_k_relevance = relevance[:k]

            if any(top_k_relevance):
                recall_counts[k] += 1

        # ----------------------------------------------------
        # Average Precision @ 10
        # ----------------------------------------------------

        total_relevant_in_top_k = sum(
            relevance
        )

        if total_relevant_in_top_k == 0:
            average_precisions.append(0.0)
            continue

        precision_sum = 0.0
        relevant_seen = 0

        for rank, relevant in enumerate(
            relevance,
            start=1,
        ):

            if relevant:
                relevant_seen += 1

                precision_at_rank = (
                    relevant_seen / rank
                )

                precision_sum += (
                    precision_at_rank
                )

        ap_at_10 = (
            precision_sum
            / total_relevant_in_top_k
        )

        average_precisions.append(
            ap_at_10
        )

    map_at_10 = (
        float(np.mean(average_precisions))
        if average_precisions
        else 0.0
    )

    results = {
        "Mode": mode_name,
        "HitRate@1": (
            recall_counts[1]
            / total_queries
            * 100.0
        ),
        "HitRate@5": (
            recall_counts[5]
            / total_queries
            * 100.0
        ),
        "HitRate@10": (
            recall_counts[10]
            / total_queries
            * 100.0
        ),
        "mAP_ret@10": map_at_10 * 100.0,
        # Preserved historical metric labels for backward compatibility
        "Recall@1": (
            recall_counts[1]
            / total_queries
            * 100.0
        ),
        "Recall@5": (
            recall_counts[5]
            / total_queries
            * 100.0
        ),
        "Recall@10": (
            recall_counts[10]
            / total_queries
            * 100.0
        ),
        "mAP": map_at_10 * 100.0,
        "Avg Retrieval Time (ms)": (
            elapsed_seconds
            / total_queries
            * 1000.0
        ),
    }

    print(
        f"Queries             : {total_queries:,}"
    )

    print(
        f"HitRate@1 (Recall@1): "
        f"{results['HitRate@1']:.4f}%"
    )

    print(
        f"HitRate@5 (Recall@5): "
        f"{results['HitRate@5']:.4f}%"
    )

    print(
        f"HitRate@10(Recall@10: "
        f"{results['HitRate@10']:.4f}%"
    )

    print(
        f"mAP_ret@10 (mAP)    : "
        f"{results['mAP_ret@10']:.4f}%"
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
        description="Week 5 Same-Modal Retrieval Evaluation (HitRate@K & mAP_ret@10)"
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
    print("WEEK 5 — TRAINED SAME-MODAL RETRIEVAL (SCIENTIFIC EVALUATION)")
    print("=" * 70)
    print()

    if faiss is None:
        raise ImportError(
            "faiss is not installed in the active Python environment.\n"
            "Please install faiss-cpu to execute retrieval evaluation:\n"
            "    pip install faiss-cpu"
        )

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata = load_test_metadata(args.test_csv)

    patch_ids = (
        metadata["patch_id"]
        .tolist()
    )

    label_sets = build_label_sets(
        metadata
    )

    print(
        f"[OK] Test samples: "
        f"{len(metadata):,}"
    )

    print(
        f"[OK] Patch IDs: "
        f"{len(patch_ids):,}"
    )

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    sar_embeddings = load_embeddings(
        args.sar_embeddings,
        "SAR",
    )

    ms_embeddings = load_embeddings(
        args.ms_embeddings,
        "MS",
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
    # LOAD FAISS INDICES
    # --------------------------------------------------------

    if not args.sar_index.exists():
        raise FileNotFoundError(
            f"SAR index not found:\n"
            f"{args.sar_index}"
        )

    if not args.ms_index.exists():
        raise FileNotFoundError(
            f"MS index not found:\n"
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
        f"[OK] SAR gallery: "
        f"{sar_index.ntotal:,} vectors"
    )

    print(
        f"[OK] MS gallery: "
        f"{ms_index.ntotal:,} vectors"
    )

    # --------------------------------------------------------
    # SAR -> SAR
    # --------------------------------------------------------

    sar_to_sar = evaluate_same_modal(
        embeddings=sar_embeddings,
        index=sar_index,
        label_sets=label_sets,
        mode_name="SAR -> SAR",
    )

    # --------------------------------------------------------
    # MS -> MS
    # --------------------------------------------------------

    ms_to_ms = evaluate_same_modal(
        embeddings=ms_embeddings,
        index=ms_index,
        label_sets=label_sets,
        mode_name="MS -> MS",
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            sar_to_sar,
            ms_to_ms,
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
    print("TRAINED SAME-MODAL EVALUATION COMPLETE")
    print("=" * 70)
    print()

    print(
        results_df.to_string(index=False)
    )

    print()
    print(
        f"[OK] Results saved: "
        f"{args.output_csv}"
    )


if __name__ == "__main__":
    main()