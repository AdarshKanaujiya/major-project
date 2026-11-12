"""
Week 5 - Build FAISS gallery indices (Scientific Revision).

Builds two IndexFlatIP indices from the trained test embeddings:
    - SAR gallery
    - Multispectral gallery

The embeddings are already L2-normalized, so inner product
is equivalent to cosine similarity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.config.settings import RESULTS_DIR
except ImportError:
    RESULTS_DIR = PROJECT_ROOT / "results" / "local_pc"

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_EMBEDDINGS_DIR = RESULTS_DIR / "week_05" / "trained_embeddings"
DEFAULT_INDEX_DIR = RESULTS_DIR / "week_05" / "faiss_indices" / "runs"

DEFAULT_SAR_EMBEDDINGS = DEFAULT_EMBEDDINGS_DIR / "trained_test_sar_embeddings.npy"
DEFAULT_MS_EMBEDDINGS = DEFAULT_EMBEDDINGS_DIR / "trained_test_ms_embeddings.npy"
DEFAULT_SAR_INDEX = DEFAULT_INDEX_DIR / "sar_gallery.index"
DEFAULT_MS_INDEX = DEFAULT_INDEX_DIR / "ms_gallery.index"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_SAMPLES = 4500
EXPECTED_DIMENSION = 512


# ============================================================
# LOAD AND VALIDATE EMBEDDINGS
# ============================================================

def load_embeddings(
    path: Path,
    name: str,
) -> np.ndarray:
    """Load and validate one embedding matrix."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} embeddings not found:\n{path}"
        )

    embeddings = np.load(path)

    print(
        f"[OK] Loaded {name} embeddings: "
        f"{embeddings.shape}"
    )

    if embeddings.shape != (
        EXPECTED_SAMPLES,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected {name} embedding shape: "
            f"{embeddings.shape}. "
            f"Expected "
            f"({EXPECTED_SAMPLES}, {EXPECTED_DIMENSION})."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(
            np.float32
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name} embeddings contain NaN or Inf."
        )

    return embeddings


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    embeddings: np.ndarray,
    output_path: Path,
    name: str,
) -> None:
    """Build and save an IndexFlatIP FAISS index."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print(
        f"Building {name} FAISS index..."
    )

    index.add(embeddings)

    if index.ntotal != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"{name} index contains "
            f"{index.ntotal} vectors instead of "
            f"{EXPECTED_SAMPLES}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(output_path),
    )

    print(
        f"[OK] {name} index saved: "
        f"{output_path}"
    )

    print(
        f"[OK] Number of vectors: "
        f"{index.ntotal:,}"
    )

    print(
        f"[OK] Dimension: "
        f"{index.d}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 5 Build FAISS Gallery Indices"
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
        help="Path to output SAR FAISS index (.index)",
    )
    parser.add_argument(
        "--ms-index",
        type=Path,
        default=DEFAULT_MS_INDEX,
        help="Path to output MS FAISS index (.index)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WEEK 5 — BUILD FAISS GALLERY INDICES (SCIENTIFIC REVISION)")
    print("=" * 70)
    print()

    if faiss is None:
        raise ImportError(
            "faiss is not installed in the active Python environment.\n"
            "Please install faiss-cpu to build FAISS indices:\n"
            "    pip install faiss-cpu"
        )

    print(f"FAISS version: {faiss.__version__}")
    print()

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

    print()

    # --------------------------------------------------------
    # BUILD INDICES
    # --------------------------------------------------------

    build_index(
        embeddings=sar_embeddings,
        output_path=args.sar_index,
        name="SAR",
    )

    print()

    build_index(
        embeddings=ms_embeddings,
        output_path=args.ms_index,
        name="MS",
    )

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    print()
    print("Verifying saved indices...")

    sar_index = faiss.read_index(
        str(args.sar_index)
    )

    ms_index = faiss.read_index(
        str(args.ms_index)
    )

    if sar_index.ntotal != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Saved SAR index verification failed."
        )

    if ms_index.ntotal != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Saved MS index verification failed."
        )

    if sar_index.d != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Saved SAR index dimension is incorrect."
        )

    if ms_index.d != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Saved MS index dimension is incorrect."
        )

    print(
        f"[OK] SAR index: "
        f"{sar_index.ntotal:,} vectors × "
        f"{sar_index.d} dimensions"
    )

    print(
        f"[OK] MS index: "
        f"{ms_index.ntotal:,} vectors × "
        f"{ms_index.d} dimensions"
    )

    print()
    print("=" * 70)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 70)
    print()

    print(
        f"SAR gallery: {args.sar_index}"
    )

    print(
        f"MS gallery : {args.ms_index}"
    )


if __name__ == "__main__":
    main()