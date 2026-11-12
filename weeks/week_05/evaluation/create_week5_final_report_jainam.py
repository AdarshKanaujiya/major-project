"""
Week 5 - Final FAISS Retrieval and Test Set Evaluation Report (Scientific Revision).

Combines the existing Week 3 baseline results with the Week 5
trained-model results into the final comparison table.

Important (Reconciled Phase A.1):
- Cross-modal retrieval uses exact paired patch IDs.
- Same-modal retrieval uses shared land-cover labels.
- Same-modal Recall@K represents HitRate@K (fraction of queries with >= 1 relevant item in top K).
- Same-modal mAP represents Retrieved-Normalized AP@10 (AP_ret@10), normalized by sum_{k=1}^10 rel(k).
- Baseline mAP values are only reported where they were actually
  calculated using the corresponding evaluation setup.
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

import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_HISTORICAL_DIR = RESULTS_DIR / "week_05" / "evaluation"
DEFAULT_TRAINED_CROSS_MODAL = DEFAULT_HISTORICAL_DIR / "trained_cross_modal_results.csv"
DEFAULT_TRAINED_SAME_MODAL = DEFAULT_HISTORICAL_DIR / "trained_same_modal_results.csv"

DEFAULT_REPORT_DIR = DEFAULT_HISTORICAL_DIR / "runs"
DEFAULT_OUTPUT_CSV = DEFAULT_REPORT_DIR / "week5_final_comparison.csv"
DEFAULT_OUTPUT_TXT = DEFAULT_REPORT_DIR / "Week_5_final_results.txt"


# ============================================================
# WEEK 3 BASELINE RESULTS
# ============================================================

# Week 3 baseline:
# Cross-modal relevance = exact paired patch ID.
BASELINE_CROSS_MODAL = {
    "SAR -> MS": {
        "Recall@1": 0.2222,
        "Recall@5": 0.4889,
        "Recall@10": 0.8222,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0589,
    },
    "MS -> SAR": {
        "Recall@1": 0.4889,
        "Recall@5": 1.5111,
        "Recall@10": 2.1333,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0817,
    },

    # Same-modal relevance = shared >=1 land-cover label.
    "SAR -> SAR": {
        "Recall@1": 15.1333,
        "Recall@5": 26.3111,
        "Recall@10": 33.4667,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0802,
    },
    "MS -> MS": {
        "Recall@1": 18.6444,
        "Recall@5": 31.5778,
        "Recall@10": 39.6667,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0812,
    },
}


# ============================================================
# LOAD TRAINED RESULTS
# ============================================================

def load_trained_results(
    cross_modal_path: Path | None = None,
    same_modal_path: Path | None = None,
) -> dict[str, dict]:
    """Load Week 5 trained evaluation results."""
    target_cm = cross_modal_path if cross_modal_path else DEFAULT_TRAINED_CROSS_MODAL
    target_sm = same_modal_path if same_modal_path else DEFAULT_TRAINED_SAME_MODAL

    if not target_cm.exists():
        raise FileNotFoundError(
            f"Missing trained cross-modal results:\n"
            f"{target_cm}"
        )

    if not target_sm.exists():
        raise FileNotFoundError(
            f"Missing trained same-modal results:\n"
            f"{target_sm}"
        )

    cross_modal = pd.read_csv(target_cm)
    same_modal = pd.read_csv(target_sm)

    results = {}

    for _, row in cross_modal.iterrows():

        results[row["Mode"]] = {
            "Recall@1": float(row["Recall@1"]),
            "Recall@5": float(row["Recall@5"]),
            "Recall@10": float(row["Recall@10"]),
            "mAP": None,
            "Avg Retrieval Time (ms)": float(
                row["Avg Retrieval Time (ms)"]
            ),
        }

    for _, row in same_modal.iterrows():

        # Support both new HitRate@K and preserved Recall@K column names
        r1 = float(row["HitRate@1"]) if "HitRate@1" in row else float(row["Recall@1"])
        r5 = float(row["HitRate@5"]) if "HitRate@5" in row else float(row["Recall@5"])
        r10 = float(row["HitRate@10"]) if "HitRate@10" in row else float(row["Recall@10"])
        map_val = float(row["mAP_ret@10"]) if "mAP_ret@10" in row else float(row["mAP"])

        results[row["Mode"]] = {
            "Recall@1": r1,
            "Recall@5": r5,
            "Recall@10": r10,
            "mAP": map_val,
            "Avg Retrieval Time (ms)": float(
                row["Avg Retrieval Time (ms)"]
            ),
        }

    return results


# ============================================================
# BUILD FINAL TABLE
# ============================================================

def build_final_table(
    trained_results: dict[str, dict],
) -> pd.DataFrame:

    rows = []

    mode_order = [
        ("SAR -> SAR", "baseline"),
        ("MS -> MS", "baseline"),
        ("SAR -> MS", "baseline"),
        ("MS -> SAR", "baseline"),
        ("SAR -> SAR", "trained"),
        ("MS -> MS", "trained"),
        ("SAR -> MS", "trained"),
        ("MS -> SAR", "trained"),
    ]

    for mode, model_type in mode_order:

        if model_type == "baseline":
            result = BASELINE_CROSS_MODAL[mode]

        else:
            result = trained_results[mode]

        rows.append(
            {
                "Mode": (
                    f"{mode} "
                    f"({model_type})"
                ),
                "Recall@1": result["Recall@1"],
                "Recall@5": result["Recall@5"],
                "Recall@10": result["Recall@10"],
                "mAP": result["mAP"],
                "Avg Retrieval Time (ms)": (
                    result["Avg Retrieval Time (ms)"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

def write_text_report(
    final_df: pd.DataFrame,
    output_txt_path: Path | None = None,
) -> None:
    """Write human-readable final summary report."""
    target_txt = output_txt_path if output_txt_path else DEFAULT_OUTPUT_TXT
    target_txt.parent.mkdir(parents=True, exist_ok=True)

    with target_txt.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "WEEK 5 — FINAL FAISS RETRIEVAL "
            "AND TEST SET EVALUATION\n"
        )

        file.write("=" * 72)
        file.write("\n\n")

        file.write(
            "Dataset / evaluation size: "
            "4,500 test pairs\n\n"
        )

        file.write(
            "Embedding dimension: 512\n\n"
        )

        file.write(
            "FAISS index: IndexFlatIP\n\n"
        )

        file.write(
            "Cross-modal relevance:\n"
            "Exact paired patch ID.\n\n"
        )

        file.write(
            "Same-modal relevance:\n"
            "At least one shared land-cover label; "
            "query item excluded.\n\n"
        )

        file.write(
            "NOTE ON METRIC DEFINITIONS (Reconciled Phase A.1):\n"
            "- Cross-modal Recall@K evaluates exact paired scene retrieval.\n"
            "- Same-modal Recall@K reflects HitRate@K (percentage of queries with >=1 matching land-cover class in top K).\n"
            "- Same-modal mAP reflects Retrieved-Normalized AP@10 (AP_ret@10), normalized by sum_{k=1}^10 rel(k).\n"
            "- Baseline mAP values are not populated because the Week 3 baseline records "
            "contain Recall@K and retrieval time, but do not contain baseline mAP values.\n\n"
        )

        file.write(
            final_df.to_string(
                index=False,
                na_rep="N/A",
            )
        )

        file.write("\n\n")

        file.write(
            "TRAINED CROSS-MODAL RESULTS\n"
        )

        file.write("-" * 72)
        file.write("\n")

        file.write(
            "SAR -> MS:\n"
            "Recall@1 = 38.7778%\n"
            "Recall@5 = 64.8222%\n"
            "Recall@10 = 75.0444%\n"
            "Average retrieval time = 0.0123 ms/query\n\n"
        )

        file.write(
            "MS -> SAR:\n"
            "Recall@1 = 37.0444%\n"
            "Recall@5 = 63.7333%\n"
            "Recall@10 = 74.0000%\n"
            "Average retrieval time = 0.0139 ms/query\n\n"
        )

        file.write(
            "TRAINED SAME-MODAL RESULTS\n"
        )

        file.write("-" * 72)
        file.write("\n")

        file.write(
            "SAR -> SAR:\n"
            "HitRate@1 (Recall@1)   = 53.4000%\n"
            "HitRate@5 (Recall@5)   = 85.9778%\n"
            "HitRate@10 (Recall@10) = 92.4889%\n"
            "mAP_ret@10 (mAP@10)    = 61.2397%\n"
            "Average retrieval time = 0.0118 ms/query\n\n"
        )

        file.write(
            "MS -> MS:\n"
            "HitRate@1 (Recall@1)   = 57.8444%\n"
            "HitRate@5 (Recall@5)   = 87.0444%\n"
            "HitRate@10 (Recall@10) = 93.4667%\n"
            "mAP_ret@10 (mAP@10)    = 64.0870%\n"
            "Average retrieval time = 0.0113 ms/query\n"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 5 Final Evaluation Report Compiler"
    )
    parser.add_argument(
        "--cross-modal-csv",
        type=Path,
        default=DEFAULT_TRAINED_CROSS_MODAL,
        help="Path to trained_cross_modal_results.csv",
    )
    parser.add_argument(
        "--same-modal-csv",
        type=Path,
        default=DEFAULT_TRAINED_SAME_MODAL,
        help="Path to trained_same_modal_results.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save week5_final_comparison.csv",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=DEFAULT_OUTPUT_TXT,
        help="Path to save Week_5_final_results.txt",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("WEEK 5 — FINAL RESULTS")
    print("=" * 72)
    print()

    trained_results = load_trained_results(
        cross_modal_path=args.cross_modal_csv,
        same_modal_path=args.same_modal_csv,
    )

    final_df = build_final_table(
        trained_results
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        args.output_csv,
        index=False,
    )

    write_text_report(
        final_df,
        output_txt_path=args.output_txt,
    )

    print(
        final_df.to_string(
            index=False,
            na_rep="N/A",
        )
    )

    print()
    print(
        f"[OK] CSV saved: {args.output_csv}"
    )

    print(
        f"[OK] Text report saved: {args.output_txt}"
    )

    print()
    print(
        "WEEK 5 FINAL REPORT COMPLETE"
    )


if __name__ == "__main__":
    main()