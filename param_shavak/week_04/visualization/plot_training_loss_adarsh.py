"""Week 4 Training Loss Visualization Script
Plots train and validation loss curves from metrics.csv.

Authored by Adarsh (Person 2).
Path and interface standardized with shared.config.settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Inject project root for robust imports
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from shared.config.settings import RESULTS_DIR

DEFAULT_METRICS_CSV = RESULTS_DIR / "week_04" / "metrics" / "metrics.csv"
DEFAULT_OUTPUT_PLOT = RESULTS_DIR / "week_04" / "visualization" / "training_loss_by_ad.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train and validation loss curves from metrics CSV."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help=f"Path to input metrics CSV (default: {DEFAULT_METRICS_CSV})",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT,
        help=f"Path to output PNG plot (default: {DEFAULT_OUTPUT_PLOT})",
    )
    return parser.parse_args()


def plot_training_loss(metrics_csv: Path, output_plot: Path) -> int:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Training log not found:\n{metrics_csv}")

    df = pd.read_csv(metrics_csv)

    required = {"epoch", "train_loss", "val_loss"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}"
        )

    train_df = (
        df[["epoch", "train_loss"]]
        .dropna()
        .drop_duplicates(subset=["epoch"])
        .sort_values("epoch")
    )

    val_df = (
        df[["epoch", "val_loss"]]
        .dropna()
        .drop_duplicates(subset=["epoch"])
        .sort_values("epoch")
    )

    if train_df.empty:
        raise ValueError("No train_loss values found.")

    if val_df.empty:
        raise ValueError("No val_loss values found.")

    print("\nTraining epochs:")
    print(train_df.to_string(index=False))

    print("\nValidation epochs:")
    print(val_df.to_string(index=False))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARNING] matplotlib is not installed in the current environment.")
        print("          Loss tables are printed above. Plot generation was skipped.")
        return 0

    output_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_df["epoch"],
        train_df["train_loss"],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        val_df["epoch"],
        val_df["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.title("Cross-Modal Retrieval Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_plot,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\n[OK] Training plot saved: {output_plot}")
    return 0


def main() -> None:
    args = parse_args()
    plot_training_loss(args.metrics_csv, args.output_plot)


if __name__ == "__main__":
    main()
