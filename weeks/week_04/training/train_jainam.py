"""Week 4 Canonical Contrastive Training Script
Cross-Modal SAR–Multispectral Satellite Image Retrieval

Training setup:
- PyTorch Lightning
- Two-tower ResNet-50 model with independent backbones and projection heads
- Symmetric InfoNCE loss with temperature tau = 0.07
- Configurable Optimizer: Adam (historical reproduction) or AdamW (experimental)
- Configurable Projection Head: Linear -> ReLU -> Linear (historical) or Linear -> BN -> ReLU -> Linear
- CosineAnnealingLR scheduler
- Physical batch size: 16 (or user-configured)
- Precision: 16-mixed (or user-configured)
- Gradient clipping: 1.0
- Best checkpoint selected using validation loss
- Isolated output directory: Never overwrites historical results in results/local_pc/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Inject project root for robust imports
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from shared.config.settings import (
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLITS_DIR,
    S1_DIR,
    S2_DIR,
    TEMPERATURE as DEFAULT_TEMPERATURE,
)
from weeks.week_02.dataloader.dataloader_adarsh import create_paired_dataloaders
from weeks.week_04.models.two_tower_jainam import InfoNCELoss, TwoTowerNetwork


# ============================================================
# LIGHTNING MODULE
# ============================================================

class RetrievalLightningModule(L.LightningModule):
    """Lightning wrapper around the two-tower retrieval network."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_type: str = "Adam",
        temperature: float = DEFAULT_TEMPERATURE,
        max_epochs: int = 20,
        use_batchnorm: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.network = TwoTowerNetwork(
            pretrained=pretrained,
            use_batchnorm=use_batchnorm,
        )

        self.loss_fn = InfoNCELoss(
            temperature=temperature
        )

    def forward(
        self,
        sar_batch: torch.Tensor,
        ms_batch: torch.Tensor,
    ):
        return self.network(
            sar_batch,
            ms_batch,
        )

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):
        sar_batch, ms_batch, _, _ = batch

        sar_embeddings, ms_embeddings = self(
            sar_batch,
            ms_batch,
        )

        loss = self.loss_fn(
            sar_embeddings,
            ms_embeddings,
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=sar_batch.size(0),
        )

        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        sar_batch, ms_batch, _, _ = batch

        sar_embeddings, ms_embeddings = self(
            sar_batch,
            ms_batch,
        )

        loss = self.loss_fn(
            sar_embeddings,
            ms_embeddings,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=sar_batch.size(0),
        )

        return loss

    def configure_optimizers(self):
        opt_choice = self.hparams.optimizer_type.lower()

        if opt_choice == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif opt_choice == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer_type '{self.hparams.optimizer_type}'. "
                "Must be 'Adam' or 'AdamW'."
            )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ============================================================
# TRAINING CURVE GENERATOR
# ============================================================

def create_training_plot(
    metrics_path: Path,
    output_path: Path,
) -> None:
    """Create train/validation loss plot from Lightning CSV logs."""
    if not metrics_path.exists():
        print(f"[WARNING] Metrics file not found: {metrics_path}")
        return

    metrics = pd.read_csv(metrics_path)

    train_metrics = metrics[["epoch", "train_loss"]].dropna() if "train_loss" in metrics.columns else pd.DataFrame()
    val_metrics = metrics[["epoch", "val_loss"]].dropna() if "val_loss" in metrics.columns else pd.DataFrame()

    if train_metrics.empty and val_metrics.empty:
        print("[WARNING] No loss data found for plotting.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    if not train_metrics.empty:
        plt.plot(
            train_metrics["epoch"],
            train_metrics["train_loss"],
            marker="o",
            label="Train Loss",
        )

    if not val_metrics.empty:
        plt.plot(
            val_metrics["epoch"],
            val_metrics["val_loss"],
            marker="o",
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.title("Cross-Modal Retrieval Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Training plot saved: {output_path}")


# ============================================================
# CLI ARGUMENT PARSER
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL-RS Week 4 Two-Tower Contrastive Training"
    )

    # Data paths (defaulting to verified settings constants)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=SPLITS_DIR / "train_split.csv",
        help="Path to training split CSV (default: results/local_pc/week_02/splits/train_split.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=SPLITS_DIR / "val_split.csv",
        help="Path to validation split CSV (default: results/local_pc/week_02/splits/val_split.csv)",
    )
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=S1_DIR,
        help="Root directory for Sentinel-1 GeoTIFF patches",
    )
    parser.add_argument(
        "--s2-root",
        type=Path,
        default=S2_DIR,
        help="Root directory for Sentinel-2 GeoTIFF patches",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RESULTS_DIR / "week_04" / "runs" / "canonical_run",
        help="Output directory for new run artifacts (never overwrites historical results)",
    )

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes (default: 4)")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum training epochs (default: 20)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate (default: 0.0001)")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "AdamW"],
        help="Optimizer type: 'Adam' (historical reproduction) or 'AdamW' (experimental alternative)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer (default: 0.0 for Adam, recommended 1e-4 for AdamW)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"InfoNCE temperature parameter (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Gradient norm clipping value (default: 1.0)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning precision mode (default: 16-mixed)",
    )
    parser.add_argument(
        "--use-batchnorm",
        action="store_true",
        default=False,
        help="Enable BatchNorm1d in projection heads (default: False, preserves historical checkpoint compatibility)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and initialization (default: 42)",
    )

    return parser.parse_args()


# ============================================================
# MAIN ROUTINE
# ============================================================

def main() -> None:
    args = parse_args()

    # Deterministic seeding (best-effort across environments)
    L.seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    print("=" * 70)
    print("SENTINEL-RS WEEK 4 — TWO-TOWER CONTRASTIVE TRAINING")
    print("=" * 70)
    print(f"Train CSV          : {args.train_csv}")
    print(f"Validation CSV     : {args.val_csv}")
    print(f"S1 Root            : {args.s1_root}")
    print(f"S2 Root            : {args.s2_root}")
    print(f"Run Output Dir     : {args.run_dir}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Workers            : {args.num_workers}")
    print(f"Max epochs         : {args.max_epochs}")
    print(f"Optimizer          : {args.optimizer}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Weight decay       : {args.weight_decay}")
    print(f"Temperature        : {args.temperature}")
    print(f"Use BatchNorm1d    : {args.use_batchnorm}")
    print(f"Gradient clipping  : {args.gradient_clip_val}")
    print(f"Precision          : {args.precision}")
    print(f"Random seed        : {args.seed}")
    print("=" * 70)
    print()

    # Check required paths
    for name, path in [
        ("Train CSV", args.train_csv),
        ("Validation CSV", args.val_csv),
        ("S1 Root", args.s1_root),
        ("S2 Root", args.s2_root),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Set up segregated output directories
    checkpoint_dir = args.run_dir / "checkpoints"
    log_dir = args.run_dir / "logs"
    plot_dir = args.run_dir / "plots"

    for d in (checkpoint_dir, log_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Instantiate DataLoaders
    print("Creating training and validation paired DataLoaders...")
    train_loader, val_loader = create_paired_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        s1_root=args.s1_root,
        s2_root=args.s2_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[OK] Training samples   : {len(train_loader.dataset):,}")
    print(f"[OK] Validation samples : {len(val_loader.dataset):,}")
    print()

    # Instantiate Model
    print("Creating TwoTowerNetwork Lightning module...")
    model = RetrievalLightningModule(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        temperature=args.temperature,
        max_epochs=args.max_epochs,
        use_batchnorm=args.use_batchnorm,
        pretrained=True,
    )
    print("[OK] Model created successfully.")
    print()

    # Callbacks and Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="contrastive_training",
    )

    # PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        log_every_n_steps=1,
    )

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    start_time = time.perf_counter()

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    elapsed_hours = (time.perf_counter() - start_time) / 3600.0

    # Process metrics and create plot
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    plot_path = plot_dir / "training_loss.png"
    create_training_plot(metrics_path, plot_path)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Elapsed time     : {elapsed_hours:.4f} hours")
    print(f"Best checkpoint  : {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint  : {checkpoint_callback.last_model_path}")
    print(f"Metrics CSV      : {metrics_path}")
    print(f"Loss plot        : {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
