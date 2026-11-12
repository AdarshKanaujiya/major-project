"""
weeks/week_05/embeddings/extract_test_embeddings_jainam.py

Week 5 - Extract trained test embeddings.
Loads the best Week 4 checkpoint and generates 512-D L2-normalized embeddings
for all 4,500 test SAR/MS pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.config.settings import (
        RESULTS_DIR,
        SPLITS_DIR,
        CHECKPOINTS_DIR,
        S1_DIR,
        S2_DIR,
    )
except ImportError:
    RESULTS_DIR = PROJECT_ROOT / "results" / "local_pc"
    SPLITS_DIR = RESULTS_DIR / "week_02" / "splits"
    CHECKPOINTS_DIR = RESULTS_DIR / "week_04" / "checkpoints"
    S1_DIR = PROJECT_ROOT / "data" / "BigEarthNet-S1"
    S2_DIR = PROJECT_ROOT / "data" / "BigEarthNet-S2"

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    F = None
    DataLoader = None

import numpy as np

# Canonical imports
try:
    from weeks.week_02.dataset.paired_dataset import PairedBigEarthNetDataset
except ImportError:
    PairedBigEarthNetDataset = None

try:
    from weeks.week_04.models.two_tower_jainam import TwoTowerNetwork
except ImportError:
    TwoTowerNetwork = None


# ============================================================
# DEFAULT PATHS & CONFIGURATION
# ============================================================

DEFAULT_TEST_CSV = SPLITS_DIR / "test_split.csv"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model-v1.ckpt"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "week_05" / "trained_embeddings"

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model(checkpoint_path: Path, device: str = "cpu") -> TwoTowerNetwork:
    """
    Load the trained TwoTowerNetwork from the PyTorch Lightning checkpoint.
    """
    if torch is None or TwoTowerNetwork is None:
        raise RuntimeError("PyTorch and TwoTowerNetwork are required to load model.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    print("Loading trained checkpoint...")
    print(f"Path: {checkpoint_path}")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    network = TwoTowerNetwork(pretrained=False)
    checkpoint_state_dict = checkpoint["state_dict"]

    network_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("network."):
            network_state_dict[key[len("network."):]] = value
        elif not key.startswith("loss_fn."):
            network_state_dict[key] = value

    missing_keys, unexpected_keys = network.load_state_dict(
        network_state_dict,
        strict=False,
    )

    if missing_keys:
        raise RuntimeError(f"Missing model keys: {missing_keys}")

    if unexpected_keys:
        print(f"Warning: Unexpected keys ignored: {unexpected_keys}")

    print("[OK] Trained model weights loaded.")
    network.to(device)
    network.eval()
    return network


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings(
    model: TwoTowerNetwork,
    dataloader: DataLoader,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract SAR and MS embeddings from the trained model.
    Applies L2 normalization for cosine similarity.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to extract embeddings.")

    model.eval()
    sar_embeddings = []
    ms_embeddings = []
    patch_ids = []

    print()
    print("Starting test embedding extraction...")
    print(f"Device       : {device}")
    print(f"Test samples : {len(dataloader.dataset):,}")
    print(f"Batch size   : {batch_size}")
    print()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            sar_batch, ms_batch, _, batch_patch_ids = batch

            sar_batch = sar_batch.to(device, non_blocking=True)
            ms_batch = ms_batch.to(device, non_blocking=True)

            sar_output, ms_output = model(sar_batch, ms_batch)

            # Normalize embeddings for cosine similarity
            sar_output = F.normalize(sar_output, p=2, dim=1)
            ms_output = F.normalize(ms_output, p=2, dim=1)

            sar_embeddings.append(sar_output.cpu().numpy())
            ms_embeddings.append(ms_output.cpu().numpy())
            patch_ids.extend(batch_patch_ids)

            if batch_index % 50 == 0:
                processed = min(batch_index * batch_size, len(dataloader.dataset))
                print(f"Processed {processed:,} / {len(dataloader.dataset):,} test pairs")

    sar_embeddings = np.concatenate(sar_embeddings, axis=0)
    ms_embeddings = np.concatenate(ms_embeddings, axis=0)

    return sar_embeddings, ms_embeddings, patch_ids


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SENTINEL-RS Week 5 - Extract Trained Test Embeddings"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help="Path to test split CSV",
    )
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=S1_DIR,
        help="Path to S1 imagery root",
    )
    parser.add_argument(
        "--s2-root",
        type=Path,
        default=S2_DIR,
        help="Path to S2 imagery root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save extracted embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        help="Compute device (cuda or cpu)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("WEEK 5 — TRAINED TEST EMBEDDING EXTRACTION")
    print("=" * 70)
    print()

    if torch is None:
        print("[BLOCKED BY ENVIRONMENT] PyTorch is not installed in the active environment.")
        print("Please run this script in an environment with PyTorch installed.")
        sys.exit(0)

    if PairedBigEarthNetDataset is None or TwoTowerNetwork is None:
        print("[ERROR] Required project modules (PairedBigEarthNetDataset, TwoTowerNetwork) not found.")
        sys.exit(1)

    required_paths = {
        "Test CSV": args.test_csv,
        "S1 root": args.s1_root,
        "S2 root": args.s2_root,
        "Checkpoint": args.checkpoint,
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found:\n{path}")
        print(f"[OK] {name}: {path}")

    print()
    print(f"[OK] Device: {args.device}")

    # Load dataset
    print("Loading test dataset...")
    test_dataset = PairedBigEarthNetDataset(
        csv_path=args.test_csv,
        s1_root=args.s1_root,
        s2_root=args.s2_root,
    )
    print(f"[OK] Test dataset: {len(test_dataset):,} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    print("[OK] Test DataLoader created.")

    # Load model
    model = load_trained_model(args.checkpoint, device=args.device)

    # Extract
    sar_embeddings, ms_embeddings, patch_ids = extract_embeddings(
        model=model,
        dataloader=test_loader,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Validate
    print()
    print("Validating embeddings...")
    print(f"SAR embedding shape: {sar_embeddings.shape}")
    print(f"MS embedding shape : {ms_embeddings.shape}")

    if not np.isfinite(sar_embeddings).all():
        raise ValueError("SAR embeddings contain NaN or Inf.")
    if not np.isfinite(ms_embeddings).all():
        raise ValueError("MS embeddings contain NaN or Inf.")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sar_output = args.output_dir / "trained_test_sar_embeddings.npy"
    ms_output = args.output_dir / "trained_test_ms_embeddings.npy"

    np.save(sar_output, sar_embeddings.astype(np.float32))
    np.save(ms_output, ms_embeddings.astype(np.float32))

    print()
    print("=" * 70)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"SAR embeddings : {sar_output}")
    print(f"MS embeddings  : {ms_output}")
    print(f"SAR shape      : {sar_embeddings.shape}")
    print(f"MS shape       : {ms_embeddings.shape}")
    print(f"Patch IDs      : {len(patch_ids):,}")


if __name__ == "__main__":
    main()
