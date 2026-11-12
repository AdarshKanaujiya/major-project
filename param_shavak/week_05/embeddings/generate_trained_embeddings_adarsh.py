"""
weeks/week_05/embeddings/generate_trained_embeddings_adarsh.py

Generates trained validation embeddings from TwoTowerNetwork.
Supports optional L2 normalization for downstream cosine similarity search.
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
except ImportError:
    torch = None

import numpy as np

# Canonical TwoTowerNetwork import
try:
    from weeks.week_04.models.two_tower_jainam import TwoTowerNetwork
except ImportError:
    TwoTowerNetwork = None

# Canonical DataLoader import
try:
    from weeks.week_02.dataloader.dataloader_adarsh import create_paired_dataloaders
except ImportError:
    create_paired_dataloaders = None


# ============================================================
# DEFAULT PATHS & CONSTANTS
# ============================================================

DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model-v1.ckpt"
DEFAULT_VAL_CSV = SPLITS_DIR / "val_split.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "week_05" / "trained_embeddings"

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


# ============================================================
# LOAD MODEL FROM LIGHTNING CHECKPOINT
# ============================================================

def load_trained_model(checkpoint_path: Path, device: str = "cpu"):
    """
    Loads TwoTowerNetwork weights from a PyTorch Lightning checkpoint.
    Strips 'network.' prefix from state_dict keys.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load trained model.")
    if TwoTowerNetwork is None:
        raise RuntimeError("TwoTowerNetwork class could not be imported from weeks.week_04.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print("=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'state_dict'."
        )

    checkpoint_state_dict = checkpoint["state_dict"]

    # Strip Lightning 'network.' prefix
    network_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("network."):
            new_key = key[len("network."):]
            network_state_dict[new_key] = value
        elif not key.startswith("loss_fn."):
            network_state_dict[key] = value

    if not network_state_dict:
        raise RuntimeError(
            "Could not find model parameters in checkpoint."
        )

    model = TwoTowerNetwork(pretrained=False)
    missing_keys, unexpected_keys = model.load_state_dict(
        network_state_dict,
        strict=False,
    )

    if missing_keys:
        print("\nWARNING - Missing keys:")
        for key in missing_keys:
            print("  ", key)

    if unexpected_keys:
        print("\nWARNING - Unexpected keys:")
        for key in unexpected_keys:
            print("  ", key)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch. "
            "Do not generate embeddings until this is resolved."
        )

    model.to(device)
    model.eval()
    print("\n[OK] Trained TwoTowerNetwork loaded successfully.")
    return model


# ============================================================
# CREATE VALIDATION DATALOADER
# ============================================================

def create_validation_loader(
    val_csv: Path,
    s1_root: Path,
    s2_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    """Creates validation DataLoader using canonical Week 2 DataLoader."""
    print("\n" + "=" * 60)
    print("CREATING VALIDATION DATALOADER")
    print("=" * 60)

    if create_paired_dataloaders is None:
        raise RuntimeError("create_paired_dataloaders could not be imported from weeks.week_02.")

    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found:\n{val_csv}")
    if not s1_root.exists():
        raise FileNotFoundError(f"S1 root not found:\n{s1_root}")
    if not s2_root.exists():
        raise FileNotFoundError(f"S2 root not found:\n{s2_root}")

    _, val_loader = create_paired_dataloaders(
        train_csv=val_csv,
        val_csv=val_csv,
        s1_root=s1_root,
        s2_root=s2_root,
        batch_size=batch_size,
        num_workers=num_workers,
        require_s2=True,
        drop_last_train=False,
    )

    print(f"[OK] Validation samples: {len(val_loader.dataset)}")
    print(f"[OK] Validation batches: {len(val_loader)}")
    return val_loader


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(model, val_loader, device: str = "cpu", normalize: bool = False):
    """
    Runs forward inference through TwoTowerNetwork.
    Optionally applies L2 normalization to extracted 512-d embeddings.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to generate embeddings.")

    print("\n" + "=" * 60)
    print("GENERATING TRAINED VALIDATION EMBEDDINGS")
    print(f"L2 Normalization: {normalize}")
    print("=" * 60)

    sar_embeddings = []
    ms_embeddings = []
    patch_ids = []

    total_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            sar_batch, ms_batch, labels, batch_patch_ids = batch

            if ms_batch is None:
                raise RuntimeError("MS batch is None. S2 data was not loaded.")

            sar_batch = sar_batch.to(device, non_blocking=True)
            ms_batch = ms_batch.to(device, non_blocking=True)

            sar_batch_emb, ms_batch_emb = model(sar_batch, ms_batch)

            if normalize:
                sar_batch_emb = F.normalize(sar_batch_emb, p=2, dim=-1)
                ms_batch_emb = F.normalize(ms_batch_emb, p=2, dim=-1)

            sar_embeddings.append(sar_batch_emb.cpu().numpy())
            ms_embeddings.append(ms_batch_emb.cpu().numpy())
            patch_ids.extend(batch_patch_ids)

            if (
                batch_idx == 0
                or (batch_idx + 1) % 10 == 0
                or batch_idx == total_batches - 1
            ):
                print(
                    f"Batch {batch_idx + 1}/{total_batches} "
                    f"| Samples processed: {len(patch_ids)}"
                )

    sar_embeddings = np.concatenate(sar_embeddings, axis=0)
    ms_embeddings = np.concatenate(ms_embeddings, axis=0)

    return sar_embeddings, ms_embeddings, patch_ids


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    sar_embeddings: np.ndarray,
    ms_embeddings: np.ndarray,
    patch_ids: list[str],
    output_dir: Path,
    normalized: bool = False,
):
    """Saves embedding arrays and patch ID manifest."""
    print("\n" + "=" * 60)
    print("SAVING TRAINED EMBEDDINGS")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SAR embedding shape: {sar_embeddings.shape}")
    print(f"MS embedding shape:  {ms_embeddings.shape}")
    print(f"Patch ID count:      {len(patch_ids)}")

    prefix = "trained_val"
    suffix = "_l2.npy" if normalized else ".npy"
    sar_output = output_dir / f"{prefix}_sar_embeddings{suffix}"
    ms_output = output_dir / f"{prefix}_ms_embeddings{suffix}"
    patch_ids_output = output_dir / f"{prefix}_patch_ids.txt"

    np.save(sar_output, sar_embeddings.astype(np.float32))
    np.save(ms_output, ms_embeddings.astype(np.float32))

    if not patch_ids_output.exists():
        with open(patch_ids_output, "w", encoding="utf-8") as f:
            for patch_id in patch_ids:
                f.write(f"{patch_id}\n")

    print("\n[OK] Files saved:")
    print(f"  SAR: {sar_output}")
    print(f"  MS:  {ms_output}")
    print(f"  IDs: {patch_ids_output}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SENTINEL-RS Week 5 - Generate Trained Embeddings"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to trained PyTorch Lightning checkpoint",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_VAL_CSV,
        help="Path to validation split CSV",
    )
    parser.add_argument(
        "--s1-root",
        type=Path,
        default=S1_DIR,
        help="Path to BigEarthNet-S1 root directory",
    )
    parser.add_argument(
        "--s2-root",
        type=Path,
        default=S2_DIR,
        help="Path to BigEarthNet-S2 root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated embeddings",
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
        "--normalize",
        action="store_true",
        help="Apply L2 normalization before saving embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        help="Compute device (cuda or cpu)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SENTINEL-RS: TRAINED EMBEDDING GENERATION")
    print("=" * 60)

    if torch is None:
        print("[BLOCKED BY ENVIRONMENT] PyTorch is not installed in the active environment.")
        print("Please run this script in an environment with PyTorch installed.")
        sys.exit(0)

    # 1. Load trained checkpoint
    model = load_trained_model(args.checkpoint, device=args.device)

    # 2. Create validation loader
    val_loader = create_validation_loader(
        args.val_csv,
        args.s1_root,
        args.s2_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 3. Generate embeddings
    sar_embeddings, ms_embeddings, patch_ids = generate_embeddings(
        model,
        val_loader,
        device=args.device,
        normalize=args.normalize,
    )

    # 4. Save
    save_outputs(
        sar_embeddings,
        ms_embeddings,
        patch_ids,
        args.output_dir,
        normalized=args.normalize,
    )

    print("\n" + "=" * 60)
    print("TRAINED EMBEDDING GENERATION COMPLETE")
    print(f"SAR: {sar_embeddings.shape}")
    print(f"MS:  {ms_embeddings.shape}")
    print(f"IDs: {len(patch_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
