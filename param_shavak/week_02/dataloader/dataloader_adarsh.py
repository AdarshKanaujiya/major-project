"""
dataloader.py - PyTorch DataLoader Integration for BigEarthNet.

This module provides:
1. create_dataloaders(): Instantiates DataLoaders for single-modality BigEarthNet-S1.
2. create_paired_dataloaders(): Instantiates paired train and validation DataLoaders
   for the official multimodal pipeline (train_split.csv and val_split.csv).
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader

# Add parent directory to sys.path to allow importing from root-level config package
import sys
# Add paths for robust imports
_week2_dir = Path(__file__).resolve().parent.parent
_repo_root = _week2_dir.parent.parent
_dl_dir = Path(__file__).resolve().parent

for _p in [str(_week2_dir), str(_repo_root), str(_dl_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config.settings import (
        BASE_DIR,
        DATA_DIR,
        SPLITS_DIR,
        DEFAULT_BATCH_SIZE,
        DEFAULT_NUM_WORKERS,
        NORMALIZATION_MODE,
        PRESERVE_RESOLUTION,
        IMAGE_SIZE,
        setup_logging,
    )
except ImportError:
    BASE_DIR = _repo_root
    DATA_DIR = _repo_root / "BigEarthNet-S1"
    SPLITS_DIR = _repo_root / "results" / "local_pc" / "week_02" / "splits"
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 0
    NORMALIZATION_MODE = "z_score"
    PRESERVE_RESOLUTION = False
    IMAGE_SIZE = (224, 224)

    def setup_logging(name: str):
        l = logging.getLogger(name)
        if not l.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"))
            l.addHandler(h)
        l.setLevel(logging.INFO)
        return l

try:
    from dataset_adarsh import (
        BigEarthNetS1Dataset,
        PairedBigEarthNetDataset,
        paired_collate_fn,
    )
except ImportError:
    from dataloader.dataset_adarsh import (
        BigEarthNetS1Dataset,
        PairedBigEarthNetDataset,
        paired_collate_fn,
    )

logger = setup_logging("DataLoader")


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    splits_dir: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    norm_mode: str = NORMALIZATION_MODE,
    preserve_resolution: bool = PRESERVE_RESOLUTION,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    pin_memory: Optional[bool] = None,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test PyTorch DataLoaders for BigEarthNet-S1.
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir_path}")

    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer. Got: {batch_size}")

    if num_workers < 0:
        raise ValueError(f"Number of workers cannot be negative. Got: {num_workers}")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    logger.info("Initializing datasets for train, validation, and test splits...")

    train_dataset = BigEarthNetS1Dataset(
        data_dir=data_dir_path,
        split="train",
        splits_dir=splits_dir,
        norm_mode=norm_mode,
        target_size=target_size,
        preserve_res=preserve_resolution,
    )

    val_dataset = BigEarthNetS1Dataset(
        data_dir=data_dir_path,
        split="validation",
        splits_dir=splits_dir,
        norm_mode=norm_mode,
        target_size=target_size,
        preserve_res=preserve_resolution,
    )

    test_dataset = BigEarthNetS1Dataset(
        data_dir=data_dir_path,
        split="test",
        splits_dir=splits_dir,
        norm_mode=norm_mode,
        target_size=target_size,
        preserve_res=preserve_resolution,
    )

    logger.info("Creating PyTorch DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_paired_dataloaders(
    train_csv: Optional[Union[str, Path]] = None,
    val_csv: Optional[Union[str, Path]] = None,
    s1_root: Union[str, Path] = DATA_DIR,
    s2_root: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    require_s2: bool = False,
    drop_last_train: bool = False,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates paired PyTorch DataLoaders for official train (21,000) and validation (4,500) splits.

    Args:
        train_csv: Path to train_split.csv (21,000 samples).
        val_csv: Path to val_split.csv (4,500 samples).
        s1_root: Directory containing BigEarthNet-S1 patches.
        s2_root: Optional directory containing BigEarthNet-S2 patches (configurable).
        batch_size: Number of samples per batch (default 64).
        num_workers: DataLoader subprocess count (default 0 for Windows stability).
        require_s2: If True, requires valid S2 imagery on disk; if False, enables safe S1 verification.
        drop_last_train: If False, preserves the last partial batch (yields all 329 train batches).
        pin_memory: If True, pins memory for CUDA transfer.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    if train_csv is None:
        c_train = SPLITS_DIR / "train_split.csv"
        train_csv = c_train if c_train.exists() else (BASE_DIR / "csvs" / "train_split.csv")
    if val_csv is None:
        c_val = SPLITS_DIR / "val_split.csv"
        val_csv = c_val if c_val.exists() else (BASE_DIR / "csvs" / "val_split.csv")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    logger.info("Initializing PairedBigEarthNetDataset instances for train and validation...")
    train_dataset = PairedBigEarthNetDataset(
        csv_path=train_csv,
        s1_root=s1_root,
        s2_root=s2_root,
        require_s2=require_s2,
    )

    val_dataset = PairedBigEarthNetDataset(
        csv_path=val_csv,
        s1_root=s1_root,
        s2_root=s2_root,
        require_s2=require_s2,
    )

    logger.info(
        f"Creating paired DataLoaders (batch_size={batch_size}, num_workers={num_workers}, "
        f"require_s2={require_s2})..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Training: shuffle=True
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Validation: deterministic shuffle=False
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing create_paired_dataloaders...")
    train_ld, val_ld = create_paired_dataloaders(batch_size=64, num_workers=0, require_s2=False)
    print(f"Paired Train Loader samples: {len(train_ld.dataset)}, batches: {len(train_ld)}")
    print(f"Paired Val Loader samples:   {len(val_ld.dataset)}, batches: {len(val_ld)}")
    assert len(train_ld.dataset) == 21000
    assert len(val_ld.dataset) == 4500
    assert len(train_ld) == 329
    assert len(val_ld) == 71
    print("[PASS] create_paired_dataloaders length and batch count checks passed successfully.")
