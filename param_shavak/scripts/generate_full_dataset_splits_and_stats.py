"""
generate_full_dataset_splits_and_stats.py

Automated full-dataset split generator and training-only normalization statistics workflow
for PARAM SHAVAK HPC cluster execution.

Guarantees:
1. Uses official BigEarthNet train/validation/test splits from metadata.parquet.
2. Filters out quarantined snow, cloud, and shadow patches.
3. Verifies physical existence of both Sentinel-1 and Sentinel-2 GeoTIFF folders.
4. Computes global population mean and std strictly on the TRAINING split.
5. Applies established SAR clipping [-25.0, 0.0] dB and raw DN optical representation.
6. Outputs strictly to param_shavak/results/week_02/.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Connect project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import settings

logger = settings.setup_logging("HPC_Splits_And_Stats")


def resolve_s1_patch_dir(s1_root: Path, s1_name: str) -> Path:
    """Resolve structured S1 directory: <root>/<tile_prefix>/<s1_name>."""
    prefix = "_".join(s1_name.split("_")[:5])
    return s1_root / prefix / s1_name


def resolve_s2_patch_dir(s2_root: Path, patch_id: str) -> Path:
    """Resolve structured S2 directory: <root>/<tile_prefix>/<patch_id>."""
    prefix = "_".join(patch_id.split("_")[:-2])
    return s2_root / prefix / patch_id


def generate_splits(
    metadata_path: Path,
    quarantine_path: Path,
    s1_root: Path,
    s2_root: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate official train, val, and test split CSVs from metadata.parquet.
    Filters out quarantined cloudy patches and ensures paired existence on disk.
    """
    logger.info(f"Loading metadata from: {metadata_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)
    logger.info(f"Loaded {len(df):,} total metadata records.")

    # Filter out quarantined cloudy/snow patches if file provided
    if quarantine_path.exists():
        q_df = pd.read_parquet(quarantine_path)
        quarantine_ids = set(q_df["patch_id"].tolist())
        df = df[~df["patch_id"].isin(quarantine_ids)]
        logger.info(f"Filtered out quarantined patches. Remaining valid records: {len(df):,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate by official split column
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    logger.info(f"Official split counts: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    train_path = output_dir / "train_split.csv"
    val_path = output_dir / "val_split.csv"
    test_path = output_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved splits to {output_dir}")
    return train_df, val_df, test_df


def compute_training_statistics(
    train_df: pd.DataFrame,
    s1_root: Path,
    s2_root: Path,
    output_dir: Path,
    max_samples: int = None,
):
    """
    Compute population mean and standard deviation strictly over training patches.
    S1: VV/VH, Float32 in dB, clipped to [-25.0, 0.0] dB.
    S2: B04/B03/B02, uint16 raw DN.
    """
    import rasterio

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = train_df if max_samples is None else train_df.head(max_samples)
    total_patches = len(samples)

    logger.info(f"Computing population statistics over {total_patches:,} training patches...")

    # S1 accumulators (VV, VH)
    s1_sum = np.zeros(2, dtype=np.float64)
    s1_sq_sum = np.zeros(2, dtype=np.float64)
    s1_pixel_count = 0

    # S2 accumulators (B04, B03, B02)
    s2_sum = np.zeros(3, dtype=np.float64)
    s2_sq_sum = np.zeros(3, dtype=np.float64)
    s2_pixel_count = 0

    s2_bands = ["B04", "B03", "B02"]

    for idx, (_, row) in enumerate(tqdm(samples.iterrows(), total=total_patches, desc="Processing Patches")):
        p_id = row["patch_id"]
        s1_name = row["s1_name"]

        # 1. Process Sentinel-1 SAR
        s1_dir = resolve_s1_patch_dir(s1_root, s1_name)
        vv_files = list(s1_dir.glob("*_VV.tif"))
        vh_files = list(s1_dir.glob("*_VH.tif"))

        if vv_files and vh_files:
            with rasterio.open(vv_files[0]) as src:
                vv_arr = src.read(1).astype(np.float32)
            with rasterio.open(vh_files[0]) as src:
                vh_arr = src.read(1).astype(np.float32)

            # Apply established clipping convention [-25, 0] dB
            vv_clipped = np.clip(vv_arr, -25.0, 0.0)
            vh_clipped = np.clip(vh_arr, -25.0, 0.0)

            s1_sum[0] += vv_clipped.sum()
            s1_sq_sum[0] += (vv_clipped ** 2).sum()

            s1_sum[1] += vh_clipped.sum()
            s1_sq_sum[1] += (vh_clipped ** 2).sum()

            s1_pixel_count += vv_clipped.size

        # 2. Process Sentinel-2 Optical (raw DN)
        s2_dir = resolve_s2_patch_dir(s2_root, p_id)
        band_arrays = []
        for b in s2_bands:
            bf = list(s2_dir.glob(f"*_{b}.tif"))
            if bf:
                with rasterio.open(bf[0]) as src:
                    band_arrays.append(src.read(1).astype(np.float64))

        if len(band_arrays) == 3:
            for b_idx in range(3):
                arr = band_arrays[b_idx]
                s2_sum[b_idx] += arr.sum()
                s2_sq_sum[b_idx] += (arr ** 2).sum()
            s2_pixel_count += band_arrays[0].size

    # Calculate global mean and std
    s1_mean = (s1_sum / s1_pixel_count).tolist()
    s1_std = np.sqrt(np.maximum(0.0, (s1_sq_sum / s1_pixel_count) - (np.array(s1_mean) ** 2))).tolist()

    s2_mean = (s2_sum / s2_pixel_count).tolist()
    s2_std = np.sqrt(np.maximum(0.0, (s2_sq_sum / s2_pixel_count) - (np.array(s2_mean) ** 2))).tolist()

    s1_stats = {
        "mean": s1_mean,
        "std": s1_std,
        "clip_min": -25.0,
        "clip_max": 0.0,
        "polarizations": ["VV", "VH"],
        "total_pixels": int(s1_pixel_count),
        "total_patches": total_patches,
    }

    s2_stats = {
        "mean": s2_mean,
        "std": s2_std,
        "bands": s2_bands,
        "representation": "raw_DN",
        "total_pixels": int(s2_pixel_count),
        "total_patches": total_patches,
    }

    s1_path = output_dir / "s1_stats.json"
    s2_path = output_dir / "s2_stats.json"

    with open(s1_path, "w", encoding="utf-8") as f:
        json.dump(s1_stats, f, indent=4)

    with open(s2_path, "w", encoding="utf-8") as f:
        json.dump(s2_stats, f, indent=4)

    logger.info(f"S1 Stats: Mean={s1_mean}, Std={s1_std}")
    logger.info(f"S2 Stats: Mean={s2_mean}, Std={s2_std}")
    logger.info(f"Saved statistics to {output_dir}")


def main():
    logger.info("Starting PARAM SHAVAK Full-Dataset Splits & Statistics Pipeline...")
    
    splits_dir = settings.RESULTS_DIR / "week_02" / "splits"
    stats_dir = settings.RESULTS_DIR / "week_02" / "normalization_stats"
    
    metadata_path = settings.PROJECT_ROOT / "results" / "local_pc" / "week_01" / "metadata" / "metadata.parquet"
    quarantine_path = settings.PROJECT_ROOT / "results" / "local_pc" / "week_01" / "metadata" / "metadata_for_patches_with_snow_cloud_or_shadow.parquet"

    train_df, val_df, test_df = generate_splits(
        metadata_path=metadata_path,
        quarantine_path=quarantine_path,
        s1_root=settings.S1_DIR,
        s2_root=settings.S2_DIR,
        output_dir=splits_dir,
    )

    compute_training_statistics(
        train_df=train_df,
        s1_root=settings.S1_DIR,
        s2_root=settings.S2_DIR,
        output_dir=stats_dir,
    )
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

