"""Preprocessing for three-band Sentinel-2 patches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

# Canonical path to verified population statistics
STATS_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "local_pc"
    / "week_02"
    / "normalization_stats"
    / "s2_stats.json"
)


def _load_s2_stats() -> tuple[np.ndarray, np.ndarray]:
    """Load verified Sentinel-2 population statistics (B04, B03, B02)."""
    target = STATS_PATH if STATS_PATH.exists() else Path(__file__).resolve().parent / "s2_stats.json"
    if not target.exists():
        raise FileNotFoundError(f"Sentinel-2 statistics file not found at: {target}")
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(data["std"], dtype=np.float32).reshape(3, 1, 1)
    if np.any(std <= 0):
        raise ValueError("Standard deviations in s2_stats.json must be positive.")
    return mean, std


def _find_band(folder: Path, band: str) -> Path:
    candidates = sorted(path for path in folder.rglob("*") if path.is_file() and band in path.stem)
    if not candidates:
        raise FileNotFoundError(f"Could not find Sentinel-2 band {band} under {folder}")
    return candidates[0]


def _read_band(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-band raster at {path}, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def preprocess_s2_patch(
    patch_folder: Path,
    target_size: tuple[int, int] = (224, 224),
    bands: Iterable[str] = ("B04", "B03", "B02"),
) -> torch.Tensor:
    bands = tuple(bands)
    arrays = [_read_band(_find_band(Path(patch_folder), band)) for band in bands]
    resized = [
        np.asarray(
            Image.fromarray(array).resize(
                (target_size[1], target_size[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        for array in arrays
    ]
    output = np.stack(resized, axis=0)
    mean, std = _load_s2_stats()
    output = (output - mean) / std
    if not np.isfinite(output).all():
        raise ValueError("S2 preprocessing produced NaN or infinite values.")
    return torch.from_numpy(output.astype(np.float32, copy=False))
