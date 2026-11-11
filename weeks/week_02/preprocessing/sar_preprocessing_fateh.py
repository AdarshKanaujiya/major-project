"""Preprocessing for two-band Sentinel-1 patches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

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
    / "s1_stats.json"
)

CLIP_MIN = -25.0
CLIP_MAX = 0.0


def _load_s1_stats() -> Tuple[np.ndarray, np.ndarray]:
    """Load verified Sentinel-1 population statistics (VV, VH)."""
    target = STATS_PATH if STATS_PATH.exists() else Path(__file__).resolve().parent / "s1_stats.json"
    if not target.exists():
        raise FileNotFoundError(f"Sentinel-1 statistics file not found at: {target}")
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    vv_mean = float(data["VV"]["mean"])
    vv_std = float(data["VV"]["std"])
    vh_mean = float(data["VH"]["mean"])
    vh_std = float(data["VH"]["std"])
    if vv_std <= 0 or vh_std <= 0:
        raise ValueError("Standard deviations in s1_stats.json must be positive.")
    mean = np.array([vv_mean, vh_mean], dtype=np.float32).reshape(2, 1, 1)
    std = np.array([vv_std, vh_std], dtype=np.float32).reshape(2, 1, 1)
    return mean, std


def _read_band(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-band raster at {path}, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _resize(array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array)
    resized = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def preprocess_sar_patch(
    vv_path: Path,
    vh_path: Path,
    target_size: Tuple[int, int] = (224, 224),
    preserve_res: bool = False,
    norm_mode: str = "population",
) -> torch.Tensor:
    vv = _read_band(Path(vv_path))
    vh = _read_band(Path(vh_path))

    # Apply verified [-25, 0] dB clipping
    vv = np.clip(vv, CLIP_MIN, CLIP_MAX)
    vh = np.clip(vh, CLIP_MIN, CLIP_MAX)

    if not preserve_res:
        vv = _resize(vv, target_size)
        vh = _resize(vh, target_size)

    output = np.stack([vv, vh], axis=0)

    if norm_mode == "population":
        mean, std = _load_s1_stats()
        output = (output - mean) / std
    elif norm_mode == "per_band":
        mean = output.mean(axis=(1, 2), keepdims=True)
        std = output.std(axis=(1, 2), keepdims=True)
        output = (output - mean) / np.maximum(std, 1e-6)

    if not np.isfinite(output).all():
        raise ValueError("SAR preprocessing produced NaN or infinite values.")

    return torch.from_numpy(output.astype(np.float32, copy=False))
