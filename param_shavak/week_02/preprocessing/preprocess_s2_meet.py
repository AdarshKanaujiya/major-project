"""
preprocess_s2.py - Sentinel-2 Multispectral (RGB) Preprocessing Pipeline.

Ingests raw Sentinel-2 10m RGB bands (B04, B03, B02), resizes to 224x224,
and applies channel-wise standardization using population statistics from
teammate_inputs/s2_stats.json.

Channel Configuration:
- Channel 0: B04 (Red)
- Channel 1: B03 (Green)
- Channel 2: B02 (Blue)

Expected Output Shape:
- (3, 224, 224), float32 PyTorch tensor
"""

from pathlib import Path
import json
import logging
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

# Base Project Directory and Canonical S2 Stats Path
BASE_DIR = Path(__file__).resolve().parent.parent
_CANONICAL_STATS = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "local_pc"
    / "week_02"
    / "normalization_stats"
    / "s2_stats.json"
)
DEFAULT_S2_STATS_PATH = (
    _CANONICAL_STATS
    if _CANONICAL_STATS.exists()
    else (BASE_DIR / "teammate_inputs" / "s2_stats.json")
)

logger = logging.getLogger("S2Preprocessing")


def load_s2_stats(stats_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads Sentinel-2 RGB normalization statistics (mean and std).

    Args:
        stats_path: Path to s2_stats.json. Defaults to canonical normalization_stats/s2_stats.json.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std) arrays of shape (3,).
    """
    target_path = Path(stats_path) if stats_path else DEFAULT_S2_STATS_PATH
    if not target_path.exists():
        # Also check local directory fallback
        local_fallback = Path(__file__).resolve().parent / "s2_stats.json"
        if local_fallback.exists():
            target_path = local_fallback

    if not target_path.exists():
        raise FileNotFoundError(
            f"Sentinel-2 normalization statistics file not found: {target_path}. "
            "Silent fallback to divergent hardcoded values is disabled for scientific integrity."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "mean" not in data or "std" not in data:
        raise ValueError(f"s2_stats.json at {target_path} must contain 'mean' and 'std' keys.")

    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Expected exactly 3 mean and 3 std values for B04, B03, B02.")

    if np.any(std <= 0):
        raise ValueError("Sentinel-2 standard deviations must be positive.")

    return mean, std


def read_rgb_patch(patch_folder: Union[str, Path]) -> np.ndarray:
    """
    Reads Sentinel-2 RGB bands in exact order: B04 (Red), B03 (Green), B02 (Blue).

    Args:
        patch_folder: Directory containing the patch's GeoTIFF band files.

    Returns:
        np.ndarray: Array of shape (3, H, W), float32.
    """
    folder = Path(patch_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Sentinel-2 patch folder does not exist: {folder}")

    # Explicit band order: Red (B04), Green (B03), Blue (B02)
    bands = ["B04", "B03", "B02"]
    channels = []

    for band in bands:
        matches = list(folder.glob(f"*_{band}.tif"))
        if not matches:
            raise FileNotFoundError(f"Missing Sentinel-2 band '{band}' in patch folder: {folder}")
        tif_path = matches[0]

        with rasterio.open(tif_path) as src:
            band_arr = src.read(1).astype(np.float32)
        channels.append(band_arr)

    # Stack into shape (3, H, W)
    rgb = np.stack(channels, axis=0)
    return rgb


def resize_s2_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes an RGB image array from (3, H, W) to (3, target_H, target_W) using bilinear interpolation.

    Args:
        image: NumPy array of shape (3, H, W).
        target_size: Target (height, width).

    Returns:
        np.ndarray: Resized array of shape (3, target_H, target_W), float32.
    """
    tensor = torch.from_numpy(image).unsqueeze(0)  # (1, 3, H, W)
    resized = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).numpy()


def normalize_s2_image(
    image: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Applies channel-wise Z-score normalization: (pixel - mean) / std.

    Args:
        image: Array of shape (3, H, W).
        mean: Channel means of shape (3,).
        std: Channel standard deviations of shape (3,).

    Returns:
        np.ndarray: Normalized array of shape (3, H, W), float32.
    """
    if mean is None or std is None:
        mean_loaded, std_loaded = load_s2_stats()
        if mean is None:
            mean = mean_loaded
        if std is None:
            std = std_loaded

    norm_img = np.empty_like(image, dtype=np.float32)
    for c in range(3):
        # Guard against division by zero
        denom = std[c] if std[c] > 1e-7 else 1.0
        norm_img[c] = (image[c] - mean[c]) / denom

    return norm_img


def preprocess_s2_patch(
    patch_folder: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    stats_path: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """
    Complete Sentinel-2 RGB Preprocessing Pipeline:
    1. Reads B04 (Red), B03 (Green), and B02 (Blue) GeoTIFFs -> (3, 120, 120).
    2. Resizes to target_size (default: (224, 224)) via bilinear interpolation.
    3. Normalizes using channel-wise mean and std from s2_stats.json.
    4. Returns a float32 PyTorch tensor of shape (3, 224, 224).

    Args:
        patch_folder: Path to the patch folder containing S2 band GeoTIFFs.
        target_size: Desired spatial resolution (H, W).
        stats_path: Optional path to s2_stats.json.

    Returns:
        torch.Tensor: Normalized PyTorch FloatTensor of shape (3, 224, 224).
    """
    mean, std = load_s2_stats(stats_path)
    rgb = read_rgb_patch(patch_folder)
    resized = resize_s2_image(rgb, target_size=target_size)
    normalized = normalize_s2_image(resized, mean=mean, std=std)
    if not np.isfinite(normalized).all():
        raise ValueError("S2 preprocessing produced NaN or infinite values.")
    return torch.from_numpy(normalized).type(torch.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Sentinel-2 Preprocessing Module loaded successfully.")
    mean, std = load_s2_stats()
    print(f"Loaded S2 Stats: Mean={mean.tolist()}, Std={std.tolist()}")