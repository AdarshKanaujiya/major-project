from pathlib import Path
import json

import numpy as np
import rasterio

from rasterio.enums import Resampling
import torch
import torch.nn.functional as F



# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2"
)


_CANONICAL_STATS = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "local_pc"
    / "week_02"
    / "normalization_stats"
    / "s2_stats.json"
)
STATS_FILE = (
    _CANONICAL_STATS
    if _CANONICAL_STATS.exists()
    else (Path(__file__).resolve().parent / "s2_stats.json")
)



# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

if not Path(STATS_FILE).exists():
    # Fallback to local cwd if present
    if Path("s2_stats.json").exists():
        STATS_FILE = Path("s2_stats.json")
    else:
        raise FileNotFoundError(f"Sentinel-2 statistics file not found at: {STATS_FILE}")

with open(STATS_FILE, "r") as f:
    stats = json.load(f)


MEAN = np.array(stats["mean"], dtype=np.float32)
STD = np.array(stats["std"], dtype=np.float32)


print("Mean:", MEAN)
print("Std:", STD)



# ============================================================
# READ RGB PATCH
# ============================================================


def read_rgb_patch(patch_folder):
    """
    Reads Sentinel-2 RGB bands.

    Bands:
    B04 -> Red
    B03 -> Green
    B02 -> Blue

    Output:
    Shape:
    (3,120,120)
    """


    bands = [
        "B02",
        "B03",
        "B04"
    ]


    images = []


    for band in bands:
        matches = list(patch_folder.glob(f"*_{band}.tif"))
        if not matches:
            raise FileNotFoundError(f"Could not find Sentinel-2 band '{band}' in: {patch_folder}")
        tif = matches[0]

        with rasterio.open(tif) as src:
            image = src.read(1)

        images.append(image)



    # Current order:
    #
    # B02
    # B03
    # B04
    #
    # Convert:
    #
    # Red
    # Green
    # Blue


    rgb = np.stack(images)

    rgb = rgb[
        [2,1,0]
    ]


    return rgb.astype(np.float32)



# ============================================================
# RESIZE IMAGE
# ============================================================


def resize_image(image):
    """
    Resize:

    (3,120,120)

    to

    (3,224,224)

    using bilinear interpolation
    """


    tensor = torch.tensor(
        image
    )


    tensor = tensor.unsqueeze(0)


    resized = F.interpolate(
        tensor,
        size=(224,224),
        mode="bilinear",
        align_corners=False
    )


    resized = resized.squeeze(0)


    return resized.numpy()



# ============================================================
# NORMALIZATION
# ============================================================


def normalize(image):
    """
    Apply:

    (pixel - mean) / std

    Channel wise normalization
    """


    for c in range(3):

        image[c] = (
            image[c] - MEAN[c]
        ) / STD[c]


    return image



# ============================================================
# COMPLETE PREPROCESS FUNCTION
# ============================================================


def preprocess_patch(patch_folder):


    # Step 1:
    # Read RGB

    image = read_rgb_patch(
        patch_folder
    )


    # Step 2:
    # Resize

    image = resize_image(
        image
    )


    # Step 3:
    # Normalize

    image = normalize(
        image
    )


    return image



if __name__ == "__main__":


    patch_folder = Path(
        r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_29_55"
    )


    output = preprocess_patch(
        patch_folder
    )


    print(
        "Final shape:",
        output.shape
    )


    print(
        "Minimum value:",
        output.min()
    )


    print(
        "Maximum value:",
        output.max()
    )


    print(
        "Contains NaN:",
        np.isnan(output).any()
    )