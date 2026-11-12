"""
Centralized, environment-portable configuration settings for SENTINEL-RS.

This module unifies paths, data directories, model constants, and hyperparameters
across all weekly modules (Week 1 to Week 5). It automatically detects the execution
environment (local Windows PC vs. PARAM SHAVAK Linux HPC) while allowing explicit
overrides via environment variables.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# 1. Base Project Directory Resolution
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent
_SHARED_DIR = _CONFIG_DIR.parent
PROJECT_ROOT = Path(os.environ.get("SENTINEL_RS_ROOT", str(_SHARED_DIR.parent))).resolve()

# ---------------------------------------------------------------------------
# 2. Execution Environment Mode
# ---------------------------------------------------------------------------
# Mode: "local" (default) or "param_shavak"
ENVIRONMENT = os.environ.get("SENTINEL_ENV", "local").lower()
IS_HPC = ENVIRONMENT in ("param_shavak", "cluster", "hpc")

# ---------------------------------------------------------------------------
# 3. Dataset Directories
# ---------------------------------------------------------------------------
DEFAULT_LOCAL_S1 = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S1-Required")
DEFAULT_LOCAL_S2 = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2")

S1_DIR = Path(os.environ.get("S1_DIR", str(DEFAULT_LOCAL_S1))).resolve()
S2_DIR = Path(os.environ.get("S2_DIR", str(DEFAULT_LOCAL_S2))).resolve()

# ---------------------------------------------------------------------------
# 4. Results & Artifact Directories (Strict Segregation)
# ---------------------------------------------------------------------------
if IS_HPC:
    RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJECT_ROOT / "param_shavak" / "results"))).resolve()
else:
    RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJECT_ROOT / "results" / "local_pc"))).resolve()

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", str(RESULTS_DIR / "week_02" / "splits"))).resolve()
STATS_DIR = Path(os.environ.get("STATS_DIR", str(RESULTS_DIR / "week_02" / "normalization_stats"))).resolve()
METADATA_DIR = Path(os.environ.get("METADATA_DIR", str(RESULTS_DIR / "week_01" / "metadata"))).resolve()
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", str(RESULTS_DIR / "week_04" / "checkpoints"))).resolve()
EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", str(RESULTS_DIR / "week_05" / "trained_embeddings"))).resolve()
FAISS_DIR = Path(os.environ.get("FAISS_DIR", str(RESULTS_DIR / "week_05" / "faiss_indices"))).resolve()

# Canonical File Paths
S1_STATS_PATH = Path(os.environ.get("S1_STATS_PATH", str(STATS_DIR / "s1_stats.json"))).resolve()
S2_STATS_PATH = Path(os.environ.get("S2_STATS_PATH", str(STATS_DIR / "s2_stats.json"))).resolve()
METADATA_PATH = Path(os.environ.get("METADATA_PATH", str(METADATA_DIR / "metadata.parquet"))).resolve()
HISTORICAL_CHECKPOINT_PATH = (PROJECT_ROOT / "shared" / "models" / "best_model-v1.ckpt").resolve()

# ---------------------------------------------------------------------------
# 5. Scientific Preprocessing & Model Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (224, 224)
S1_CHANNELS: int = 2
S2_CHANNELS: int = 3
S1_BANDS: List[str] = ["VV", "VH"]
S2_BANDS: List[str] = ["B04", "B03", "B02"]
VV_BAND_SUFFIX: str = "_VV.tif"
VH_BAND_SUFFIX: str = "_VH.tif"

# SAR Clipping Convention (decibels)
S1_CLIP_MIN: float = -25.0
S1_CLIP_MAX: float = 0.0

# Multilabel Classification Taxonomy
NUM_CLASSES: int = 19
BIGEARTHNET_19_CLASSES: List[str] = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

# Model Architecture & Metric Loss
EMBEDDING_DIM: int = 512
TEMPERATURE: float = 0.07

# ---------------------------------------------------------------------------
# 6. DataLoader & Training Defaults
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "64" if IS_HPC else "4"))
DEFAULT_NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "8" if (IS_HPC and os.name != "nt") else "0"))
PIN_MEMORY: bool = IS_HPC and (os.environ.get("PIN_MEMORY", "1") == "1")

# ---------------------------------------------------------------------------
# 7. Logging Helper
# ---------------------------------------------------------------------------
def setup_logging(name: str = "SENTINEL-RS", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
