"""
weeks/week_05/retrieval/retriever_meet.py

retriever_meet.py - Cross-Modal Satellite Image Retriever with Hybrid Support.

Core Week 5 inference module integrating:
- Trained TwoTowerNetwork (ResNet-50 towers + 512-D projections)
- Model checkpoint (best_model-v1.ckpt)
- Sentinel-1 (SAR: VV+VH) and Sentinel-2 (MS: B04,B03,B02) preprocessing
- FAISS L2-normalized IndexFlatIP search (inner product = cosine similarity)
  * SAR query -> MS gallery (ms_gallery.index)
  * MS query -> SAR gallery (sar_gallery.index)
  * Hybrid query -> Dual gallery search (MS + SAR) with deterministic score merge
- Non-destructive identifier mapping (test_patch_id_mapping.csv)
- Metadata resolution from test_split.csv
- Real-time thumbnail rendering from local raw patch directories
- Domain and subcategory configuration routing
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    from rasterio.errors import NotGeoreferencedWarning
except ImportError:
    NotGeoreferencedWarning = UserWarning

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from PIL import Image

# Canonical TwoTowerNetwork import
try:
    from weeks.week_04.models.two_tower_jainam import TwoTowerNetwork
except ImportError:
    TwoTowerNetwork = None

# Canonical preprocessing imports
try:
    from weeks.week_02.preprocessing.sar_preprocessing_meet import (
        clip_sar_band,
        load_s1_stats,
        load_sar_band,
        normalize_sar_band,
        preprocess_sar_patch,
        resize_sar_band,
    )
except ImportError:
    clip_sar_band = None
    load_s1_stats = None
    load_sar_band = None
    normalize_sar_band = None
    preprocess_sar_patch = None
    resize_sar_band = None

try:
    from weeks.week_02.preprocessing.preprocess_s2_meet import (
        load_s2_stats,
        normalize_s2_image,
        preprocess_s2_patch,
        read_rgb_patch,
        resize_s2_image,
    )
except ImportError:
    load_s2_stats = None
    normalize_s2_image = None
    preprocess_s2_patch = None
    read_rgb_patch = None
    resize_s2_image = None

# Identifier mapper
try:
    from weeks.week_05.retrieval.identifier_mapper import get_identifier_mapper, IdentifierMapper
except ImportError:
    get_identifier_mapper = None
    IdentifierMapper = None


logger = logging.getLogger("CrossModalRetriever")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model-v1.ckpt"
DEFAULT_SAR_INDEX_PATH = RESULTS_DIR / "week_05" / "faiss_indices" / "sar_gallery.index"
DEFAULT_MS_INDEX_PATH = RESULTS_DIR / "week_05" / "faiss_indices" / "ms_gallery.index"

DEFAULT_SAR_PATCH_IDS_PATH = RESULTS_DIR / "week_05" / "trained_embeddings" / "trained_test_sar_patch_ids.txt"
DEFAULT_MS_PATCH_IDS_PATH = RESULTS_DIR / "week_05" / "trained_embeddings" / "trained_test_ms_patch_ids.txt"
DEFAULT_TEST_SPLIT_PATH = SPLITS_DIR / "test_split.csv"

DEFAULT_S1_STATS_PATH = RESULTS_DIR / "week_02" / "normalization_stats" / "s1_stats.json"
DEFAULT_S2_STATS_PATH = RESULTS_DIR / "week_02" / "normalization_stats" / "s2_stats.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "shared" / "config" / "disaster_modality_config.json"


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def load_modality_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads authoritative disaster modality routing configuration."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        logger.warning(f"Configuration not found at: {target_path}. Returning default empty config.")
        return {"domains": {}}
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_subcategory_info(
    subcategory_name_or_slug: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolves domain and subcategory details given a subcategory slug or name.
    Matches case-insensitively across configured domains.
    """
    if config is None:
        config = load_modality_config()

    target = subcategory_name_or_slug.strip().lower().replace("_", "-")

    alias_map = {
        "flood-mapping": "flood",
        "wildfire-scar": "wildfire",
        "forest-change": "deforestation",
        "water-quality": "drought",
        "crop-monitoring": "crop-mapping",
    }
    target = alias_map.get(target, target)

    for domain_slug, domain_info in config.get("domains", {}).items():
        for sub_slug, sub_info in domain_info.get("subcategories", {}).items():
            if (
                target == sub_slug.lower()
                or target == sub_info.get("name", "").strip().lower()
                or target == sub_slug.replace("-", " ")
            ):
                result = dict(sub_info)
                result["domain_slug"] = domain_slug
                result["domain_name"] = domain_info.get("name", "")
                result["subcategory_slug"] = sub_slug
                return result

    return None


# ============================================================
# RETRIEVER CLASS
# ============================================================

class CrossModalRetriever:
    """
    Cross-modal retriever for Sentinel satellite imagery.
    Loads trained TwoTowerNetwork and FAISS galleries to perform:
    - SAR -> MS retrieval (cross-modal)
    - MS -> SAR retrieval (cross-modal)
    - Hybrid retrieval (dual gallery search + deterministic score merge)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        sar_index_path: Optional[Union[str, Path]] = None,
        ms_index_path: Optional[Union[str, Path]] = None,
        sar_patch_ids_path: Optional[Union[str, Path]] = None,
        ms_patch_ids_path: Optional[Union[str, Path]] = None,
        test_split_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise RuntimeError("[BLOCKED BY ENVIRONMENT] PyTorch is required to initialize CrossModalRetriever.")
        if faiss is None:
            raise RuntimeError("[BLOCKED BY ENVIRONMENT] FAISS is required to initialize CrossModalRetriever.")

        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.sar_index_path = Path(sar_index_path) if sar_index_path else DEFAULT_SAR_INDEX_PATH
        self.ms_index_path = Path(ms_index_path) if ms_index_path else DEFAULT_MS_INDEX_PATH
        self.sar_patch_ids_path = Path(sar_patch_ids_path) if sar_patch_ids_path else DEFAULT_SAR_PATCH_IDS_PATH
        self.ms_patch_ids_path = Path(ms_patch_ids_path) if ms_patch_ids_path else DEFAULT_MS_PATCH_IDS_PATH
        self.test_split_path = Path(test_split_path) if test_split_path else DEFAULT_TEST_SPLIT_PATH

        # 1. Identifier mapper
        self.mapper = get_identifier_mapper() if get_identifier_mapper else None

        # 2. Load model
        self.model = self._load_model()

        # 3. Load FAISS indices
        self.sar_index, self.ms_index = self._load_indices()

        # 4. Load patch IDs
        self.sar_patch_ids, self.ms_patch_ids = self._load_patch_ids()

        # 5. Load metadata dictionary from test_split.csv
        self.metadata_lookup = self._load_metadata()

        # 6. Pre-load normalization stats
        if load_s1_stats and DEFAULT_S1_STATS_PATH.exists():
            self.s1_stats = load_s1_stats(DEFAULT_S1_STATS_PATH)
        else:
            self.s1_stats = {"VV": {"mean": -11.5, "std": 4.5}, "VH": {"mean": -18.2, "std": 5.0}}

        if load_s2_stats and DEFAULT_S2_STATS_PATH.exists():
            self.s2_mean, self.s2_std = load_s2_stats(DEFAULT_S2_STATS_PATH)
        else:
            self.s2_mean = np.array([1353.7, 1117.2, 1041.8], dtype=np.float32)
            self.s2_std = np.array([653.2, 532.1, 510.4], dtype=np.float32)

        # 7. Lazy patch cache for image thumbnails
        self._local_s1_patches: Optional[Dict[str, Path]] = None
        self._local_s2_patches: Optional[Dict[str, Path]] = None

        # Load authoritative modality config
        self.config = load_modality_config()

        logger.info(
            "CrossModalRetriever initialized successfully. "
            f"SAR Gallery: {self.sar_index.ntotal} vectors, MS Gallery: {self.ms_index.ntotal} vectors."
        )

    # --------------------------------------------------------
    # INITIALIZATION HELPERS
    # --------------------------------------------------------

    def _load_model(self) -> TwoTowerNetwork:
        """Load trained TwoTowerNetwork checkpoint in eval mode."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)

        if "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint at {self.checkpoint_path} missing 'state_dict' key.")

        raw_state_dict = checkpoint["state_dict"]
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith("network."):
                clean_state_dict[k[len("network."):]] = v
            elif not k.startswith("loss_fn."):
                clean_state_dict[k] = v

        model = TwoTowerNetwork(pretrained=False)
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        if missing_keys:
            raise RuntimeError(f"State dict mismatch! Missing keys: {missing_keys}")

        model.to(self.device)
        model.eval()
        return model

    def _load_indices(self) -> Tuple[faiss.Index, faiss.Index]:
        """Load FAISS gallery indices."""
        if not self.sar_index_path.exists():
            raise FileNotFoundError(f"SAR gallery index not found at: {self.sar_index_path}")
        if not self.ms_index_path.exists():
            raise FileNotFoundError(f"MS gallery index not found at: {self.ms_index_path}")

        sar_index = faiss.read_index(str(self.sar_index_path))
        ms_index = faiss.read_index(str(self.ms_index_path))

        if sar_index.d != 512 or ms_index.d != 512:
            raise ValueError(f"Expected index dimension 512, got SAR={sar_index.d}, MS={ms_index.d}")

        return sar_index, ms_index

    def _load_patch_ids(self) -> Tuple[List[str], List[str]]:
        """Load ordered test patch IDs for SAR and MS galleries."""
        if not self.sar_patch_ids_path.exists():
            raise FileNotFoundError(f"SAR patch IDs file not found: {self.sar_patch_ids_path}")
        if not self.ms_patch_ids_path.exists():
            raise FileNotFoundError(f"MS patch IDs file not found: {self.ms_patch_ids_path}")

        with open(self.sar_patch_ids_path, "r", encoding="utf-8") as f:
            sar_ids = [line.strip() for line in f if line.strip()]

        with open(self.ms_patch_ids_path, "r", encoding="utf-8") as f:
            ms_ids = [line.strip() for line in f if line.strip()]

        if len(sar_ids) != self.sar_index.ntotal:
            raise ValueError(
                f"SAR patch IDs count ({len(sar_ids)}) != SAR index count ({self.sar_index.ntotal})"
            )
        if len(ms_ids) != self.ms_index.ntotal:
            raise ValueError(
                f"MS patch IDs count ({len(ms_ids)}) != MS index count ({self.ms_index.ntotal})"
            )

        return sar_ids, ms_ids

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load and index test_split.csv metadata by patch_id."""
        if not self.test_split_path.exists():
            raise FileNotFoundError(f"test_split.csv not found at: {self.test_split_path}")

        df = pd.read_csv(self.test_split_path)
        lookup: Dict[str, Dict[str, Any]] = {}

        for _, row in df.iterrows():
            pid = str(row["patch_id"]).strip()
            raw_labels = str(row.get("labels", ""))
            extracted_labels = re.findall(r"'([^']+)'", raw_labels)
            if not extracted_labels and raw_labels:
                extracted_labels = [l.strip(" []'\"") for l in raw_labels.split(",") if l.strip(" []'\"")]

            primary_label = extracted_labels[0] if extracted_labels else "Unlabeled"

            lookup[pid] = {
                "patch_id": pid,
                "label": primary_label,
                "labels": extracted_labels,
                "country": str(row.get("country", "")),
                "s1_name": str(row.get("s1_name", "")),
                "s2v1_name": str(row.get("s2v1_name", "")),
                "split": str(row.get("split", "test")),
            }

        return lookup

    # --------------------------------------------------------
    # LOCAL THUMBNAIL CACHE & RENDERER
    # --------------------------------------------------------

    def _ensure_patch_cache(self) -> None:
        """Lazily builds lookup table for local raw patch directories."""
        if self._local_s1_patches is not None and self._local_s2_patches is not None:
            return

        self._local_s1_patches = {}
        self._local_s2_patches = {}

        for s1_root in [S1_DIR, PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S1-Required"]:
            if s1_root.exists():
                for p in s1_root.rglob("*"):
                    if p.is_dir() and (list(p.glob("*_VV.tif")) or list(p.glob("*_vh.tif"))):
                        self._local_s1_patches[p.name] = p

        for s2_root in [S2_DIR, PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S2-Required"]:
            if s2_root.exists():
                for p in s2_root.rglob("*"):
                    if p.is_dir() and list(p.glob("*_B04.tif")):
                        self._local_s2_patches[p.name] = p

    def render_patch_thumbnail(self, patch_id: str, modality: Optional[str] = None) -> Optional[Image.Image]:
        """
        Renders a PIL RGB thumbnail for a given patch_id if available locally.
        Uses identifier mapping to resolve native S1 folder name when needed.
        """
        self._ensure_patch_cache()
        meta = self.metadata_lookup.get(patch_id, {})

        # S2 lookup
        if modality is None or modality.strip().upper() in ("MS", "MULTISPECTRAL", "S2"):
            if patch_id in self._local_s2_patches:
                folder = self._local_s2_patches[patch_id]
                try:
                    b04_files = list(folder.glob("*_B04.tif"))
                    b03_files = list(folder.glob("*_B03.tif"))
                    b02_files = list(folder.glob("*_B02.tif"))
                    if b04_files and b03_files and b02_files:
                        import rasterio
                        b04 = rasterio.open(b04_files[0]).read(1).astype(np.float32)
                        b03 = rasterio.open(b03_files[0]).read(1).astype(np.float32)
                        b02 = rasterio.open(b02_files[0]).read(1).astype(np.float32)
                        rgb = np.stack([b04, b03, b02], axis=-1)
                        p2, p98 = np.percentile(rgb, (2, 98))
                        rgb_scaled = np.clip((rgb - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
                        return Image.fromarray(rgb_scaled)
                except Exception as e:
                    logger.warning(f"Error rendering S2 thumbnail for {patch_id}: {e}")

        # S1 lookup (resolve native radar folder name via mapper if available)
        s1_name = meta.get("s1_name", "")
        if not s1_name and self.mapper:
            s1_name = self.mapper.to_s1_name(patch_id) or ""
        if not s1_name:
            s1_name = patch_id

        candidate_s1 = self._local_s1_patches.get(s1_name, self._local_s1_patches.get(patch_id))
        if candidate_s1 is not None:
            try:
                vv_files = list(candidate_s1.glob("*_VV.tif"))
                if vv_files:
                    import rasterio
                    vv = rasterio.open(vv_files[0]).read(1).astype(np.float32)
                    vv_clipped = np.clip(vv, -25.0, 0.0)
                    vv_scaled = ((vv_clipped - (-25.0)) / 25.0 * 255.0).astype(np.uint8)
                    return Image.fromarray(vv_scaled).convert("RGB")
            except Exception as e:
                logger.warning(f"Error rendering S1 thumbnail for {patch_id}: {e}")

        return None

    # --------------------------------------------------------
    # PREPROCESSING METHODS
    # --------------------------------------------------------

    def preprocess_sar(self, query_input: Union[str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocesses SAR query to (1, 2, 224, 224) float32 tensor."""
        if torch is None:
            raise RuntimeError("PyTorch is required for preprocessing.")

        if isinstance(query_input, torch.Tensor):
            t = query_input.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != 2 or t.shape[2:] != (224, 224):
                raise ValueError(f"SAR tensor must have shape (1, 2, 224, 224), got {t.shape}")
            return t.to(self.device)

        if isinstance(query_input, np.ndarray):
            arr = query_input.astype(np.float32)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, 0)
            t = torch.from_numpy(arr)
            return self.preprocess_sar(t)

        input_path = Path(query_input)
        if not input_path.exists():
            raise FileNotFoundError(f"SAR query input path does not exist: {input_path}")

        if input_path.is_dir():
            vv_files = list(input_path.glob("*_VV.tif")) + list(input_path.glob("*VV*.tif"))
            vh_files = list(input_path.glob("*_VH.tif")) + list(input_path.glob("*VH*.tif"))
            if not vv_files or not vh_files:
                raise FileNotFoundError(f"Directory {input_path} missing VV or VH GeoTIFFs.")
            if preprocess_sar_patch is None:
                raise RuntimeError("preprocess_sar_patch could not be imported.")
            tensor = preprocess_sar_patch(vv_files[0], vh_files[0])
            return tensor.unsqueeze(0).to(self.device)

        if input_path.is_file():
            name = input_path.name
            parent = input_path.parent
            if "_VV" in name:
                partner_name = name.replace("_VV", "_VH")
                if (parent / partner_name).exists() and preprocess_sar_patch:
                    tensor = preprocess_sar_patch(input_path, parent / partner_name)
                    return tensor.unsqueeze(0).to(self.device)
            elif "_VH" in name:
                partner_name = name.replace("_VH", "_VV")
                if (parent / partner_name).exists() and preprocess_sar_patch:
                    tensor = preprocess_sar_patch(parent / partner_name, input_path)
                    return tensor.unsqueeze(0).to(self.device)

            try:
                import rasterio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(input_path) as src:
                        if src.count >= 2:
                            vv_raw = src.read(1).astype(np.float32)
                            vh_raw = src.read(2).astype(np.float32)
                        else:
                            vv_raw = src.read(1).astype(np.float32)
                            vh_raw = vv_raw.copy()
            except Exception:
                img = Image.open(input_path).convert("L")
                vv_raw = np.array(img, dtype=np.float32)
                vh_raw = vv_raw.copy()

            if clip_sar_band and normalize_sar_band and resize_sar_band:
                vv_clipped = clip_sar_band(vv_raw, (-25.0, 0.0))
                vh_clipped = clip_sar_band(vh_raw, (-25.0, 0.0))
                vv_norm = normalize_sar_band(vv_clipped, (-25.0, 0.0), "z-score", self.s1_stats.get("VV", {"mean": -11.5, "std": 4.5}))
                vh_norm = normalize_sar_band(vh_clipped, (-25.0, 0.0), "z-score", self.s1_stats.get("VH", {"mean": -18.2, "std": 5.0}))
                vv_final = resize_sar_band(vv_norm, (224, 224))
                vh_final = resize_sar_band(vh_norm, (224, 224))
                stacked = np.stack([vv_final, vh_final], axis=0).astype(np.float32)
                return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        raise ValueError(f"Unsupported SAR input: {query_input}")

    def preprocess_ms(self, query_input: Union[str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocesses Multispectral query to (1, 3, 224, 224) float32 tensor."""
        if torch is None:
            raise RuntimeError("PyTorch is required for preprocessing.")

        if isinstance(query_input, torch.Tensor):
            t = query_input.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != 3 or t.shape[2:] != (224, 224):
                raise ValueError(f"MS tensor must have shape (1, 3, 224, 224), got {t.shape}")
            return t.to(self.device)

        if isinstance(query_input, np.ndarray):
            arr = query_input.astype(np.float32)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, 0)
            t = torch.from_numpy(arr)
            return self.preprocess_ms(t)

        input_path = Path(query_input)
        if not input_path.exists():
            raise FileNotFoundError(f"MS query input path does not exist: {input_path}")

        if input_path.is_dir():
            if preprocess_s2_patch is None:
                raise RuntimeError("preprocess_s2_patch could not be imported.")
            tensor = preprocess_s2_patch(input_path, target_size=(224, 224))
            return tensor.unsqueeze(0).to(self.device)

        if input_path.is_file():
            parent = input_path.parent
            b04 = list(parent.glob("*_B04.tif"))
            b03 = list(parent.glob("*_B03.tif"))
            b02 = list(parent.glob("*_B02.tif"))
            if b04 and b03 and b02 and any(b.name == input_path.name for b in (b04[0], b03[0], b02[0])) and preprocess_s2_patch:
                tensor = preprocess_s2_patch(parent, target_size=(224, 224))
                return tensor.unsqueeze(0).to(self.device)

            try:
                import rasterio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(input_path) as src:
                        if src.count >= 3:
                            rgb = src.read([1, 2, 3]).astype(np.float32)
                        else:
                            single = src.read(1).astype(np.float32)
                            rgb = np.stack([single, single, single], axis=0)
            except Exception:
                img = Image.open(input_path).convert("RGB")
                rgb = np.array(img, dtype=np.float32).transpose(2, 0, 1)

            if resize_s2_image and normalize_s2_image:
                resized = resize_s2_image(rgb, target_size=(224, 224))
                normalized = normalize_s2_image(resized, mean=self.s2_mean, std=self.s2_std)
                return torch.from_numpy(normalized).type(torch.float32).unsqueeze(0).to(self.device)

        raise ValueError(f"Unsupported MS input: {query_input}")

    # --------------------------------------------------------
    # EMBEDDING INFERENCE
    # --------------------------------------------------------

    def encode_sar(self, sar_tensor: torch.Tensor) -> np.ndarray:
        """Runs SAR tower + projection -> 512-D L2-normalized embedding."""
        sar_tensor = sar_tensor.to(self.device)
        with torch.no_grad():
            features = self.model.sar_encoder(sar_tensor)
            proj = self.model.sar_projection(features)
            normalized = F.normalize(proj, p=2, dim=1)
            emb = normalized.cpu().numpy().astype(np.float32)
        return emb

    def encode_ms(self, ms_tensor: torch.Tensor) -> np.ndarray:
        """Runs MS tower + projection -> 512-D L2-normalized embedding."""
        ms_tensor = ms_tensor.to(self.device)
        with torch.no_grad():
            features = self.model.ms_encoder(ms_tensor)
            proj = self.model.ms_projection(features)
            normalized = F.normalize(proj, p=2, dim=1)
            emb = normalized.cpu().numpy().astype(np.float32)
        return emb

    # --------------------------------------------------------
    # FAISS GALLERY SEARCH
    # --------------------------------------------------------

    def search_gallery(
        self,
        query_embedding: np.ndarray,
        target_gallery: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search target FAISS gallery and map indices to patch IDs and metadata."""
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, 0)

        if query_embedding.shape != (1, 512):
            raise ValueError(f"Query embedding shape must be (1, 512), got {query_embedding.shape}")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains NaN or Inf.")

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        target = target_gallery.strip().upper()
        if target == "MS":
            index = self.ms_index
            patch_ids = self.ms_patch_ids
            result_modality = "Multispectral"
        elif target == "SAR":
            index = self.sar_index
            patch_ids = self.sar_patch_ids
            result_modality = "SAR"
        else:
            raise ValueError(f"Unsupported target gallery: {target_gallery}. Must be 'MS' or 'SAR'.")

        scores, indices = index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(patch_ids):
                continue

            patch_id = patch_ids[idx]
            meta = self.metadata_lookup.get(patch_id, {})
            sim_score = float(score)

            # Determine native radar name via mapper if meta lacks it
            s1_name = meta.get("s1_name", "")
            if not s1_name and self.mapper:
                s1_name = self.mapper.to_s1_name(patch_id) or ""

            result_entry: Dict[str, Any] = {
                "rank": rank,
                "patch_id": patch_id,
                "similarity_score": sim_score,
                "score": f"{sim_score:.4f}",
                "modality": result_modality,
                "label": meta.get("label", "Unlabeled"),
                "labels": meta.get("labels", []),
                "country": meta.get("country", ""),
                "s1_name": s1_name,
                "s2v1_name": meta.get("s2v1_name", ""),
                "split": meta.get("split", "test"),
                "tone": ((rank - 1) % 5) + 1,
                "provenance": f"gallery:{target}",
            }
            results.append(result_entry)

        return results

    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    def retrieve_hybrid(
        self,
        query_image_path: Union[str, Path, torch.Tensor, np.ndarray],
        query_modality: str = "auto",
        subcategory: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Hybrid search with dual gallery query and deterministic score merge."""
        modality = query_modality.strip().upper()
        if modality in ("AUTO", "HYBRID", "DUAL"):
            if isinstance(query_image_path, (torch.Tensor, np.ndarray)):
                channels = query_image_path.shape[1] if query_image_path.ndim >= 3 else query_image_path.shape[0]
                modality = "SAR" if channels == 2 else "MS"
            else:
                p_str = str(query_image_path).upper()
                if "_VV" in p_str or "_VH" in p_str or "S1" in p_str:
                    modality = "SAR"
                else:
                    modality = "MS"

        if modality in ("SAR", "S1", "SENTINEL-1"):
            sar_tensor = self.preprocess_sar(query_image_path)
            query_emb = self.encode_sar(sar_tensor)
            res_cross = self.search_gallery(query_emb, target_gallery="MS", k=k)
            for r in res_cross:
                r["provenance"] = "cross-modal (SAR -> MS)"
            res_uni = self.search_gallery(query_emb, target_gallery="SAR", k=k)
            for r in res_uni:
                r["provenance"] = "unimodal (SAR -> SAR)"

        elif modality in ("MS", "S2", "MULTISPECTRAL", "SENTINEL-2"):
            ms_tensor = self.preprocess_ms(query_image_path)
            query_emb = self.encode_ms(ms_tensor)
            res_cross = self.search_gallery(query_emb, target_gallery="SAR", k=k)
            for r in res_cross:
                r["provenance"] = "cross-modal (MS -> SAR)"
            res_uni = self.search_gallery(query_emb, target_gallery="MS", k=k)
            for r in res_uni:
                r["provenance"] = "unimodal (MS -> MS)"
        else:
            raise ValueError(f"Cannot resolve query modality '{query_modality}' for Hybrid retrieval.")

        combined = res_cross + res_uni
        combined.sort(key=lambda x: x["similarity_score"], reverse=True)

        seen_pids = set()
        merged: List[Dict[str, Any]] = []
        for item in combined:
            pid = item["patch_id"]
            if pid not in seen_pids:
                seen_pids.add(pid)
                merged.append(item)
            if len(merged) == k:
                break

        for rank, item in enumerate(merged, start=1):
            item["rank"] = rank
            item["tone"] = ((rank - 1) % 5) + 1
            if subcategory:
                item["subcategory"] = subcategory

        return merged

    # --------------------------------------------------------
    # MAIN RETRIEVAL INTERFACE
    # --------------------------------------------------------

    def retrieve(
        self,
        query_image_path: Union[str, Path, torch.Tensor, np.ndarray],
        query_modality: str,
        subcategory: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Main retrieval method: retrieve(query_image_path, query_modality, subcategory=None, k=10)."""
        configured_modality = None
        if subcategory:
            sub_info = resolve_subcategory_info(subcategory, self.config)
            if sub_info:
                configured_modality = sub_info.get("modality", "")

        is_hybrid = (
            query_modality.strip().upper() in ("HYBRID", "DUAL")
            or (configured_modality and configured_modality.strip().upper() == "HYBRID")
        )

        if is_hybrid:
            effective_mod = "SAR" if query_modality.strip().upper() in ("SAR", "S1") else "MS"
            if query_modality.strip().upper() in ("HYBRID", "DUAL", "AUTO"):
                effective_mod = "auto"
            return self.retrieve_hybrid(
                query_image_path=query_image_path,
                query_modality=effective_mod,
                subcategory=subcategory,
                k=k,
            )

        modality = query_modality.strip().upper()

        if modality in ("SAR", "S1", "SENTINEL-1"):
            sar_tensor = self.preprocess_sar(query_image_path)
            query_emb = self.encode_sar(sar_tensor)
            results = self.search_gallery(query_emb, target_gallery="MS", k=k)
            for r in results:
                r["provenance"] = "cross-modal (SAR -> MS)"

        elif modality in ("MS", "S2", "MULTISPECTRAL", "SENTINEL-2"):
            ms_tensor = self.preprocess_ms(query_image_path)
            query_emb = self.encode_ms(ms_tensor)
            results = self.search_gallery(query_emb, target_gallery="SAR", k=k)
            for r in results:
                r["provenance"] = "cross-modal (MS -> SAR)"

        else:
            raise ValueError(
                f"Unsupported query modality: '{query_modality}'. Expected 'SAR', 'MS'/'Multispectral', or 'Hybrid'."
            )

        if subcategory:
            for item in results:
                item["subcategory"] = subcategory

        return results


# ============================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================

_RETRIEVER_INSTANCE: Optional[CrossModalRetriever] = None


def get_retriever() -> CrossModalRetriever:
    """Singleton getter for CrossModalRetriever."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = CrossModalRetriever()
    return _RETRIEVER_INSTANCE


def retrieve(
    query_image_path: Union[str, Path, torch.Tensor, np.ndarray],
    query_modality: str,
    subcategory: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Module-level retrieval function for direct use."""
    retriever_instance = get_retriever()
    return retriever_instance.retrieve(
        query_image_path=query_image_path,
        query_modality=query_modality,
        subcategory=subcategory,
        k=k,
    )
