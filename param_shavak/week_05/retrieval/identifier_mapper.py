"""
weeks/week_05/retrieval/identifier_mapper.py

Non-destructive identifier mapper bridging canonical paired-scene `patch_id`
with native sensor-specific folder and file names:
    - patch_id  : Canonical paired-scene identifier (e.g. S2A_MSIL2A_..._39_66)
    - s1_name   : Native Sentinel-1 radar scene folder (e.g. S1B_IW_GRDH_..._39_66)
    - s2v1_name : Native Sentinel-2 optical identifier (e.g. S2A_MSIL2A_..._39_66)

Preserves exact embedding matrix row alignment across the 4,500 test set pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

logger = logging.getLogger("IdentifierMapper")

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "results"
    / "local_pc"
    / "week_05"
    / "trained_embeddings"
    / "test_patch_id_mapping.csv"
)


class PatchIdentifierMapper:
    """
    Bi-directional mapper between canonical paired `patch_id` and native
    sensor names (`s1_name`, `s2v1_name`), maintaining exact matrix row indices.
    """

    def __init__(self, mapping_path: Optional[Union[str, Path]] = None) -> None:
        self.mapping_path = Path(mapping_path) if mapping_path else DEFAULT_MAPPING_PATH
        self._df: Optional[pd.DataFrame] = None
        self._by_patch_id: Dict[str, Dict[str, str]] = {}
        self._by_s1_name: Dict[str, str] = {}
        self._by_s2v1_name: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Identifier mapping CSV not found at: {self.mapping_path}\n"
                f"Please ensure test_patch_id_mapping.csv has been generated."
            )

        self._df = pd.read_csv(self.mapping_path)
        required_cols = {"row_index", "patch_id", "s1_name", "s2v1_name"}
        missing = required_cols - set(self._df.columns)
        if missing:
            raise ValueError(f"Mapping CSV missing required columns: {missing}")

        for _, row in self._df.iterrows():
            pid = str(row["patch_id"]).strip()
            s1 = str(row["s1_name"]).strip()
            s2v1 = str(row["s2v1_name"]).strip() if pd.notna(row.get("s2v1_name")) else ""
            idx = int(row["row_index"])

            info = {
                "row_index": idx,
                "patch_id": pid,
                "s1_name": s1,
                "s2v1_name": s2v1,
            }
            self._by_patch_id[pid] = info
            self._by_s1_name[s1] = pid
            if s2v1:
                self._by_s2v1_name[s2v1] = pid

        logger.info(f"Loaded {len(self._by_patch_id):,} identifier mappings from {self.mapping_path}")

    @property
    def total_count(self) -> int:
        return len(self._by_patch_id)

    def to_s1_name(self, patch_id: str) -> Optional[str]:
        """Convert canonical patch_id to native Sentinel-1 directory name."""
        info = self._by_patch_id.get(patch_id.strip())
        return info["s1_name"] if info else None

    def to_s2v1_name(self, patch_id: str) -> Optional[str]:
        """Convert canonical patch_id to native Sentinel-2 directory name."""
        info = self._by_patch_id.get(patch_id.strip())
        return info["s2v1_name"] if info else None

    def to_patch_id(self, identifier: str) -> Optional[str]:
        """
        Resolve any identifier (patch_id, s1_name, or s2v1_name) to canonical patch_id.
        """
        clean_id = identifier.strip()
        if clean_id in self._by_patch_id:
            return clean_id
        if clean_id in self._by_s1_name:
            return self._by_s1_name[clean_id]
        if clean_id in self._by_s2v1_name:
            return self._by_s2v1_name[clean_id]
        return None

    def get_row_index(self, patch_id: str) -> Optional[int]:
        """Return embedding row index for a given patch_id."""
        info = self._by_patch_id.get(patch_id.strip())
        return info["row_index"] if info else None


# Module-level singleton
_MAPPER_INSTANCE: Optional[PatchIdentifierMapper] = None


def get_identifier_mapper() -> PatchIdentifierMapper:
    """Singleton getter for PatchIdentifierMapper."""
    global _MAPPER_INSTANCE
    if _MAPPER_INSTANCE is None:
        _MAPPER_INSTANCE = PatchIdentifierMapper()
    return _MAPPER_INSTANCE

