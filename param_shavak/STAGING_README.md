# SENTINEL-RS — PARAM SHAVAK HPC Staging Guide

**Directory:** `param_shavak/`  
**Consolidation Date:** 2026-11-12  
**Staging Status:** `STAGING COMPLETE — ALL HASHES VERIFIED`  
**Execution Environment Target:** PARAM SHAVAK (PS-BOSS-OS Linux x86_64, NVIDIA RTX 5000 Ada GPU, SLURM Workload Manager)

---

## 1. Project Overview

**SENTINEL-RS** is a cross-modal remote-sensing image retrieval system enabling bi-directional search between Sentinel-1 Synthetic Aperture Radar (SAR) backscatter and Sentinel-2 Multispectral (MS) optical reflectance images.

This `param_shavak/` directory contains a curated, bitwise-verified staging copy of the canonical Week 1–5 codebase, configurations, model weights, and split metadata prepared for future high-performance computing deployment.

> [!IMPORTANT]
> **NO REMOTE HPC EXECUTION HAS BEEN RUN.**
> All files in this directory have been staged and verified locally. No connections to the PARAM SHAVAK cluster, SSH sessions, GPU runs, or SLURM jobs have been initiated.

---

## 2. Staged Directory Structure

```
param_shavak/
├── STAGING_README.md             # This comprehensive deployment guide
├── staging_manifest.json         # Complete JSON manifest of all 104 staged artifacts
├── scripts/                      # Pre-staged HPC SLURM job scripts
│   ├── 01_compute_full_dataset_stats.sh
│   ├── 02_train_two_tower_hpc.sh
│   ├── 03_extract_embeddings_and_eval.sh
│   ├── generate_full_dataset_splits_and_stats.py
│   └── run_all_hpc.sh
├── models/                       # Staged PyTorch Lightning checkpoint
│   └── best_model-v1.ckpt        # 628,042,368 bytes (SHA-256: 83c11cbf...)
├── data/                         # Core dataset splits, statistics, and mappings
│   ├── splits/                   # Canonical 21k/4.5k/4.5k train/val/test splits
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   └── test_split.csv
│   ├── normalization_stats/      # Training z-score normalization parameters
│   │   ├── s1_stats.json
│   │   ├── s1_stats_regeneration_summary.json
│   │   └── s2_stats.json
│   └── test_patch_id_mapping.csv # 4,500-row bi-directional scene mapping
├── shared/                       # Shared modules and configurations
│   ├── config/                   # HPC environment and configuration files
│   │   ├── environment_hpc.yml   # Conda environment specification
│   │   ├── requirements_hpc.txt  # Cluster pip dependencies
│   │   ├── disaster_modality_config.json
│   │   └── settings.py           # Unified dynamic path configuration
│   ├── models/                   # Mirrored checkpoint path for config compatibility
│   │   └── best_model-v1.ckpt    # 628,042,368 bytes (SHA-256: 83c11cbf...)
│   └── utilities/                # Shared utilities
│       ├── config_jainam.py
│       ├── io_jainam.py
│       ├── static_code_auditor.py
│       └── visualization_meet.py
├── week_01/                      # Week 1: Data Exploration & Taxonomy
│   ├── README.md                 # Week 1 cluster reference guide
│   ├── data_exploration/         # Metadata exploration and label analysis
│   └── inspection/               # GeoTIFF inspection utilities
├── week_02/                      # Week 2: Preprocessing, Splits & Dataloaders
│   ├── README.md                 # Week 2 cluster reference guide
│   ├── dataloader/               # PyTorch DataLoader and Dataset implementations
│   ├── dataset_preparation/      # Split generation and pairing verification
│   └── preprocessing/            # Canonical S1 (VV/VH) & S2 (RGB) preprocessors
├── week_03/                      # Week 3: Baseline Zero-Shot Embeddings & Initial Retrieval
│   ├── README.md                 # Week 3 documentation
│   ├── embeddings/               # Baseline ResNet-50 feature extractors
│   ├── evaluation/               # Baseline mAP evaluators
│   ├── retrieval/                # Baseline cosine search engines
│   └── visualization/            # Baseline t-SNE visualizers
├── week_04/                      # Week 4: Two-Tower Contrastive Training (InfoNCE)
│   ├── README.md                 # Week 4 documentation
│   ├── models/                   # TwoTowerNetwork & InfoNCELoss architectures
│   ├── training/                 # Canonical PyTorch Lightning trainer (train_jainam.py)
│   ├── evaluation/               # Overfitting auditor (monitor_overfitting_adarsh.py)
│   └── visualization/            # Loss curve plotter (plot_training_loss_adarsh.py)
└── week_05/                      # Week 5: Trained Retrieval, Benchmarking & Atlas Portal
    ├── README.md                 # Week 5 documentation
    ├── benchmarking/             # Fail-closed latency benchmarker (timing_meet.py)
    ├── deployment/               # Atlas portal (app_meet.py, templates/, static/)
    ├── embeddings/               # Trained embedding extraction scripts
    ├── evaluation/               # Cross-modal & same-modal evaluation suites
    ├── retrieval/                # CrossModalRetriever & FAISS index builders
    └── visualization/            # t-SNE, grid, and case visualizers
```

---

## 3. How to Identify Canonical Source Files

The staged codebase preserves multi-contributor provenance while designating primary canonical execution paths:

| Function | Canonical Script / Module | Alternative / Specialized Variant |
|---|---|---|
| **Two-Tower Architecture** | `param_shavak/week_04/models/two_tower_jainam.py` | `two_tower_meet.py`, `two_tower_adarsh.py` |
| **Model Training** | `param_shavak/week_04/training/train_jainam.py` | — (Prototypes excluded) |
| **SAR Preprocessing** | `param_shavak/week_02/preprocessing/sar_preprocessing_meet.py` | `sar_preprocessing_adarsh.py`, `jainam.py` |
| **MS Preprocessing** | `param_shavak/week_02/preprocessing/preprocess_s2_meet.py` | `preprocess_s2_adarsh.py`, `jainam.py` |
| **Paired Dataset** | `param_shavak/week_02/dataloader/paired_bigearthnet_dataset_meet.py` | `dataset_adarsh.py`, `dataset_meet.py` |
| **Multimodal Retrieval**| `param_shavak/week_05/retrieval/retriever_meet.py` | — |
| **FAISS Indexing** | `param_shavak/week_05/retrieval/build_faiss_indices_jainam.py` | `build_faiss_indices_meet.py` |
| **Cross-Modal Evaluation**| `param_shavak/week_05/evaluation/evaluate_trained_cross_modal_jainam.py` | `evaluate_trained_cross_modal_meet.py` |
| **Same-Modal Evaluation** | `param_shavak/week_05/evaluation/evaluate_trained_same_modal_jainam.py` | `evaluate_trained_same_modal_meet.py` |
| **Latency Benchmark** | `param_shavak/week_05/benchmarking/timing_meet.py` | — |
| **Atlas Web Portal** | `param_shavak/week_05/deployment/app_meet.py` | — (Prototype `app_fateh.py` excluded) |

---

## 4. Dataset and Checkpoint Locations

1. **Model Checkpoint**:
   - Location: `param_shavak/models/best_model-v1.ckpt` (and mirrored in `param_shavak/shared/models/best_model-v1.ckpt`)
   - Size: `628,042,368 bytes` (598.95 MB)
   - SHA-256: `83C11CBFA3865DFF907B28B4EFC28817B40DECB7D5790F4A88B3DC03C0423A7C`
   - Verified: Bitwise identical to the canonical local checkpoint.

2. **Split Metadata**:
   - `param_shavak/data/splits/train_split.csv` (21,000 paired samples)
   - `param_shavak/data/splits/val_split.csv` (4,500 paired samples)
   - `param_shavak/data/splits/test_split.csv` (4,500 paired samples)
   - Disjointness: Zero overlap across train, val, and test splits.

3. **Normalization Statistics**:
   - `param_shavak/data/normalization_stats/s1_stats.json`: VV ($\mu=-12.54, \sigma=4.78$), VH ($\mu=-18.65, \sigma=4.36$) clipped to `[-25, 0]` dB.
   - `param_shavak/data/normalization_stats/s2_stats.json`: B04/B03/B02 channel means and standard deviations computed from the 21,000-patch training split.

4. **Bi-Directional Scene Mapping**:
   - `param_shavak/data/test_patch_id_mapping.csv` (4,500 rows, 100% verified row-for-row alignment with `test_split.csv`).

5. **BigEarthNet Satellite Imagery (HPC Target)**:
   - Raw satellite imagery is NOT staged in this directory.
   - On PARAM SHAVAK, storage paths must point to cluster storage:
     - SAR Imagery: `/home/data/BigEarthNet-S1-Required`
     - MS Imagery: `/home/data/BigEarthNet-S2`

---

## 5. Environment Setup Specifications

Two canonical environment specifications are staged in `param_shavak/shared/config/`:

1. **`environment_hpc.yml` (Recommended Conda Spec)**:
   - Target Python: `3.10`
   - Target CUDA: `12.1` (`cudatoolkit=12.1`, `pytorch-cuda=12.1`)
   - PyTorch: `pytorch>=2.2.0`, `torchvision>=0.17.0`, `lightning>=2.2.0`
   - Vector Indexing: `faiss-gpu>=1.7.4`
   - Geospatial: `rasterio>=1.3.8`, `gdal>=3.7.0`
   - Numerical: `numpy>=1.24.0,<2.0.0` (prevents NumPy 2.x C-extension ABI breaks)
   - Activation Command: `conda env create -f shared/config/environment_hpc.yml && conda activate sentinel_rs`

2. **`requirements_hpc.txt` (Pip Fallback)**:
   - Lists Linux-compatible packages for standard pip environments.

---

## 6. Explicitly Excluded Files

The following files were intentionally excluded from staging to protect HPC execution integrity:

| Excluded File | Category | Rationale for Exclusion |
|---|---|---|
| `weeks/week_04/training/train_adarsh.py` | Obsolete Prototype | Contains broken legacy `src.data.dataloader` imports |
| `weeks/week_04/training/train_two_tower_fateh.py`| Obsolete Prototype | Contains broken legacy `src.dataset.dataloader` imports |
| `weeks/week_02/dataloader/test_paired_batch_jainam.py` | Obsolete Test | Contains broken legacy `src.data.dataloader` imports |
| `weeks/week_02/dataloader/verify_dataloader_jainam.py` | Obsolete Test | Contains broken legacy `src.data.dataloader` imports |
| `weeks/week_02/preprocessing/test_sar_preprocessing_jainam.py` | Obsolete Test | Contains broken legacy `src.preprocessing` imports |
| `weeks/week_05/deployment/app_fateh.py` | Mock Prototype | Generates random uniform numbers; not real ML inference |
| `results/local_pc/week_05/trained_embeddings/derived_val_l2/` | Quarantined Artifacts | Derived validation arrays quarantined in Phase B.2 |
| `results/local_pc/week_05/evaluation/*.csv` | Historical Results | Historical local PC outputs; must not pollute HPC results |
| Original contributor repositories (`cross-modal-*`, etc.) | Legacy Repos | Superseded by canonical `weeks/` modules |

---

## 7. Known Unresolved Integration Issues (For Later HPC Phase)

In accordance with strict staging rules, source code was not modified during staging. Two known integration issues remain in the staged code and must be addressed during the cluster integration phase:

### Issue 1: Test Embedding Extractor Import Binding
- **Affected Files:**
  - `param_shavak/week_05/embeddings/extract_test_embeddings_jainam.py` (line 48)
  - `param_shavak/week_05/embeddings/extract_test_embeddings_meet.py` (line 49)
- **Current Code:**
  ```python
  from weeks.week_02.dataset.paired_dataset import PairedBigEarthNetDataset
  ```
- **Known Issue:** `weeks.week_02.dataset` does not exist on disk.
- **HPC Resolution:** Update import to target `weeks.week_02.dataloader.paired_bigearthnet_dataset_meet` or use `generate_trained_embeddings_adarsh.py`.

### Issue 2: Atlas Portal Flask Template & Static Relative Paths
- **Affected File:** `param_shavak/week_05/deployment/app_meet.py` (lines 89–93)
- **Current Code:**
  ```python
  app = Flask(
      __name__,
      template_folder=str(PROJECT_ROOT / "templates"),
      static_folder=str(PROJECT_ROOT / "static"),
  )
  ```
- **Known Issue:** Looks for `templates/` at repository root rather than inside `deployment/`.
- **HPC Resolution:** Bind `template_folder` and `static_folder` to `Path(__file__).resolve().parent / "templates"` and `static`.

---

## 8. Verification Summary

- **Total Staged Items:** 104
- **Retained Identical (Pre-Staged W1/W2):** 22
- **Newly Staged & Verified:** 82
- **SHA-256 Mismatches:** 0
- **File Conflicts:** 0
- **Final Decision:** **`STAGING COMPLETE — ALL HASHES VERIFIED`**

