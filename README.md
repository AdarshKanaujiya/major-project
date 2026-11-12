# SENTINEL-RS: Cross-Modal Remote-Sensing Image Retrieval System

**Deep Cross-Modal Satellite Image Retrieval Using Multi-Sensor Sentinel-1 (SAR) and Sentinel-2 (Multispectral) Earth Observation Data**

[![Status: Staging Complete](https://img.shields.io/badge/PARAM%20SHAVAK%20Staging-Complete%20(104%20Files%20Verified)-brightgreen.svg)](#6-param-shavak-hpc-staging--readiness)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#7-installation--environment-setup)
[![PyTorch: 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](#7-installation--environment-setup)
[![FAISS: IndexFlatIP](https://img.shields.io/badge/FAISS-IndexFlatIP%20(Cosine)-purple.svg)](#2-architecture--scientific-pipeline)
[![Checkpoint: 598.95MB](https://img.shields.io/badge/Checkpoint-best__model--v1.ckpt%20(Verified)-success.svg)](#5-model-checkpoint--artifact-provenance)

---

## 1. Project Overview & Motivation

Earth observation satellites collect diverse modalities of planetary imagery. Synthetic Aperture Radar (**Sentinel-1 SAR**) operates in the microwave C-band and penetrates cloud cover, smoke, and nighttime darkness, providing structural roughness and dielectric moisture measurements. Optical/Multispectral sensors (**Sentinel-2 MS**) capture solar reflectance across optical and infrared bands, providing rich multispectral and thematic surface classifications.

**SENTINEL-RS** bridges these complementary sensor regimes through joint contrastive representation learning. Given a SAR radar query (e.g., during cloud-obscured floods or severe weather), the system retrieves semantically and physically corresponding optical multispectral scenes from a multi-thousand-image gallery in milliseconds, and vice versa.

### Core Capabilities
- **Bi-Directional Cross-Modal Retrieval**: Query SAR $\to$ Retrieve MS Optical; Query MS Optical $\to$ Retrieve SAR.
- **Same-Modal Semantic Search**: Query SAR $\to$ Retrieve SAR; Query MS $\to$ Retrieve MS.
- **Fail-Closed Latency Benchmarking**: Sub-millisecond similarity search across 4,500 unseen test scenes.
- **Interactive Atlas Retrieval Portal**: Production Flask web application supporting image upload, modality routing across 3 domains and 6 disaster/monitoring subcategories, and real-time GeoTIFF thumbnail streaming.
- **Strict HPC Segregation**: Complete decoupling between local PC exploratory experiments (`results/local_pc/`) and PARAM SHAVAK Linux cluster execution (`param_shavak/`).

---

## 2. Architecture & Scientific Pipeline

```
                                  +-----------------------------------------------+
                                  |         BigEarthNet Satellite Imagery         |
                                  |   Sentinel-1 (SAR)   |   Sentinel-2 (MS Optical)  |
                                  +-----------------------------------------------+
                                                          |
                                                          v
                                  +-----------------------------------------------+
                                  |           Week 2 Canonical Preprocessing       |
                                  |  - SAR: 2 channels (VV, VH) clipped [-25, 0] |
                                  |    Resize to (2, 224, 224), Train Z-Score Norm|
                                  |  - MS: 3 channels (B04, B03, B02)             |
                                  |    Resize to (3, 224, 224), Train Z-Score Norm|
                                  +-----------------------------------------------+
                                                          |
                                                          v
                                  +-----------------------------------------------+
                                  |           Week 4 Two-Tower Architecture       |
                                  |  - SAR Tower: Modified ResNet-50 (conv1 in=2) |
                                  |  - MS Tower : Standard ResNet-50 (conv1 in=3) |
                                  |  - Projections: Linear(2048->1024) -> ReLU()  |
                                  |                 -> Linear(1024->512)          |
                                  |  - Training: Symmetric InfoNCE Loss (T=0.07)  |
                                  +-----------------------------------------------+
                                                          |
                                                          v
                                  +-----------------------------------------------+
                                  |       Week 5 Feature Extraction & Normalization|
                                  |  - 512-D L2-normalized feature embeddings     |
                                  |  - Dual FAISS IndexFlatIP Index Construction  |
                                  +-----------------------------------------------+
                                                          |
                                                          v
                                  +-----------------------------------------------+
                                  |           Multimodal Retrieval & Atlas        |
                                  |  - Cross-Modal: SAR <-> MS (Exact-pair R@K)   |
                                  |  - Same-Modal: SAR<->SAR, MS<->MS (HR@K, mAP) |
                                  |  - Latency Benchmarking (Fail-Closed Mode)    |
                                  |  - Atlas Remote Sensing Intelligence Portal   |
                                  +-----------------------------------------------+
```

### Technical Specifications & Tensor Contracts

| Component | Specification | Technical Detail |
|---|---|---|
| **SAR Input** | `(B, 2, 224, 224)`, Float32 | Dual-pol backscatter (VV, VH) clipped to `[-25.0, 0.0]` dB, normalized using training-only Z-score statistics (`s1_stats.json`). |
| **MS Input** | `(B, 3, 224, 224)`, Float32 | Sentinel-2 RGB bands in strict order: B04 (Red), B03 (Green), B02 (Blue), normalized using training-only Z-score statistics (`s2_stats.json`). |
| **SAR Tower** | Modified ResNet-50 | First convolution replaced (`in_channels=2`, `out_channels=64`, `kernel_size=7x7`, `stride=2`, `bias=False`). Final FC layer replaced with `Identity()`. |
| **MS Tower** | Standard ResNet-50 | Standard 3-channel ResNet-50 backbone. Final FC layer replaced with `Identity()`. |
| **Metric Heads** | 512-D Non-Linear MLP | `Linear(2048, 1024)` $\to$ `ReLU()` $\to$ `Linear(1024, 512)`. Default `use_batchnorm=False` preserves exact historical checkpoint weight alignment. |
| **Embeddings** | 512-D Unit Vectors | Explicit L2 normalization ($\|z\|_2 = 1.0$) ensures inner product equals cosine similarity. |
| **Loss Function**| Symmetric InfoNCE | $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{\text{SAR}\to\text{MS}} + \mathcal{L}_{\text{MS}\to\text{SAR}})$ with temperature $\tau = 0.07$, AdamW optimizer, and Cosine Annealing learning rate schedule. |
| **Index Engine** | FAISS `IndexFlatIP(512)` | Dual exact inner-product galleries (SAR gallery and MS gallery) indexing 4,500 test scenes each. |

---

## 3. Audited Performance Benchmarks

All performance metrics below represent **authoritative ground-truth evaluation results** reconciled against historical CSV files in `results/local_pc/week_05/evaluation/` across 4,500 unseen BigEarthNet test scenes.

### Cross-Modal Exact-Pair Retrieval
Measures whether the physically corresponding scene in the opposite modality is retrieved within the top $K$ nearest neighbors:

| Mode | Recall@1 | Recall@5 | Recall@10 | Baseline Untrained R@10 | Absolute Improvement | Avg Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SAR $\to$ MS** | **38.78%** | **64.82%** | **75.04%** | 1.13% | **+73.91%** | 0.012 ms |
| **MS $\to$ SAR** | **37.04%** | **63.73%** | **74.00%** | 1.09% | **+72.91%** | 0.014 ms |

### Same-Modal Multi-Label Semantic Retrieval
Measures retrieval of semantically relevant scenes sharing land-cover classes under the official 19-class BigEarthNet taxonomy (excluding the query itself from rank 0):

| Mode | HitRate@1 | HitRate@5 | HitRate@10 | $\text{mAP\_ret}@10$ | Baseline Untrained HR@10 | Avg Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SAR $\to$ SAR** | **53.40%** | **85.98%** | **92.49%** | **61.24%** | 12.33% | 0.012 ms |
| **MS $\to$ MS** | **57.84%** | **87.04%** | **93.47%** | **64.09%** | 94.67% | 0.011 ms |

> [!NOTE]
> **Metric Naming & Scientific Reconciliation**:
> - **HitRate@K**: Measures the percentage of query scenes for which at least one semantically matching gallery scene is retrieved in the top $K$ results.
> - **$\text{mAP\_ret}@10$**: Represents Mean Average Precision truncated at rank 10 and normalized by the number of retrieved relevant items ($\le 10$).
> - **Exact-Pair Recall@K**: Evaluates exact scene correspondence via paired identifiers (`patch_id`), verified using the canonical mapping `test_patch_id_mapping.csv`.

---

## 4. Consolidated Repository Structure

```
SENTINEL-RS/
├── shared/                                 # Centralized project-wide assets
│   ├── config/                             # Unified settings, environment specs, modal configs
│   │   ├── settings.py                     # Central dynamic environment & path configuration
│   │   ├── environment_hpc.yml             # Conda environment specification (Python 3.10, CUDA 12.1)
│   │   ├── requirements_hpc.txt            # PARAM SHAVAK pip requirements
│   │   ├── requirements_local.txt          # Local Windows development requirements
│   │   └── disaster_modality_config.json   # 3-domain, 6-subcategory modality routing config
│   ├── docs/                               # Engineering documentation & comprehensive audit reports
│   │   ├── agent_handoff/                  # Living status board, handoffs, and audit trails
│   │   │   ├── CURRENT_PROJECT_STATUS.md   # Current milestone & operational constraints
│   │   │   ├── AGENT_HANDOFF.md            # Actionable next-agent execution roadmap
│   │   │   └── RESEARCH_AND_AUDIT_LOG.md   # Chronological log of all audited phases
│   │   └── audit_report/                   # Authoritative audit reports (Weeks 1 to 5)
│   ├── models/                             # Trained neural model checkpoint
│   │   └── best_model-v1.ckpt              # 628,042,368 bytes (SHA-256: 83c11cbf...)
│   └── utilities/                          # Shared I/O, configuration, and audit utilities
│
├── weeks/                                  # Canonical source code organized by research milestone
│   ├── week_01/                            # Milestone 1: Exploration, Taxonomy & Inspection
│   │   ├── data_exploration/               # Metadata parquets, class distribution & cloud filters
│   │   └── inspection/                     # SAR dB statistics, GeoTIFF headers & RGB visualizers
│   ├── week_02/                            # Milestone 2: Preprocessing & Paired DataLoaders
│   │   ├── dataset_preparation/            # Canonical 30k splits (21k train, 4.5k val, 4.5k test)
│   │   ├── preprocessing/                  # S1 VV/VH clipping [-25, 0] dB, S2 RGB normalization
│   │   └── dataloader/                     # PairedBigEarthNetDataset & PyTorch DataLoaders
│   ├── week_03/                            # Milestone 3: Baseline Retrieval & Zero-Shot Evaluation
│   │   ├── embeddings/                     # Untrained ResNet-50 2048-D feature extractors
│   │   ├── retrieval/                      # Baseline cosine similarity retrieval engines
│   │   ├── evaluation/                     # Multi-label mAP calculation engine
│   │   └── visualization/                  # Pretrained 2D t-SNE projections
│   ├── week_04/                            # Milestone 4: Two-Tower Training & Contrastive Learning
│   │   ├── models/                         # TwoTowerNetwork & InfoNCELoss (two_tower_jainam.py)
│   │   ├── training/                       # Lightning trainer (train_jainam.py), AdamW, Cosine Annealing
│   │   ├── evaluation/                     # Overfitting monitor (monitor_overfitting_adarsh.py)
│   │   └── visualization/                  # Training loss curves (plot_training_loss_adarsh.py)
│   └── week_05/                            # Milestone 5: Retrieval, Benchmarking & Atlas Portal
│       ├── embeddings/                     # 512-D L2-normalized embedding extraction scripts
│       ├── retrieval/                      # CrossModalRetriever & FAISS IndexFlatIP index builders
│       ├── evaluation/                     # Cross-modal and same-modal evaluation suites
│       ├── visualization/                  # Aligned t-SNE projections & retrieval grid visualizers
│       ├── benchmarking/                   # Fail-closed latency profiler (timing_meet.py)
│       └── deployment/                     # Atlas Remote Sensing Intelligence Portal (Flask app)
│           ├── templates/                  # HTML UI templates (domain, search, subcategory)
│           └── static/                     # CSS stylesheets (style.css)
│
├── results/                                # Centralized experimental artifacts
│   └── local_pc/                           # Historical outputs from the 30,000-patch local PC experiment
│       ├── week_01/metadata/               # Parquet metadata and coverage reports
│       ├── week_02/normalization_stats/    # s1_stats.json, s2_stats.json (training-only)
│       ├── week_02/splits/                 # train_split.csv (21k), val_split.csv (4.5k), test_split.csv (4.5k)
│       ├── week_03/baseline_embeddings/    # Baseline feature arrays
│       ├── week_04/checkpoints/            # best_model-v1.ckpt (628MB), hparams.yaml
│       ├── week_04/metrics/                # metrics.csv, Week_4_training_record.txt
│       ├── week_05/trained_embeddings/     # trained_test_sar/ms_embeddings.npy (4500, 512)
│       │   ├── derived_val_l2/             # [QUARANTINED] Derived validation L2 arrays
│       │   └── test_patch_id_mapping.csv   # Authoritative 4,500-row bi-directional scene mapping
│       ├── week_05/faiss_indices/          # sar_gallery.index, ms_gallery.index (IndexFlatIP)
│       ├── week_05/evaluation/             # trained_cross_modal_results.csv, same_modal CSVs
│       └── week_05/benchmark_proxies/      # Isolated synthetic architectural benchmark outputs
│
├── param_shavak/                           # Staged Cluster Environment (104 Files Verified)
│   ├── STAGING_README.md                   # Dedicated HPC staging & execution guide
│   ├── staging_manifest.json               # 104-file bitwise SHA-256 verification manifest
│   ├── models/                             # Staged checkpoint (best_model-v1.ckpt, 628MB)
│   ├── data/                               # Staged split CSVs, normalization stats, patch mapping
│   ├── shared/                             # Staged shared config & utilities
│   ├── scripts/                            # SLURM execution scripts for Linux cluster
│   └── week_01/ to week_05/                # Staged canonical source modules (prototypes excluded)
│
├── needs_classification/                   # Specialized data copying, search, and deployment scripts
├── PROJECT_FILE_MAPPING.md                 # Complete provenance mapping of all 189 original files
└── README.md                               # This master documentation
```

---

## 5. Model Checkpoint & Artifact Provenance

### Checkpoint File Integrity
The trained PyTorch Lightning model checkpoint `best_model-v1.ckpt` is physically present on disk and bitwise verified:
- **Canonical Locations**:
  - `shared/models/best_model-v1.ckpt`
  - `results/local_pc/week_04/checkpoints/best_model-v1.ckpt`
  - `param_shavak/models/best_model-v1.ckpt`
  - `param_shavak/shared/models/best_model-v1.ckpt`
- **File Size**: Exactly **`628,042,368 bytes`** (598.95 MB)
- **SHA-256 Hash**: `83C11CBFA3865DFF907B28B4EFC28817B40DECB7D5790F4A88B3DC03C0423A7C`
- **State-Dict Architecture**: Two-Tower ResNet-50 with 512-D projection heads. Model loaders cleanly strip the `network.` Lightning attribute prefix and filter out `loss_fn.` keys.

### Dataset Split & Mapping Provenance
- **Splits Directory**: `results/local_pc/week_02/splits/`
  - `train_split.csv`: 21,000 paired patches
  - `val_split.csv`: 4,500 paired patches
  - `test_split.csv`: 4,500 paired patches
  - **Zero Leakage**: All three splits are mutually disjoint ($\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$).
- **Scene Mapping**: `results/local_pc/week_05/trained_embeddings/test_patch_id_mapping.csv`
  - 4,500 rows providing 100% row-for-row alignment between canonical `patch_id`, native SAR `s1_name`, and native MS `s2v1_name`.

---

## 6. PARAM SHAVAK HPC Staging & Readiness

The canonical Week 1–5 codebase, configurations, model weights, and split metadata have been staged into `param_shavak/` under controlled verification:
- **Staging Verdict**: **`STAGING COMPLETE — ALL HASHES VERIFIED`**
- **Manifest**: 104 files verified bitwise against source files with zero conflicts and zero mismatches.
- **Excluded Items**: Obsolete prototypes containing legacy `src.*` imports (`train_adarsh.py`, `train_two_tower_fateh.py`, etc.), synthetic mock demo (`app_fateh.py`), quarantined validation arrays, local evaluation CSVs, and Python bytecode were strictly omitted.
- **Known Integration Notes for HPC**:
  1. *Test Extractor Import*: `extract_test_embeddings_jainam.py` and `meet.py` reference `weeks.week_02.dataset.paired_dataset` (canonical is `weeks.week_02.dataloader.paired_bigearthnet_dataset_meet`).
  2. *Atlas Flask Asset Path*: `app_meet.py` points to `PROJECT_ROOT / templates` instead of `deployment/templates`.

For complete staging details, see:
- [`param_shavak/STAGING_README.md`](file:///c:/Major%20Project/major-project/param_shavak/STAGING_README.md)
- [`shared/docs/audit_report/param_shavak_staging_manifest.md`](file:///c:/Major%20Project/major-project/shared/docs/audit_report/param_shavak_staging_manifest.md)
- [`shared/docs/audit_report/param_shavak_final_integrity_verification.md`](file:///c:/Major%20Project/major-project/shared/docs/audit_report/param_shavak_final_integrity_verification.md)

---

## 7. Installation & Environment Setup

### Option A: Conda Environment (Recommended for Linux / PARAM SHAVAK)
```bash
# Create conda environment from staged specification
conda env create -f shared/config/environment_hpc.yml

# Activate environment
conda activate sentinel_rs
```

### Option B: Local Windows Development
```bash
# Create virtual environment with Python 3.10 - 3.12
python -m venv venv
venv\Scripts\activate

# Install local dependencies
pip install -r shared/config/requirements_local.txt
```

---

## 8. Execution Guide

### A. Model Training (PyTorch Lightning)
To train the Two-Tower Network with InfoNCE contrastive loss:
```bash
python weeks/week_04/training/train_jainam.py \
    --batch-size 32 \
    --max-epochs 30 \
    --learning-rate 0.0001 \
    --optimizer adamw \
    --temperature 0.07 \
    --num-workers 4
```

### B. Fail-Closed Latency Benchmarking
To profile retrieval pipeline components with strict process exit codes:
```bash
# Default Real Mode: Fails closed (exit code 2) if PyTorch, checkpoint, or data are absent
python weeks/week_05/benchmarking/timing_meet.py --mode sar

# Explicit Synthetic Architectural Proxy Mode:
python weeks/week_05/benchmarking/timing_meet.py --allow-synthetic-proxy --mode sar
```
*Process Exit Status Behavior*:
- `Exit 0`: Benchmark completed successfully.
- `Exit 2`: Execution blocked due to missing environment dependencies, checkpoint, or imagery (`[BLOCKED BY ENVIRONMENT]`).
- `Exit 1`: Unexpected execution failure.

### C. Offline Retrieval Evaluation
To evaluate cross-modal and same-modal retrieval on pre-computed test embeddings:
```bash
# Cross-modal exact-pair retrieval evaluation (SAR -> MS and MS -> SAR)
python weeks/week_05/evaluation/evaluate_trained_cross_modal_meet.py

# Same-modal multi-label semantic evaluation (SAR -> SAR and MS -> MS)
python weeks/week_05/evaluation/evaluate_trained_same_modal_meet.py
```

### D. Interactive Atlas Web Portal
To start the production Atlas retrieval portal locally:
```bash
python weeks/week_05/deployment/app_meet.py
```
Open your browser to `http://localhost:5000` to interactively upload Sentinel-1 or Sentinel-2 imagery, select disaster and environmental subcategories, and inspect retrieved satellite scenes in real time.

### E. PARAM SHAVAK SLURM Execution
On the HPC cluster:
```bash
# Step 1: Compute full dataset population statistics
sbatch param_shavak/scripts/01_compute_full_dataset_stats.sh

# Step 2: Train Two-Tower network on GPU
sbatch param_shavak/scripts/02_train_two_tower_hpc.sh

# Step 3: Extract embeddings, build FAISS galleries & evaluate
sbatch param_shavak/scripts/03_extract_embeddings_and_eval.sh
```

---

## 9. Team Member Contributions

The unified codebase consolidates research contributions across four team members:

| Member | Primary Research Role | Canonical Modules & Core Deliverables |
|:---|:---|:---|
| **Jainam (Person 1)** | **Project Lead & Pipeline Architect** | 30k stratified split generation (`create_final_splits_jainam.py`), pair alignment verification (`verify_pairs_jainam.py`), canonical Two-Tower model architecture (`two_tower_jainam.py`), PyTorch Lightning trainer (`train_jainam.py`), FAISS gallery index builder (`build_faiss_indices_jainam.py`), and final Week 5 performance reports. |
| **Adarsh (Person 2)** | **Preprocessing, Evaluation & Visualization** | BigEarthNet parquet metadata exploration & cloud/snow filtering (`filter_clean_data_adarsh.py`), Sentinel-2 21k training normalization (`s2_stats.json`), Sentinel-2 preprocessing pipeline, multi-label mAP calculation engine (`compute_map_adarsh.py`), and t-SNE embedding visualizations. |
| **Meet (Person 3)** | **SAR Pipeline, Web Portal & Latency Profiling** | Sentinel-1 SAR dB clipping (`[-25, 0]` dB) & population statistics (`s1_stats.json`), canonical SAR preprocessing (`sar_preprocessing_meet.py`), paired dataset loader (`paired_bigearthnet_dataset_meet.py`), interactive Atlas Flask portal (`app_meet.py`), production retrieval engine (`retriever_meet.py`), and fail-closed latency benchmarker (`timing_meet.py`). |
| **Fateh (Person 4)** | **Architecture Foundations & Baseline Setup** | Early Two-Tower contrastive architecture design, baseline feature extraction (`baseline_extractor_fateh.py`), baseline mAP evaluation (`evaluate_baseline_map_fateh.py`), and research logs. |

---

## 10. Key Documentation & Audit Index

| Document Path | Description |
|---|---|
| [`AI_START_HERE.md`](file:///c:/Major%20Project/major-project/AI_START_HERE.md) | Universal AI reading directive & project orientation |
| [`PROJECT_FILE_MAPPING.md`](file:///c:/Major%20Project/major-project/PROJECT_FILE_MAPPING.md) | File-by-file provenance table across all 189 original files |
| [`CURRENT_PROJECT_STATUS.md`](file:///c:/Major%20Project/major-project/shared/docs/agent_handoff/CURRENT_PROJECT_STATUS.md) | Living status board, active milestones & constraints |
| [`AGENT_HANDOFF.md`](file:///c:/Major%20Project/major-project/shared/docs/agent_handoff/AGENT_HANDOFF.md) | Actionable handoff instructions for successive agents |
| [`RESEARCH_AND_AUDIT_LOG.md`](file:///c:/Major%20Project/major-project/shared/docs/agent_handoff/RESEARCH_AND_AUDIT_LOG.md) | Chronological audit trail across Weeks 1 to 5 |
| [`param_shavak/STAGING_README.md`](file:///c:/Major%20Project/major-project/param_shavak/STAGING_README.md) | PARAM SHAVAK HPC deployment and staging guide |
| [`param_shavak_staging_manifest.md`](file:///c:/Major%20Project/major-project/shared/docs/audit_report/param_shavak_staging_manifest.md) | Complete 104-file staging manifest with SHA-256 hashes |
| [`param_shavak_final_integrity_verification.md`](file:///c:/Major%20Project/major-project/shared/docs/audit_report/param_shavak_final_integrity_verification.md) | Final read-only post-staging verification report |
| [`week1_to_week5_final_transfer_readiness_audit.md`](file:///c:/Major%20Project/major-project/shared/docs/audit_report/week1_to_week5_final_transfer_readiness_audit.md) | Comprehensive 15-section integrated transfer audit |
| [`week5_phaseB2_1_exit_status_correction.md`](file:///c:/Major%20Project/major-project/shared/docs/audit_report/week5_phaseB2_1_exit_status_correction.md) | Benchmark exit-code compliance fix verification report |

---
*Maintained under the Google DeepMind agentic engineering workflow for SENTINEL-RS.*
