# PROJECT_KNOWLEDGE_BASE.md — Technical Reference & Architecture Manual

> **DOCUMENT STATUS**: Comprehensive Reference Manual  
> **PROJECT**: SENTINEL-RS (Cross-Modal Satellite Image Retrieval System)  
> **LAST UPDATED**: November 2026

---

## Table of Contents
1. [Project Overview & Scientific Motivation](#1-project-overview--scientific-motivation)
2. [Sensor Modalities: Sentinel-1 SAR vs. Sentinel-2 Multispectral](#2-sensor-modalities-sentinel-1-sar-vs-sentinel-2-multispectral)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Mathematical Formulation & Loss Functions](#4-mathematical-formulation--loss-functions)
5. [Five-Week Research Development Roadmap](#5-five-week-research-development-roadmap)
6. [Team Member Provenance & Individual Contributions](#6-team-member-provenance--individual-contributions)
7. [Experimental History: The Local 30k Dataset Benchmark](#7-experimental-history-the-local-30k-dataset-benchmark)
8. [Technical Component Inventory](#8-technical-component-inventory)
9. [Operational Limitations & Implementation Constraints](#9-operational-limitations--implementation-constraints)

---

## 1. Project Overview & Scientific Motivation

### 1.1 Problem Statement
Modern Earth Observation (EO) satellite constellations continuously collect massive petabyte-scale streams of imagery across diverse sensor modalities. However, retrieving relevant satellite imagery across modalities remains a significant open problem in remote sensing:
- Optical/Multispectral sensors (such as Sentinel-2) deliver rich spectral and land-cover information, but are severely obstructed by cloud cover, atmospheric haze, and darkness.
- Synthetic Aperture Radar (SAR) sensors (such as Sentinel-1) operate in the microwave spectrum, penetrating cloud cover, rain, and darkness, but capture structural and dielectric roughness rather than visual color.

Traditional retrieval systems rely on geographic coordinates or text metadata tags (e.g. timestamp, country). These fail when searching for visually or semantically similar terrain across different acquisition dates, seasons, or when an operator possesses only a radar image of a disaster zone and needs to retrieve clear optical reference imagery.

### 1.2 Core Objective
**SENTINEL-RS** establishes a **cross-modal metric learning retrieval pipeline** that bridges the modality gap. The system projects paired Sentinel-1 SAR and Sentinel-2 Multispectral patches into a **common 512-dimensional metric embedding space**. In this shared space:
- Semantically corresponding scenes (same terrain, land-cover, and structural properties) have high cosine similarity (close together).
- Non-matching scenes have low cosine similarity (far apart).
- Retrieval operates bi-directionally across sensors in real time (<2 ms vector search).

---

## 2. Sensor Modalities: Sentinel-1 SAR vs. Sentinel-2 Multispectral

| Parameter | Sentinel-1 (Synthetic Aperture Radar) | Sentinel-2 (Multispectral Instrument) |
| :--- | :--- | :--- |
| **Constellation** | Sentinel-1A & Sentinel-1B (ESA Copernicus) | Sentinel-2A & Sentinel-2B (ESA Copernicus) |
| **Sensor Type** | Active C-band Synthetic Aperture Radar (5.405 GHz) | Passive Multi-Spectral Optical Radiometer |
| **Polarization / Bands** | Dual Polarization: **VV** (Vertical Tx/Rx) & **VH** (Vertical Tx, Horizontal Rx) | 13 Spectral Bands; RGB used: **B04** (Red), **B03** (Green), **B02** (Blue) |
| **Resolution** | 10 m / 20 m spatial resolution | 10 m spatial resolution for RGB bands |
| **Physical Property** | Surface roughness, geometric structure, soil moisture, dielectric constant | Optical surface reflectance across solar spectrum |
| **Atmospheric Sensitivity** | **All-weather & day/night**: penetrates clouds, smoke, rain | **Weather-gated**: blocked by clouds, fog, snow, shadows |
| **Input Tensor Shape** | `(B, 2, 224, 224)` (2 channels: VV, VH) | `(B, 3, 224, 224)` (3 channels: B04, B03, B02) |
| **Normalization** | dB conversion + outlier clipping `[-25, 0]` dB + z-score scaling (`s1_stats.json`) | Bilinear resize + RGB channel standardization (`s2_stats.json`) |

---

## 3. End-to-End System Architecture

The pipeline consists of four distinct operational stages:

```
[ Sentinel-1 SAR (VV, VH) ] ──> [ SAR Preprocessing ] ──> [ SAR Tower (ResNet-50) ] ──> [ 512-D L2 Head ] ──┐
                                                                                                               ├──> [ Shared Metric Space ] ──> [ FAISS IndexFlatIP ] ──> Top-K Results
[ Sentinel-2 MS (RGB)    ] ──> [ MS Preprocessing  ] ──> [ MS Tower (ResNet-50)  ] ──> [ 512-D L2 Head ] ──┘
```

### 3.1 SAR Preprocessing Pipeline
1. **Raw GeoTIFF Ingestion**: Ingests Sentinel-1 Level-1 Ground Range Detected (GRD) GeoTIFFs containing dual-polarization bands (VV and VH).
2. **Decibel Conversion & Outlier Clipping**:
   SAR backscatter values $\sigma^0$ exhibit extreme heavy-tailed distributions due to radar speckle and corner reflection:
   $$\sigma^0_{\text{clipped}} = \text{clip}(\sigma^0_{\text{dB}}, -25.0, 0.0)$$
3. **Z-Score Normalization**:
   Standardized using precomputed channel-wise statistics across 21,000 training patches (`s1_stats.json`):
   $$x_{\text{SAR}} = \frac{\sigma^0_{\text{clipped}} - \mu_{\text{S1}}}{\sigma_{\text{S1}}}$$
4. **Resizing**: Interpolated to `(2, 224, 224)` matching ResNet input dimensions.

### 3.2 Multispectral Preprocessing Pipeline
1. **Band Extraction**: Extracts the top three high-resolution (10m) bands from Sentinel-2 L2A: Band 4 (Red, 665 nm), Band 3 (Green, 560 nm), and Band 2 (Blue, 490 nm).
2. **Spatial Resampling**: Resizes raw $120 \times 120$ BigEarthNet patches to $224 \times 224$ via bilinear interpolation.
3. **RGB Standardization**:
   Standardized using precomputed channel-wise statistics from 21,000 training patches (`s2_stats.json`):
   $$x_{\text{MS}} = \frac{x_{\text{RGB}} - \mu_{\text{S2}}}{\sigma_{\text{S2}}}$$
   *Note*: The standardized values lie approximately in range $[-3.0, +4.0]$, not $[0, 1]$.

### 3.3 Two-Tower Neural Network Architecture
Implemented in `weeks/week_04/models/two_tower_jainam.py`:
- **SAR Tower (Encoder 1)**:
  - Backbone: ResNet-50 pretrained on ImageNet.
  - First Convolution Adaptation: Standard `conv1` (`Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`) is replaced with `Conv2d(2, 64, ...)`. Pretrained weights for the first two channels are retained, or averaged across the original 3 channels, to preserve low-level edge feature detection filters.
  - Output: 2048-dimensional feature representation $f_{\text{SAR}}$.
- **Multispectral Tower (Encoder 2)**:
  - Backbone: Standard ResNet-50 accepting 3 RGB channels.
  - Output: 2048-dimensional feature representation $f_{\text{MS}}$.
- **Dual Projection Metric Heads**:
  - Each tower terminates in an identical multi-layer perceptron (MLP) projection head:
    $$\text{Linear}(2048 \rightarrow 1024) \longrightarrow \text{BatchNorm1d} \longrightarrow \text{ReLU} \longrightarrow \text{Linear}(1024 \rightarrow 512)$$
  - Embeddings are strictly **L2-normalized** to lie on the unit hypersphere $\mathbb{S}^{511}$:
    $$z_{\text{SAR}} = \frac{g(f_{\text{SAR}})}{\|g(f_{\text{SAR}})\|_2}, \quad z_{\text{MS}} = \frac{h(f_{\text{MS}})}{\|h(f_{\text{MS}})\|_2}$$

---

## 4. Mathematical Formulation & Loss Functions

### 4.1 Symmetric InfoNCE Contrastive Loss
To align the two distinct metric towers into a single shared latent manifold, the network is trained using the **Symmetric InfoNCE (Normalized Temperature-scaled Cross Entropy / NT-Xent)** objective.

For a batch of $B$ paired scenes, let $(z_i^{\text{SAR}}, z_i^{\text{MS}})$ represent the normalized 512-D embeddings of the $i$-th pair. The pairwise cosine similarity is given by the inner product $s_{i,j} = \langle z_i^{\text{SAR}}, z_j^{\text{MS}} \rangle$.

1. **SAR-to-MS Directional Loss**:
   $$\mathcal{L}_{\text{SAR}\rightarrow\text{MS}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\langle z_i^{\text{SAR}}, z_i^{\text{MS}} \rangle / \tau)}{\sum_{j=1}^B \exp(\langle z_i^{\text{SAR}}, z_j^{\text{MS}} \rangle / \tau)}$$

2. **MS-to-SAR Directional Loss**:
   $$\mathcal{L}_{\text{MS}\rightarrow\text{SAR}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\langle z_i^{\text{MS}}, z_i^{\text{SAR}} \rangle / \tau)}{\sum_{j=1}^B \exp(\langle z_i^{\text{MS}}, z_j^{\text{SAR}} \rangle / \tau)}$$

3. **Total Symmetric Loss**:
   $$\mathcal{L}_{\text{Total}} = \frac{1}{2} \left( \mathcal{L}_{\text{SAR}\rightarrow\text{MS}} + \mathcal{L}_{\text{MS}\rightarrow\text{SAR}} \right)$$
   Where temperature hyperparameter $\tau = 0.07$.

### 4.2 FAISS Vector Similarity Search
Because all embeddings are L2-normalized ($\|z\|_2 = 1$), the cosine similarity between any query $q$ and gallery item $g$ is identical to their inner product:
$$\text{CosineSimilarity}(q, g) = \frac{\langle q, g \rangle}{\|q\|_2 \|g\|_2} = \langle q, g \rangle$$
The system employs **FAISS `IndexFlatIP`** (Inner Product), guaranteeing exact, exhaustive nearest-neighbor search without quantization error at latencies below 2 ms.

---

## 5. Five-Week Research Development Roadmap

| Milestone | Core Objective | Inputs | Key Scripts (in `weeks/`) | Output Artifacts (in `results/local_pc/`) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Week 1** | Environment setup, BigEarthNet metadata exploration, cloud filtering & SAR band inspection | BigEarthNet parquet archives | `explore_metadata_adarsh.py`, `filter_clean_data_adarsh.py`, `inspect_dataset_meet.py` | `metadata.parquet`, `metadata_for_patches_with_snow_cloud_or_shadow.parquet` | `[VERIFIED COMPLETE]` |
| **Week 2** | Stratified 30k sampling, train/val/test splits, normalization statistics & paired DataLoader | Clean BigEarthNet pairs | `create_final_splits_jainam.py`, `compute_stats_adarsh.py`, `paired_bigearthnet_dataset_meet.py` | `train_split.csv` (21k), `val_split.csv` (4.5k), `test_split.csv` (4.5k), `s1_stats.json`, `s2_stats.json` | `[VERIFIED COMPLETE]` |
| **Week 3** | Baseline ResNet-50 feature extraction, baseline Recall@K, mAP computation & baseline t-SNE | Pretrained ResNet-50, val set | `generate_baseline_embeddings_adarsh.py`, `baseline_retrieval_jainam.py`, `compute_map_adarsh.py` | `baseline_val_ms_embeddings.npy` (4500x2048), `baseline_tsne.png`, baseline evaluation reports | `[VERIFIED COMPLETE]` |
| **Week 4** | Two-Tower architecture design, Symmetric InfoNCE contrastive training & loss monitoring | Paired 21k training set | `two_tower_jainam.py`, `train_jainam.py`, `monitor_overfitting_adarsh.py` | `best_model-v1.ckpt` (598.95 MB), `metrics.csv`, `training_loss.png` | `[VERIFIED COMPLETE]` |
| **Week 5** | Trained 512-D test embeddings, FAISS dual galleries, cross/same-modal evaluation & Atlas portal | Trained checkpoint, 4.5k test set | `extract_test_embeddings_jainam.py`, `build_faiss_indices_meet.py`, `app_meet.py`, `retriever_meet.py` | `trained_test_ms/sar_embeddings.npy`, `ms/sar_gallery.index`, `visual_retrieval_comparison.png`, portal | `[VERIFIED COMPLETE]` |

---

## 6. Team Member Provenance & Individual Contributions

The original research was divided among four team members, whose contributions are cataloged across the repository:

### 6.1 Person 1: Jainam (Project Lead, Data Splits & FAISS Evaluation)
- Repository: `cross-modal-remote-sensing-retrieval` (30 Python files).
- **Core Deliverables**:
  - Setup of the core repository structure, virtual environments, and `requirements.txt`.
  - Implemented the stratified sampling script (`create_final_splits.py`) that sampled 30,000 class-balanced pairs from BigEarthNet and produced `train_split.csv` (21k), `val_split.csv` (4.5k), and `test_split.csv` (4.5k) with zero patch leakage.
  - Implemented the canonical Two-Tower model architecture (`src/models/two_tower.py`) and PyTorch Lightning contrastive training loop (`scripts/train.py`).
  - Implemented batch evaluation scripts (`evaluate_trained_cross_modal.py`, `evaluate_trained_same_modal.py`) and compiled the final Week 5 performance comparison report.

### 6.2 Person 2: Adarsh (Metadata Analysis, S2 Preprocessing, mAP Engine & Visualizations)
- Repository: `deep-cross-final-yr-project` (27 Python files).
- **Core Deliverables**:
  - Analyzed BigEarthNet metadata parquets, identified 69,450 cloud/snow/shadow contaminated patches, and performed class distribution imbalance studies across the 19 land-cover classes.
  - Developed the Sentinel-2 multispectral preprocessing module (`preprocess_s2.py`) extracting B04, B03, B02 and resizing to $224 \times 224$.
  - Calculated channel-wise mean and standard deviation across the 21,000 training patches to produce `s2_stats.json`.
  - Developed the multi-label Mean Average Precision (mAP) calculation engine (`compute_map.py`).
  - Extracted baseline 2048-D validation embeddings and generated baseline 2D t-SNE scatter plots (`baseline_tsne.png`) illustrating the cross-sensor domain gap.
  - Generated visual retrieval comparison grids and qualitative success/failure case plots (`visual_retrieval_comparison.png`, `cases_report.csv`).

### 6.3 Person 3: Meet (SAR Preprocessing, Paired DataLoader, FAISS Galleries & Atlas Web Portal)
- Repository: `majorProject` (36 Python files).
- **Core Deliverables**:
  - Ingested raw Sentinel-1 SAR GeoTIFFs, implemented dB clipping `[-25, 0]` dB, and calculated training set statistics to produce `s1_stats.json`.
  - Developed the core PyTorch Dataset class (`paired_bigearthnet_dataset.py`) with cached disk path indexing and built multi-worker DataLoaders.
  - Developed the interactive, full-stack **Atlas Remote Sensing Intelligence Portal** (`app.py`, `templates/`, `static/`) providing drag-and-drop satellite retrieval with sub-500ms query latency.
  - Built the production FAISS galleries (`ms_gallery.index`, `sar_gallery.index`) indexing 4,500 test scenes.
  - Conducted detailed latency profiling (`timing.py`), measuring image decoding, model inference, and FAISS vector lookup.

### 6.4 Person 4: Fateh (Model Architecture Design, Training Baseline & Research Log)
- Repository: `majorproject_week3` (17 Python files).
- **Core Deliverables**:
  - Designed the foundational TwoTowerNetwork specifications and initial layer adaptation for 2-channel SAR input.
  - Implemented the baseline ResNet-50 feature extractor wrapper (`baseline_extractor.py`) and initial FAISS index testing.
  - Maintained the central project research log (`Project_log.md`) recording daily experimental progression, loss trajectories, and literature review summaries.

---

## 7. Experimental History: The Local 30k Dataset Benchmark

### 7.1 Dataset Configuration
- **Dataset Source**: BigEarthNet-S1 and BigEarthNet-S2 (v2.0).
- **Subset Size**: 30,000 paired satellite patches (stratified across 10 dominant Corine Land Cover classes, excluding snow, cloud, and shadow).
- **Split Breakdown**:
  - Training Set: **21,000 pairs** (70%)
  - Validation Set: **4,500 pairs** (15%)
  - Test Set: **4,500 pairs** (15%, held out as the evaluation gallery)

### 7.2 Reported Retrieval Benchmarks
> **STATUS LABEL**: `[PREVIOUSLY REPORTED]` — Metrics achieved on the local Windows PC workstation using `best_model-v1.ckpt` across 4,500 test patches:

| Retrieval Mode | Query Sensor | Gallery Sensor | Baseline (Untrained) Recall@10 | Trained (Two-Tower) Recall@10 | Absolute Performance Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cross-Modal** | Sentinel-1 (SAR) | Sentinel-2 (MS) | 1.13% | **75.04%** | **+73.91%** |
| **Cross-Modal** | Sentinel-2 (MS) | Sentinel-1 (SAR) | 1.09% | **74.00%** | **+72.91%** |
| **Same-Modal** | Sentinel-1 (SAR) | Sentinel-1 (SAR) | 12.33% | **97.47%** | **+85.14%** |
| **Same-Modal** | Sentinel-2 (MS) | Sentinel-2 (MS) | 94.67% | **98.42%** | **+3.75%** |

### 7.3 Latency Profile Breakdown
`[PREVIOUSLY REPORTED]` From `results/local_pc/week_05/benchmarking/timing_results.json`:
- Preprocessing & Image Tensor Resampling: ~360.0 ms
- Two-Tower Forward Embedding Pass (ResNet-50): ~60.7 ms
- FAISS `IndexFlatIP` Similarity Search (4,500 vectors): **1.2 ms**
- **Total End-to-End Latency**: **~421.9 ms**

---

## 8. Technical Component Inventory

### 8.1 Key Source Scripts (`weeks/`)
- `weeks/week_01/data_exploration/explore_metadata_adarsh.py`: Ingests and inspects `metadata.parquet`.
- `weeks/week_02/preprocessing/preprocess_s2_adarsh.py`: Standardizes optical RGB bands to shape `(3, 224, 224)`.
- `weeks/week_02/preprocessing/sar_preprocessing_meet.py`: Clips backscatter to `[-25, 0]` dB and normalizes SAR to `(2, 224, 224)`.
- `weeks/week_02/dataloader/paired_bigearthnet_dataset_meet.py`: PyTorch Dataset delivering `(sar, ms, label, id)` tuples with disk caching.
- `weeks/week_03/embeddings/generate_baseline_embeddings_adarsh.py`: Extracts 2048-D ImageNet feature vectors.
- `weeks/week_04/models/two_tower_jainam.py`: Two-Tower model definition with 512-D projection heads.
- `weeks/week_04/training/train_jainam.py`: PyTorch Lightning training loop with InfoNCE loss.
- `weeks/week_05/retrieval/retriever_meet.py`: Real-time `CrossModalRetriever` inference coordinator.
- `weeks/week_05/deployment/app_meet.py`: Production Flask application hosting the Atlas portal.

### 8.2 Models & Indices
- `shared/models/best_model-v1.ckpt` (598.95 MB): PyTorch Lightning checkpoint containing trained weights for both encoders and projection heads.
- `results/local_pc/week_05/faiss_indices/ms_gallery.index` (9.2 MB): Prebuilt FAISS IndexFlatIP indexing 4,500 test MS scenes.
- `results/local_pc/week_05/faiss_indices/sar_gallery.index` (9.2 MB): Prebuilt FAISS IndexFlatIP indexing 4,500 test SAR scenes.

---

## 9. Operational Limitations & Implementation Constraints

1. **Hardcoded Windows Paths**:  
   Many scripts currently contain hardcoded path strings (e.g. `C:\Users\hp\out\...`). These must be replaced with `.env` / `settings.py` variables before executing on Linux (PARAM SHAVAK).
2. **Inference Channel Duplication Issue**:  
   In `retriever_meet.py`, a fallback mechanism previously duplicated single bands when multi-band TIFFs were not resolved. This has been audited and flagged for resolution in Phase 4.
3. **Multi-Worker Spawning on Windows**:  
   Local DataLoaders used `num_workers=0` due to Windows spawn overhead. On PARAM SHAVAK (Linux), multi-processing DataLoaders (`num_workers=8` or `16`) must be enabled.
4. **Disaster Labels vs. Land-Cover Labels**:  
   The web portal presents operational disaster domain subcategories, but the underlying retrieval space operates on Corine Land Cover physical terrain features. The system retrieves terrain correspondence, not automatic disaster damage classifications.

