# SENTINEL-RS — Week 3: Baseline Feature Extraction & Retrieval Evaluation

## 1. Overview & Research Objective
Week 3 establishes the empirical lower-bound benchmark for the cross-modal retrieval task before applying contrastive learning (Two-Tower InfoNCE in Week 4).
The baseline uses frozen, ImageNet-pretrained ResNet-50 backbones without any contrastive fine-tuning to extract 2048-dimensional feature representations from Sentinel-1 SAR and Sentinel-2 Multispectral validation imagery.

---

## 2. Directory Structure & File Inventory

```
weeks/week_03/
├── .backup_pre_correction_w3/            # Pristine backups of Week 3 files before corrections
├── embeddings/
│   ├── baseline_extractor_fateh.py        # [NON-CANONICAL] ResNet50 extractor with Strategy A channel init
│   ├── faiss_baseline_fateh.py            # Basic FAISS index build & query smoke test
│   ├── test_faiss_fateh.py                # Synthetic vector indexing test
│   ├── generate_baseline_embeddings_adarsh.py # [CANONICAL MS] Generates 2048-D MS baseline embeddings
│   └── generate_s1_val_embeddings_meet.py # [CANONICAL SAR] Generates 2048-D SAR baseline embeddings
├── evaluation/
│   ├── build_results_table_adarsh.py      # Template scaffold for baseline reporting
│   ├── compute_map_adarsh.py              # [DEPRECATED / SUPERSEDED] Legacy label-overlap evaluation
│   ├── compute_map_week3_adarsh.py        # [CANONICAL EVALUATOR] Cosine similarity + exact-pair protocol
│   └── evaluate_baseline_map_fateh.py     # Alternative MS->MS mAP evaluator
├── retrieval/
│   ├── baseline_retrieval_jainam.py       # SAR->MS retrieval test
│   ├── baseline_ms_to_ms_retrieval_jainam.py # MS->MS same-modal retrieval test
│   ├── baseline_ms_to_sar_retrieval_jainam.py # MS->SAR cross-modal retrieval test
│   └── baseline_sar_to_sar_retrieval_jainam.py # SAR->SAR same-modal retrieval test
└── visualization/
    ├── generate_tsne_adarsh.py            # t-SNE visualization of domain gap
    └── label_distribution_adarsh.py       # Multi-label class distribution chart
```

---

## 3. Canonical Architecture & Model Decisions

### 3.1 Sentinel-1 SAR Baseline Encoder
- **Canonical Model**: `ResNet50S1Encoder` from `weeks/week_04/models/resnet_meet.py`
- **Channel Adapter Strategy B**: Preserves VV (ch 0) and VH (ch 1) pretrained weights from RGB conv1, and distributes the Blue channel (ch 2) evenly across both polarizations:
  $$W_{conv1}[:, 0] = W_{orig}[:, 0] + \frac{W_{orig}[:, 2]}{2}, \quad W_{conv1}[:, 1] = W_{orig}[:, 1] + \frac{W_{orig}[:, 2]}{2}$$
  - Measured filter L2-norm: **3.25** (closely matching original 3-channel norm **3.14**; retains 100% of filter energy).
  - Provenance: Verified as the exact encoder used to produce historical `baseline_val_s1_embeddings.npy`.
- **Non-Canonical Alternative**: `baseline_extractor_fateh.py` (Strategy A: copies ch 0 and 1, discarding ch 2). Preserved for provenance.

### 3.2 Sentinel-2 Multispectral Baseline Encoder
- **Canonical Model**: Standard Torchvision `resnet50(weights=ResNet50_Weights.DEFAULT)` with `fc = nn.Identity()`.
- **Input Channels**: 3 (B04-Red, B03-Green, B02-Blue).
- **Output**: 2048-D average-pooled feature representation.

---

## 4. Evaluation Protocol & Standardization

### 4.1 Canonical Retrieval Metric
- **Standardized Metric**: Cosine similarity via **L2-normalized embeddings** and `faiss.IndexFlatIP`.
- **Equivalence**: Top-K rankings under normalized inner product are identical to normalized L2 distance.
- **Historical Distinction**: Original Jainam/Fateh scripts used raw L2 distance on unnormalized vectors. Both numbers are preserved and documented; new canonical runs standardize on cosine similarity for direct comparability with Week 5.

### 4.2 Canonical Relevance Definitions
| Retrieval Mode | Ground-Truth Relevance Protocol |
| :--- | :--- |
| **SAR → MS (cross-modal)** | **Exact paired patch match** via official `val_split.csv` mapping (S1 name $\leftrightarrow$ S2 patch ID). |
| **MS → SAR (cross-modal)** | **Exact paired patch match** via official `val_split.csv` mapping. |
| **SAR → SAR (same-modal)** | **Semantic label-overlap**: $\ge 1$ shared BigEarthNet class; self-match query patch excluded from top-K. |
| **MS → MS (same-modal)** | **Semantic label-overlap**: $\ge 1$ shared BigEarthNet class; self-match query patch excluded from top-K. |

### 4.3 Canonical Evaluator Script
- **Canonical Script**: `weeks/week_03/evaluation/compute_map_week3_adarsh.py` (incorporating localized per-query timing fix).
- **Deprecated**: `weeks/week_03/evaluation/compute_map_adarsh.py` (used label-overlap across all 4 modes, which is scientifically invalid for cross-modal evaluation).

---

## 5. Provenance & Artifacts Policy
- **Historical baseline embeddings** (`results/local_pc/week_03/baseline_embeddings/`) are read-only historical development assets and are **never overwritten**.
- Any new embedding extraction runs constitute new experiments and must output to dedicated, timestamped experiment folders.

