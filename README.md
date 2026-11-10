# SENTINEL-RS: Cross-Modal Satellite Image Retrieval System

**Deep Cross-Modal Remote Sensing Image Retrieval Using Sentinel-1 (SAR) and Sentinel-2 (Multispectral) Multi-Sensor Data**

---

## 1. Project Overview & Architecture

**SENTINEL-RS** is a deep learning cross-modal satellite image retrieval system capable of searching across different Earth observation sensors. Given a Synthetic Aperture Radar (SAR) query from Sentinel-1, it retrieves physically and semantically corresponding optical/multispectral scenes from Sentinel-2, and vice versa.

### Key Technical Specifications
- **Sensors**: Sentinel-1 SAR (C-band radar, dual-polarization: VV & VH) and Sentinel-2 Multispectral (RGB bands: B04, B03, B02).
- **Core Architecture**: Two-Tower Deep Neural Network with dual ResNet-50 backbones:
  - **SAR Tower**: Modified ResNet-50 accepting 2-channel normalized input (VV, VH) clipped to `[-25, 0]` dB.
  - **MS Tower**: Standard 3-channel ResNet-50 accepting standardized RGB composites (B04, B03, B02).
  - **Metric Heads**: Non-linear projection heads (2048 -> 1024 -> 512-D) producing L2-normalized metric embeddings.
- **Contrastive Learning**: Symmetric InfoNCE Loss with learnable temperature (`tau = 0.07`), optimized with AdamW and Cosine Annealing learning rate schedule.
- **Fast Similarity Search**: FAISS IndexFlatIP (Inner Product) galleries computing exact Cosine Similarity across 4,500 unseen test patches in under 2 ms.
- **Performance Benchmark**:
  - **Cross-Modal Retrieval (SAR -> MS)**: Recall@10 = **75.04%** (Baseline untrained = 1.13%, absolute gain = **+73.91%**).
  - **Cross-Modal Retrieval (MS -> SAR)**: Recall@10 = **74.00%** (Baseline untrained = 1.09%, absolute gain = **+72.91%**).
  - **Same-Modal Retrieval (SAR -> SAR)**: Recall@10 = **97.47%** (Baseline untrained = 12.33%).
  - **Same-Modal Retrieval (MS -> MS)**: Recall@10 = **98.42%** (Baseline untrained = 94.67%).

---

## 2. Consolidated Project Structure

The project has been organized according to the **Five-Week Research Development Roadmap** and features strict segregation between local workstation experiments and the PARAM SHAVAK HPC cluster environment:

```
SENTINEL-RS/
├── shared/                                 # Centralized project-wide assets
│   ├── config/                             # Requirements, YAML configs, settings, disaster mappings
│   ├── docs/                               # 5-week plan, audit reports, member logs and READMEs
│   ├── models/                             # Trained master model checkpoint (best_model-v1.ckpt)
│   └── utilities/                          # Shared I/O, configuration, and visualization modules
│
├── weeks/                                  # Source scripts organized by research milestone
│   ├── week_01/                            # Environment Setup, Dataset Exploration & Inspection
│   │   ├── data_exploration/               # Metadata parquets, class distribution & cloud filters
│   │   └── inspection/                     # SAR dB statistics, GeoTIFF headers & RGB visualizers
│   ├── week_02/                            # Preprocessing Pipelines & Paired DataLoader
│   │   ├── dataset_preparation/            # 30k stratified sampling, train/val/test splits, pair verification
│   │   ├── preprocessing/                  # S1 dB clipping [-25, 0], S2 RGB standardization, s1/s2_stats
│   │   └── dataloader/                     # PairedBigEarthNetDataset & multi-worker PyTorch DataLoaders
│   ├── week_03/                            # Baseline Retrieval & Zero-Shot Evaluation
│   │   ├── embeddings/                     # Untrained ResNet-50 2048-D feature extraction (4,500 patches)
│   │   ├── retrieval/                      # Pretrained cross-modal & same-modal baseline retrieval
│   │   ├── evaluation/                     # Multi-label Mean Average Precision (mAP) computation
│   │   └── visualization/                  # Pretrained 2D t-SNE projections & class distributions
│   ├── week_04/                            # Two-Tower Network & Contrastive Learning
│   │   ├── models/                         # TwoTowerNetwork (SAR 2-ch, MS 3-ch, 512-D heads)
│   │   ├── training/                       # Contrastive training loops, InfoNCE loss, AdamW optimizer
│   │   ├── evaluation/                     # Generalization gap & overfitting monitoring
│   │   └── visualization/                  # Epoch loss curves (training_loss.png)
│   └── week_05/                            # Trained Retrieval, FAISS Galleries & Atlas Portal
│       ├── embeddings/                     # Trained 512-D test embedding extraction (4,500 patches)
│       ├── retrieval/                      # Dual FAISS IndexFlatIP galleries & online retriever engine
│       ├── evaluation/                     # Final Recall@K and mAP evaluation scripts & comparison tables
│       ├── visualization/                  # Aligned t-SNE plots & query-retrieval visual grids
│       ├── benchmarking/                   # Query latency profiling (preprocessing + model + FAISS)
│       └── deployment/                     # Atlas Remote Sensing Intelligence Portal (Flask web app)
│           ├── templates/                  # HTML templates (domain.html, search.html, subcategory.html)
│           └── static/                     # CSS stylesheets (style.css)
│
├── results/                                # Centralized experimental artifacts
│   └── local_pc/                           # Historical outputs from the 30k local PC experiment
│       ├── week_01/metadata/               # Parquet metadata and coverage reports
│       ├── week_02/normalization_stats/    # s1_stats.json, s2_stats.json, statistics.json
│       ├── week_02/splits/                 # train_split.csv, val_split.csv, test_split.csv (30k)
│       ├── week_03/baseline_embeddings/    # baseline_val_ms_embeddings.npy, baseline_val_s1_embeddings.npy
│       ├── week_03/evaluation/             # Baseline Recall@K reports, baseline mAP results
│       ├── week_03/visualization/          # baseline_tsne.png, label_distribution.png
│       ├── week_04/checkpoints/            # best_model-v1.ckpt (598.95 MB), hparams.yaml
│       ├── week_04/metrics/                # metrics.csv, Week_4_training_record.txt
│       ├── week_04/visualization/          # training_loss.png
│       ├── week_05/trained_embeddings/     # trained_test_ms/sar_embeddings.npy (4500, 512)
│       ├── week_05/faiss_indices/          # ms_gallery.index, sar_gallery.index (FAISS IndexFlatIP)
│       ├── week_05/evaluation/             # trained_cross_modal_results.csv, final comparison reports
│       ├── week_05/visualization/          # tsne_comparison.png, visual_retrieval_comparison.png
│       └── week_05/benchmarking/           # timing_results.json (sub-500ms retrieval profile)
│
├── param_shavak/                           # PARAM SHAVAK HPC Cluster Environment (Segregated)
│   ├── week_01/ to week_05/                # Target directories for validated Linux cluster execution scripts
│   └── results/                            # Dedicated output directory for PARAM SHAVAK runs
│       ├── week_01/ to week_05/            # Ensures cluster outputs never mix with local PC results
│
├── needs_classification/                   # Specialized data copying, search, and deployment scripts
└── PROJECT_FILE_MAPPING.md                 # Complete provenance mapping of all 189 files
```

---

## 3. Team Member Contributions

The unified codebase brings together contributions from four team members:

| Member | Primary Role | Core Scripts & Contributions |
| :--- | :--- | :--- |
| **Jainam (Person 1)** | Project Lead & Pipeline Architecture | 30k stratified sampling (`create_final_splits.py`), pair verification, canonical training loop (`train.py`), FAISS gallery indexing, and final Week 5 performance reports. |
| **Adarsh (Person 2)** | Preprocessing, Evaluation Engine & Visualization | Metadata filtering parquets, 21,000-patch `s2_stats.json` calculation, `preprocess_s2.py`, multi-label mAP calculation engine (`compute_map.py`), and t-SNE visualizations. |
| **Meet (Person 3)** | SAR Pipeline, Web Portal & Latency Optimization | SAR dB clipping & `s1_stats.json` calculation, `paired_bigearthnet_dataset.py`, interactive Atlas Flask web portal (`app.py`), FAISS dual galleries, and query latency profiling. |
| **Fateh (Person 4)** | Model Architecture & Baseline Setup | Foundational Two-Tower architecture design, baseline feature extraction (`baseline_extractor.py`), initial contrastive training loops, and research project logging. |

---

## 4. How to Use the Codebase

### A. Local Development & Atlas Web Portal
To run the interactive Atlas web portal locally:
```bash
# Activate your Python virtual environment
# Ensure dependencies from shared/config/requirements.txt are installed
python weeks/week_05/deployment/app_meet.py
```
Open your browser to `http://localhost:5000` to upload Sentinel-1 or Sentinel-2 imagery and retrieve matching cross-modal satellite scenes in real time.

### B. PARAM SHAVAK HPC Execution Workflow
1. Working scripts validated on local PC will be staged into `param_shavak/week_01/` through `param_shavak/week_05/`.
2. Paths are resolved using Linux environment variables and `shared/config/settings.py`.
3. Execution outputs and checkpoints are written exclusively to `param_shavak/results/`, ensuring complete isolation from local PC benchmark data.
