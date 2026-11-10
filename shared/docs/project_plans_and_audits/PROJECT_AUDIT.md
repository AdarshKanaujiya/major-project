# SENTINEL-RS: Complete Project & Repository Audit (Phase 1)
**Project Title**: Cross-Modal Remote Sensing Retrieval Using Multi-Sensor Satellite Data (Sentinel-1 SAR and Sentinel-2 Multispectral)  
**Target Environment**: PARAM SHAVAK HPC Server (Linux, Multi-GPU) & Local Workstations  
**Audit Date**: November 2026  
**Auditor**: Senior ML Research Engineer & MLOps Architect  

---

## 1. Executive Summary & Repository Provenance

The project **SENTINEL-RS** was independently developed across four repositories by four team members, following a structured 5-week research and development roadmap:

| Member | Primary Role (per plan.md) | Identified Repository Folder | Total .py Scripts | Core Research Contribution |
| :--- | :--- | :--- | :---: | :--- |
| **Person 1 (Jainam)** | Project Lead, Data Pipeline, FAISS Retrieval | cross-modal-remote-sensing-retrieval | 30 | Master repo setup, subset verification, 30k pair splits, baseline retrieval scripts, final Week 5 cross-modal evaluation reports |
| **Person 2 (Adarsh)** | Metadata Exploration, S2 Preprocessing, mAP Engine | deep-cross-final-yr-project | 27 | metadata.parquet analysis, class distribution, 21k s2_stats.json calculation, preprocess_s2.py, mAP retrieval metric engine, Week 3–5 validation embeddings & t-SNE |
| **Person 3 (Meet)** | SAR Pipeline, Paired DataLoader, Web Portal | majorProject | 36 | sar_preprocessing.py, s1_stats.json, paired_bigearthnet_dataset.py, production Flask UI portal (pp.py), FAISS dual galleries (ms_gallery.index, sar_gallery.index) |
| **Person 4 (Fateh)** | Model Architecture, Training, Baseline Setup | majorproject_week3 | 17 | aseline_extractor.py, 	wo_tower.py, contrastive training loop (	rain_two_tower.py), baseline FAISS indexing (aiss_baseline.py), early web prototype |

---

## 2. Complete File-by-File Audit Table

### Legend for Recommendations:
- **REUSE**: Production-ready or canonical implementation; retain directly in unified tree.
- **MODIFY**: Core logic is solid, but paths or parameters need centralization via .env / config.
- **RERUN**: Code is usable, but embeddings / metrics / plots should be regenerated systematically on PARAM SHAVAK.
- **REBUILD**: Major flaws or anti-patterns detected; rewrite cleanly.
- **ARCHIVE**: Retain in historical / benchmark directory for provenance, do not use in main pipeline.

| Original Repository | Original File Path | Week | Component | Purpose | Dependencies | Status | Recommendation |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| cross-modal | scripts/create_final_splits.py | W02 | Dataset | Stratified sampling of 30k paired patches into train (21k), val (4.5k), test (4.5k) | pandas, numpy | Working | **REUSE** |
| cross-modal | scripts/create_subset.py | W02 | Dataset | Initial 30k patch subset filter from clean BigEarthNet metadata | pandas | Working | **ARCHIVE** |
| cross-modal | scripts/verify_pairs.py | W02 | Dataset | Verifies S1 SAR and S2 MS patch name correspondence | pandas | Working | **REUSE** |
| cross-modal | scripts/verify_dataloader.py | W02 | Dataset | Tests Paired DataLoader batch generation and timing | torch, DataLoader | Working | **REUSE** |
| cross-modal | src/models/two_tower.py | W04 | Models | TwoTowerNetwork with ResNet-50 SAR (2-ch) & MS (3-ch) + 512-D projection heads | torch, torchvision | Working | **REUSE** |
| cross-modal | scripts/train.py | W04 | Training | Training loop with Symmetric InfoNCE contrastive loss and checkpointing | torch, Lightning/PyTorch | Working | **MODIFY** |
| cross-modal | scripts/build_faiss_indices.py | W05 | Embeddings | Builds L2-normalized IndexFlatIP FAISS galleries for SAR and MS test sets | faiss, numpy | Working | **REUSE** |
| cross-modal | scripts/evaluate_trained_cross_modal.py| W05 | Evaluation | Computes Recall@1, 5, 10 and mAP for SAR->MS and MS->SAR retrieval | faiss, numpy, pandas | Working | **REUSE** |
| cross-modal | scripts/evaluate_trained_same_modal.py | W05 | Evaluation | Computes Recall@1, 5, 10 and mAP for SAR->SAR and MS->MS retrieval | faiss, numpy, pandas | Working | **REUSE** |
| cross-modal | scripts/create_week5_final_report.py | W05 | Evaluation | Generates final comparison tables between Baseline and Trained models | pandas | Working | **REUSE** |
| deep-cross | nalysiis of dataset/explore_metadata.py | W01 | Processing | Inspects BigEarthNet v2.0 metadata columns, counts, and data types | pandas, pyarrow | Working | **REUSE** |
| deep-cross | nalysiis of dataset/filter_clean_data.py| W01 | Processing | Excludes snow, cloud, and shadow patches using metadata parquets | pandas, pyarrow | Working | **REUSE** |
| deep-cross | nalysiis of dataset/label_analysis.py | W01 | Processing | Evaluates multi-label class frequency and imbalance across 19 classes | pandas, matplotlib | Working | **REUSE** |
| deep-cross | compute_stats.py | W02 | Processing | Computes channel-wise RGB mean & std across all 21,000 training patches | rasterio, numpy | Working | **REUSE** |
| deep-cross | preprocessing/preprocess_s2.py | W02 | Processing | Extracts B04, B03, B02, resizes to (3, 224, 224), and standardizes with s2_stats | rasterio, torch | Working | **REUSE** |
| deep-cross | generate_baseline_embeddings.py | W03 | Embeddings | Extracts 2048-D feature vectors from pretrained ResNet-50 for 4,500 val patches | torch, torchvision | Working | **REUSE** |
| deep-cross | 
ecalls/compute_map.py | W03 | Evaluation | Multi-label mAP and Recall engine by string patch_id matching | numpy, pandas | Working | **REUSE** |
| deep-cross | week3/generate_tsne.py | W03 | Visualization| Generates t-SNE 2D scatter plots of baseline embeddings colored by land cover | sklearn, matplotlib | Working | **REUSE** |
| deep-cross | week4/two_tower.py | W04 | Models | TwoTowerNetwork with InfoNCELoss implementation | torch, nn | Working | **REUSE** |
| deep-cross | week4/train.py | W04 | Training | Training loop with learning rate scheduling and epoch metrics logging | torch, tqdm | Working | **MODIFY** |
| deep-cross | week4/plot_training_loss.py | W04 | Visualization| Plots train/val loss curves across epochs from metrics.csv | matplotlib, pandas | Working | **REUSE** |
| deep-cross | week5/visual_retrieval_grid.py | W05 | Visualization| Renders visual retrieval grid comparing query scene with top-5 retrieved patches| PIL, matplotlib | Working | **REUSE** |
| deep-cross | week5/success_failure_cases.py | W05 | Visualization| Isolates Rank-1 exact matches vs semantic failure cases | pandas, PIL | Working | **REUSE** |
| majorProject | inspect_dataset.py | W01 | Processing | Inspects Sentinel-1 GeoTIFF headers, band min/max dB ranges | rasterio | Working | **REUSE** |
| majorProject | preprocessing/sar_preprocessing.py | W02 | Processing | Clips VV/VH to [-25, 0] dB, z-score normalizes with s1_stats, resizes to (2, 224, 224) | rasterio, torch | Working | **REUSE** |
| majorProject | paired_bigearthnet_dataset.py | W02 | Dataset | PyTorch Dataset loading paired S1 and S2 patches with cached path lookups | torch, rasterio | Working | **REUSE** |
| majorProject | dataset/dataloader.py | W02 | Dataset | Factory functions for multi-worker paired train, val, test DataLoaders | torch.utils.data | Working | **MODIFY** |
| majorProject | pp.py | W05 | Web App | Flask-based Atlas Intelligence Portal with live cross-modal retrieval & streaming | Flask, torch, faiss | Working | **MODIFY** |
| majorProject | src/inference/retriever.py | W05 | Retrieval | CrossModalRetriever class coordinating model forward pass, FAISS, and thumbnails | torch, faiss, PIL | Working | **MODIFY** |
| majorProject | evaluation/timing.py | W05 | Evaluation | Benchmarks average query latency (preprocessing + inference + FAISS search) | time, numpy | Working | **REUSE** |
| majorProject | evaluation/tsne_visualization.py | W05 | Visualization| Generates dual-modality t-SNE projections before and after contrastive learning | sklearn, matplotlib | Working | **REUSE** |
| majorproject_w3| src/models/baseline_extractor.py| W03 | Models | Untrained ResNet-50 feature extraction wrapper | torchvision | Working | **ARCHIVE** |
| majorproject_w3| scripts/faiss_baseline.py | W03 | Embeddings | Baseline FAISS IndexFlatL2 indexer | faiss | Working | **ARCHIVE** |
| majorproject_w3| scripts/train_two_tower.py | W04 | Training | Alternate training implementation | torch | Working | **ARCHIVE** |
| majorproject_w3| Project_log.md | W01-W05 | Docs | Formal experiment logs for paper record keeping | Markdown | Working | **REUSE** |

---

## 3. Duplication, Overlap & Divergence Analysis

1. **Two-Tower Architecture (	wo_tower.py)**:
   - cross-modal/src/models/two_tower.py, deep-cross/week4/two_tower.py, and majorProject/src/models/two_tower.py share identical encoder structures:
     - SAR Tower: Modified ResNet-50 with conv1 adapted from 3 channels to 2 channels (VV and VH) via weight averaging.
     - MS Tower: Standard 3-channel ResNet-50 (B04, B03, B02).
     - Projection Heads: Linear(2048 -> 1024) -> ReLU -> Linear(1024 -> 512).
   - **Canonical Selection**: cross-modal/src/models/two_tower.py is the cleanest, modular implementation.

2. **Preprocessing Modules**:
   - preprocess_s2.py (Adarsh): Best documented, correctly applies the 21,000-patch s2_stats.json.
   - sar_preprocessing.py (Meet): Correctly implements [-25, 0] dB clipping, z-score scaling with s1_stats.json, and bilinear interpolation to (2, 224, 224).

3. **DataLoaders & Datasets**:
   - majorProject/paired_bigearthnet_dataset.py contains optimized disk path caching (.s1_patch_cache.json).
   - deep-cross/week4/dataset.py is simpler and self-contained.
   - **Canonical Selection**: Unify into a single, clean PairedBigEarthNetDataset that supports both local paths and PARAM SHAVAK environment variables.

4. **FAISS Retrieval & Evaluation**:
   - Both cross-modal/scripts/ and majorProject/scripts/ have uild_faiss_indices.py and evaluate_trained_cross_modal.py.
   - They use IndexFlatIP (Inner Product) on L2-normalized 512-D vectors, which is mathematically exact for Cosine Similarity.
   - **Canonical Selection**: Jainam's cross-modal scripts for batch evaluation, Meet's src/inference/retriever.py for online single-query retrieval.

---

## 4. Model Checkpoint & Embeddings Inventory

| Artifact File | Location | Shape / Dimension | Status | Verified Integrity |
| :--- | :--- | :---: | :---: | :---: |
| est_model-v1.ckpt | deep-cross/models/ | 628 MB PyTorch State Dict | Verified | Contains complete trained weights for SAR + MS towers & projections |
| 	rained_test_ms_embeddings.npy | deep-cross/week5/ | (4500, 512) Float32 | Verified | Test set Multispectral 512-D normalized embeddings |
| 	rained_test_sar_embeddings.npy| deep-cross/week5/ | (4500, 512) Float32 | Verified | Test set SAR 512-D normalized embeddings |
| aseline_val_ms_embeddings.npy | deep-cross/outputs/ | (4500, 2048) Float32 | Verified | Pretrained baseline features (val set) |
| aseline_val_s1_embeddings.npy | deep-cross/outputs/ | (4500, 2048) Float32 | Verified | Pretrained baseline features (val set) |
| ms_gallery.index | majorProject/outputs/indices/ | 9.2 MB FAISS IndexFlatIP | Verified | 4,500 L2-normalized MS test vectors |
| sar_gallery.index| majorProject/outputs/indices/ | 9.2 MB FAISS IndexFlatIP | Verified | 4,500 L2-normalized SAR test vectors |

---

## 5. Identified Gaps & HPC Migration Issues

1. **Hardcoded Windows Paths**:
   - Scripts in all four repos contain hardcoded Windows strings like C:\Users\hp\out\.... These will break immediately on PARAM SHAVAK (Linux).
   - **Fix**: Centralize all paths in config/settings.py powered by .env.
2. **DataLoader Multiprocessing on Linux**:
   - Windows required 
um_workers=0 to prevent spawn deadlocks. PARAM SHAVAK's Linux kernel allows 
um_workers=8 or 16 with pin_memory=True for massive training speedups.
3. **Gallery Data Leakage Safeguard**:
   - All evaluation and gallery indices are confirmed built strictly from 	est_split.csv (4,500 unseen patches), preserving zero-leakage research integrity.
