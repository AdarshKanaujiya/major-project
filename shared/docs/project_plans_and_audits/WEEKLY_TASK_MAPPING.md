# SENTINEL-RS: Weekly Task Mapping (Phase 1)
**Project Title**: Cross-Modal Remote Sensing Retrieval Using Multi-Sensor Satellite Data  
**Roadmap Reference**: Deep Cross-Modal Remote Sensing Image Retrieval Using Sentinel-1 and Sentinel-2 Data.md  

---

## Week 1: Environment, Dataset Exploration & Metadata Analysis
**Primary Objectives**: Environment configuration, dataset exploration, metadata filtering, band visualization.

| Member | Planned Weekly Task (from plan) | Implementing Script / File | Output Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Jainam (P1)** | Repository setup, environment specification, requirements.txt | cross-modal/requirements.txt, cross-modal/README.md | Python environment spec | Complete |
| **Jainam (P1)** | Sentinel-2 GeoTIFF RGB band extraction & visualization | cross-modal/scripts/view_random_images.py | 10 sample visual checks | Complete |
| **Adarsh (P2)** | Metadata exploration & inspection of clean BigEarthNet patches | deep-cross/analysiis of dataset/explore_metadata.py | metadata.parquet summary | Complete |
| **Adarsh (P2)** | Exclusion analysis for snow, cloud, and shadow patches (69,450) | deep-cross/analysiis of dataset/filter_clean_data.py | Filter verification report | Complete |
| **Adarsh (P2)** | Multi-label land cover class frequency & imbalance analysis | deep-cross/analysiis of dataset/label_analysis.py | Class distribution tables | Complete |
| **Meet (P3)** | Sentinel-1 SAR GeoTIFF loading, dB stats inspection | majorProject/inspect_dataset.py | SAR shape & min/max statistics | Complete |
| **Fateh (P4)** | Project documentation setup & foundational paper review | majorproject_week3/Project_log.md | SENTINEL-RS Project Log | Complete |

---

## Week 2: Preprocessing Pipelines, Subset Sampling & Paired DataLoader
**Primary Objectives**: 30k stratified subset creation, S1 & S2 normalization stats, Paired PyTorch Dataset & DataLoader.

| Member | Planned Weekly Task (from plan) | Implementing Script / File | Output Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Jainam (P1)** | S1-S2 pair correspondence verification | cross-modal/scripts/verify_pairs.py | Missing pair audit | Complete |
| **Jainam (P1)** | Stratified 30k subset creation & train/val/test splitting | cross-modal/scripts/create_final_splits.py | 	rain_split.csv (21k), al_split.csv (4.5k), 	est_split.csv (4.5k) | Complete |
| **Adarsh (P2)** | Sentinel-2 RGB band extraction (B04, B03, B02) & bilinear resize | deep-cross/preprocessing/preprocess_s2.py | Tensor shape (3, 224, 224) | Complete |
| **Adarsh (P2)** | Channel-wise RGB mean & std calculation on 21,000 training patches | deep-cross/compute_stats.py | s2_stats.json (Red, Green, Blue stats) | Complete |
| **Meet (P3)** | SAR preprocessing: [-25, 0] dB clipping, z-score scaling, resize | majorProject/preprocessing/sar_preprocessing.py | Tensor shape (2, 224, 224) | Complete |
| **Meet (P3)** | Channel-wise SAR VV/VH mean & std calculation on 21k train patches | majorProject/utils/stats_generator.py | s1_stats.json (VV, VH stats) | Complete |
| **Meet (P3)** | PairedBigEarthNetDataset PyTorch Dataset class implementation | majorProject/paired_bigearthnet_dataset.py | (sar, ms, label, id) tuples | Complete |
| **Meet (P3)** | Multi-worker PyTorch DataLoaders (batch_size=64, shuffle=True) | majorProject/dataset/dataloader.py | Train & Val DataLoader generators | Complete |

---

## Week 3: Baseline Retrieval Evaluation & t-SNE Visualization
**Primary Objectives**: Pretrained ResNet-50 feature extraction, baseline similarity search, mAP metric engine, baseline t-SNE.

| Member | Planned Weekly Task (from plan) | Implementing Script / File | Output Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Adarsh (P2)** | Pretrained ResNet-50 2048-D feature extraction for 4,500 val patches| deep-cross/generate_baseline_embeddings.py | aseline_val_ms_embeddings.npy (4500, 2048) | Complete |
| **Meet (P3)** | Pretrained ResNet-50 feature extraction for 4,500 SAR val patches | majorProject/utils/generate_s1_val_embeddings.py | aseline_val_s1_embeddings.npy (4500, 2048) | Complete |
| **Adarsh (P2)** | Multi-label Mean Average Precision (mAP) computation engine | deep-cross/recalls/compute_map.py | evaluate_retrieval_by_patch_id() | Complete |
| **Jainam (P1)** | Baseline cross-modal and same-modal retrieval evaluation | cross-modal/scripts/baseline_retrieval.py | Baseline metrics (SAR->MS R@10: 1.13%) | Complete |
| **Adarsh (P2)** | Baseline t-SNE 2D latent space visualization | deep-cross/week3/generate_tsne.py | outputs/visualizations/baseline_tsne.png | Complete |
| **Adarsh (P2)** | Before-training baseline performance report generation | deep-cross/week3/build_results_table.py | aseline_results_table.csv | Complete |

---

## Week 4: Two-Tower Network Architecture & Contrastive Training
**Primary Objectives**: ResNet-50 two-tower architecture, symmetric InfoNCE loss, training execution, overfitting monitoring.

| Member | Planned Weekly Task (from plan) | Implementing Script / File | Output Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Fateh (P4)** | TwoTowerNetwork architecture design (SAR 2-ch, MS 3-ch, 512-D head)| majorproject_week3/src/models/two_tower.py | TwoTowerNetwork PyTorch Module | Complete |
| **Fateh (P4)** | Symmetric InfoNCE contrastive loss function | cross-modal/src/models/two_tower.py | InfoNCELoss (temperature=0.07) | Complete |
| **Jainam (P1)** | Model training pipeline with AdamW, Cosine Annealing, validation | cross-modal/scripts/train.py | est_model-v1.ckpt (628 MB) | Complete |
| **Adarsh (P2)** | Epoch-wise loss tracking and overfitting analysis | deep-cross/week4/monitor_overfitting.py | metrics.csv, overfitting_analysis.txt | Complete |
| **Adarsh (P2)** | Training loss and validation loss visualization | deep-cross/week4/plot_training_loss.py | 	raining_loss.png | Complete |

---

## Week 5: Trained Retrieval Evaluation, FAISS Galleries & Atlas Portal
**Primary Objectives**: 512-D test embeddings, FAISS galleries, Recall@K & mAP evaluation, interactive web portal.

| Member | Planned Weekly Task (from plan) | Implementing Script / File | Output Artifact | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Jainam (P1)** | Extract 512-D L2-normalized embeddings for 4,500 test patches | cross-modal/scripts/extract_test_embeddings.py | 	rained_test_ms_embeddings.npy, 	rained_test_sar_embeddings.npy | Complete |
| **Jainam (P1)** | Build FAISS IndexFlatIP dual galleries (SAR & MS) | cross-modal/scripts/build_faiss_indices.py | ms_gallery.index, sar_gallery.index | Complete |
| **Jainam (P1)** | Evaluate trained cross-modal retrieval (SAR->MS, MS->SAR) | cross-modal/scripts/evaluate_trained_cross_modal.py | SAR->MS R@10: 75.04%, MS->SAR R@10: 74.00% | Complete |
| **Jainam (P1)** | Evaluate trained same-modal retrieval (SAR->SAR, MS->MS) | cross-modal/scripts/evaluate_trained_same_modal.py | SAR->SAR R@10: 97.47%, MS->MS R@10: 98.42% | Complete |
| **Adarsh (P2)** | Post-training t-SNE comparison plot (Baseline vs Trained) | deep-cross/week3/generate_tsne.py | 	sne_comparison.png | Complete |
| **Adarsh (P2)** | Success/failure visual retrieval grid generation | deep-cross/week5/visual_retrieval_grid.py | isual_retrieval_comparison.png | Complete |
| **Meet (P3)** | Interactive Atlas Remote Sensing Portal (Flask web application) | majorProject/app.py, majorProject/templates/ | Running localhost:5000 portal | Complete |
| **Meet (P3)** | Real-time CrossModalRetriever inference engine | majorProject/src/inference/retriever.py | Sub-500ms retrieval engine | Complete |
