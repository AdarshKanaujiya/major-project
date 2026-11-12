# SENTINEL-RS — PARAM SHAVAK Controlled Staging Manifest Report

**Execution Mode:** Controlled File Staging & SHA-256 Verification  
**Source Roots:** `weeks/`, `shared/`, `results/local_pc/`  
**Destination Root:** `param_shavak/`  
**Date:** 2026-11-12  
**Final Status:** **`STAGING COMPLETE — ALL HASHES VERIFIED`**  

---

## 1. Executive Summary

In accordance with authorized transfer preparation instructions, the canonical SENTINEL-RS Week 1–5 source code, configuration files, model checkpoint binary, and dataset metadata were staged into the local `param_shavak/` directory.

Every staged file was verified bit-for-bit against its canonical source using SHA-256 checksums. Pre-staged Week 1 and Week 2 files were inspected; all 22 existing source files matched their canonical counterparts with 0 discrepancies. An additional 82 canonical files were copied and verified, bringing the total staged inventory to 104 verified files.

**Key Highlights:**
- **Zero Conflicts:** No existing staged file was overwritten with differing contents.
- **Zero Hash Mismatches:** All 104 source and destination file hashes match 100%.
- **Checkpoint Verified:** `best_model-v1.ckpt` (628,042,368 bytes, SHA-256 `83c11cbfa3865dff907b28b4efc28817b40decb7d5790f4a88b3dc03c0423a7c`) is staged in both `param_shavak/models/` and `param_shavak/shared/models/`.
- **Metadata Verified:** Split CSVs (21k train, 4.5k val, 4.5k test), population normalization statistics, and the 4,500-row `test_patch_id_mapping.csv` were successfully staged.
- **Exclusions Enforced:** All 7 obsolete prototypes and historical results directories were strictly excluded.
- **Provenance Safety:** Canonical `weeks/`, `shared/`, and historical `results/local_pc/` files were unmodified.

---

## 2. Staging Statistics by Category

| Category | Staged Count | Verification Status | Notes |
|---|:---:|:---:|---|
| `EXISTING_STAGED_W1` | 7 | Retained Identical | Pre-staged Week 1 exploration & inspection scripts |
| `EXISTING_STAGED_W2` | 15 | Retained Identical | Pre-staged Week 2 preprocessing & dataloaders |
| `REQUIRED_SOURCE_W2` | 3 | Copied & Verified | Added canonical split creator, pair verifier, stats generator |
| `REQUIRED_SOURCE_W3` | 16 | Copied & Verified | Baseline zero-shot extractors, evaluators, retrievers |
| `REQUIRED_SOURCE_W4` | 9 | Copied & Verified | TwoTowerNetwork model, Lightning trainer, evaluation |
| `REQUIRED_SOURCE_W5` | 26 | Copied & Verified | Retrieval engine, evaluators, Atlas portal & UI assets |
| `REQUIRED_SHARED_MODULE` | 4 | Copied & Verified | Shared utilities (`static_code_auditor.py`, `visualization_meet.py`, etc.) |
| `REQUIRED_CONFIG` | 11 | Copied & Verified | HPC environment (`environment_hpc.yml`), pip requirements, configs |
| `REQUIRED_MODEL_CHECKPOINT` | 2 | Copied & Verified | 628 MB checkpoint (`param_shavak/models/` and `shared/models/`) |
| `REQUIRED_DATASET` | 11 | Copied & Verified | Split CSVs (train/val/test), stats JSONs, patch mapping CSV |
| **TOTAL** | **104** | **100% VERIFIED** | **Zero errors, zero conflicts, zero mismatches** |

---

## 3. Explicit Exclusion Audit

The following items were explicitly excluded from staging to protect HPC execution integrity:

| Excluded Path | Type | Reason for Exclusion | Status |
|---|---|---|:---:|
| `weeks/week_04/training/train_adarsh.py` | Python Script | Obsolete prototype containing legacy `src.data.dataloader` imports | **EXCLUDED** |
| `weeks/week_04/training/train_two_tower_fateh.py` | Python Script | Obsolete prototype containing legacy `src.dataset.dataloader` imports | **EXCLUDED** |
| `weeks/week_02/dataloader/test_paired_batch_jainam.py` | Python Script | Obsolete test containing legacy `src.data.dataloader` imports | **EXCLUDED** |
| `weeks/week_02/dataloader/verify_dataloader_jainam.py` | Python Script | Obsolete test containing legacy `src.data.dataloader` imports | **EXCLUDED** |
| `weeks/week_02/preprocessing/test_sar_preprocessing_jainam.py` | Python Script | Obsolete test containing legacy `src.preprocessing` imports | **EXCLUDED** |
| `weeks/week_05/deployment/app_fateh.py` | Python Script | Synthetic mock UI demo using random numbers; not real ML inference | **EXCLUDED** |
| `results/local_pc/week_05/trained_embeddings/derived_val_l2/` | Directory | Quarantined derived validation arrays from Phase B.2 | **EXCLUDED** |
| `results/local_pc/week_05/evaluation/*.csv` | Evaluation CSVs | Historical local PC results; must not pollute HPC outputs | **EXCLUDED** |
| Original contributor repositories (`cross-modal-*`, etc.) | Git Repositories | Superseded by canonical `weeks/` modules | **EXCLUDED** |
| Python Bytecode / Caches (`__pycache__`, `*.pyc`) | Cache Folders | Host-specific compiled bytecode | **EXCLUDED** |
| Backup Directories (`.backup*`) | Backup Folders | Local PC pre-correction backup archives | **EXCLUDED** |
| Raw Satellite Imagery (`BigEarthNet-S1`, `BigEarthNet-S2`) | Datasets | Large imagery volumes; cluster uses native mounts | **EXCLUDED** |

---

## 4. Complete Staging Manifest Table (104 Files)

| # | Relative Source Path | Staged Destination Path | Size (Bytes) | SHA-256 (First 16 chars) | Category | Staging Status |
|---|---|---|---:|---|---|:---:|
| 1 | `weeks/week_01/data_exploration/dataset_exploration_meet.py` | `param_shavak/week_01/data_exploration/dataset_exploration_meet.py` | 9,334 | `e8ba600771508c9f...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 2 | `weeks/week_01/data_exploration/explore_metadata_adarsh.py` | `param_shavak/week_01/data_exploration/explore_metadata_adarsh.py` | 677 | `da10abe29ec82e6b...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 3 | `weeks/week_01/data_exploration/explore_metadata_jainam.py` | `param_shavak/week_01/data_exploration/explore_metadata_jainam.py` | 2,618 | `000b1f46bbd339b9...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 4 | `weeks/week_01/data_exploration/filter_clean_data_adarsh.py` | `param_shavak/week_01/data_exploration/filter_clean_data_adarsh.py` | 1,496 | `3c7e2f93b4aadf52...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 5 | `weeks/week_01/data_exploration/label_analysis_adarsh.py` | `param_shavak/week_01/data_exploration/label_analysis_adarsh.py` | 2,307 | `2be281d880b459d0...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 6 | `weeks/week_01/inspection/inspect_dataset_meet.py` | `param_shavak/week_01/inspection/inspect_dataset_meet.py` | 12,278 | `0a6f8bb5c9aed43b...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 7 | `weeks/week_01/inspection/view_random_images_jainam.py` | `param_shavak/week_01/inspection/view_random_images_jainam.py` | 2,271 | `5fd1cafafca20213...` | `EXISTING_STAGED_W1` | `RETAINED_IDENTICAL` |
| 8 | `weeks/week_02/dataloader/dataloader_adarsh.py` | `param_shavak/week_02/dataloader/dataloader_adarsh.py` | 8,384 | `6573366765c4fc0f...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 9 | `weeks/week_02/dataloader/dataloader_fateh.py` | `param_shavak/week_02/dataloader/dataloader_fateh.py` | 8,416 | `2df52b5101bdb987...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 10 | `weeks/week_02/dataloader/dataloader_meet.py` | `param_shavak/week_02/dataloader/dataloader_meet.py` | 8,404 | `d8188b9b08ff3c39...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 11 | `weeks/week_02/dataloader/dataset_adarsh.py` | `param_shavak/week_02/dataloader/dataset_adarsh.py` | 21,631 | `113013b44a89774d...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 12 | `weeks/week_02/dataloader/dataset_fateh.py` | `param_shavak/week_02/dataloader/dataset_fateh.py` | 21,575 | `8ac2f83e2610782a...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 13 | `weeks/week_02/dataloader/dataset_meet.py` | `param_shavak/week_02/dataloader/dataset_meet.py` | 21,571 | `166c47c9464495da...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 14 | `weeks/week_02/dataloader/paired_bigearthnet_dataset_meet.py` | `param_shavak/week_02/dataloader/paired_bigearthnet_dataset_meet.py` | 16,733 | `9e562b4bd33189c8...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 15 | `weeks/week_02/preprocessing/preprocess_s2_adarsh.py` | `param_shavak/week_02/preprocessing/preprocess_s2_adarsh.py` | 4,502 | `1818487581ec5275...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 16 | `weeks/week_02/preprocessing/preprocess_s2_fateh.py` | `param_shavak/week_02/preprocessing/preprocess_s2_fateh.py` | 2,654 | `6005f24a970e8839...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 17 | `weeks/week_02/preprocessing/preprocess_s2_jainam.py` | `param_shavak/week_02/preprocessing/preprocess_s2_jainam.py` | 6,224 | `b71a09ea38a115fb...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 18 | `weeks/week_02/preprocessing/preprocess_s2_meet.py` | `param_shavak/week_02/preprocessing/preprocess_s2_meet.py` | 6,852 | `36e922652d47987e...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 19 | `weeks/week_02/preprocessing/sar_preprocessing_adarsh.py` | `param_shavak/week_02/preprocessing/sar_preprocessing_adarsh.py` | 8,281 | `7dd94a30cfc5db69...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 20 | `weeks/week_02/preprocessing/sar_preprocessing_fateh.py` | `param_shavak/week_02/preprocessing/sar_preprocessing_fateh.py` | 3,051 | `c274a104e62f6a49...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 21 | `weeks/week_02/preprocessing/sar_preprocessing_jainam.py` | `param_shavak/week_02/preprocessing/sar_preprocessing_jainam.py` | 8,274 | `c581576a750b8bad...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 22 | `weeks/week_02/preprocessing/sar_preprocessing_meet.py` | `param_shavak/week_02/preprocessing/sar_preprocessing_meet.py` | 16,494 | `21476eb47dd9919a...` | `EXISTING_STAGED_W2` | `RETAINED_IDENTICAL` |
| 23 | `weeks/week_02/dataset_preparation/create_final_splits_jainam.py` | `param_shavak/week_02/dataset_preparation/create_final_splits_jainam.py` | 5,773 | `34c6fd981046fd9e...` | `REQUIRED_SOURCE_W2` | `COPIED_VERIFIED` |
| 24 | `weeks/week_02/dataset_preparation/verify_pairs_jainam.py` | `param_shavak/week_02/dataset_preparation/verify_pairs_jainam.py` | 3,371 | `c522f0d2c4926543...` | `REQUIRED_SOURCE_W2` | `COPIED_VERIFIED` |
| 25 | `weeks/week_02/preprocessing/stats_generator_meet.py` | `param_shavak/week_02/preprocessing/stats_generator_meet.py` | 30,601 | `fd5c493224a83f70...` | `REQUIRED_SOURCE_W2` | `COPIED_VERIFIED` |
| 26 | `weeks/week_03/embeddings/baseline_extractor_fateh.py` | `param_shavak/week_03/embeddings/baseline_extractor_fateh.py` | 2,725 | `6175a71db01e6583...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 27 | `weeks/week_03/embeddings/faiss_baseline_fateh.py` | `param_shavak/week_03/embeddings/faiss_baseline_fateh.py` | 1,452 | `3ec70cd6834a5f4f...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 28 | `weeks/week_03/embeddings/generate_baseline_embeddings_adarsh.py` | `param_shavak/week_03/embeddings/generate_baseline_embeddings_adarsh.py` | 5,527 | `bb8244eb26f0ab34...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 29 | `weeks/week_03/embeddings/generate_s1_val_embeddings_meet.py` | `param_shavak/week_03/embeddings/generate_s1_val_embeddings_meet.py` | 19,262 | `4b2b9eddf272d1c6...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 30 | `weeks/week_03/embeddings/test_faiss_fateh.py` | `param_shavak/week_03/embeddings/test_faiss_fateh.py` | 649 | `7d07bca87f989efd...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 31 | `weeks/week_03/evaluation/build_results_table_adarsh.py` | `param_shavak/week_03/evaluation/build_results_table_adarsh.py` | 1,035 | `1121ba2cd8f7fa46...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 32 | `weeks/week_03/evaluation/compute_map_adarsh.py` | `param_shavak/week_03/evaluation/compute_map_adarsh.py` | 10,317 | `8873c13daa74c846...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 33 | `weeks/week_03/evaluation/compute_map_week3_adarsh.py` | `param_shavak/week_03/evaluation/compute_map_week3_adarsh.py` | 6,662 | `d7e1c48cfd7aa041...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 34 | `weeks/week_03/evaluation/evaluate_baseline_map_fateh.py` | `param_shavak/week_03/evaluation/evaluate_baseline_map_fateh.py` | 6,231 | `57b55168b24eb2d2...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 35 | `weeks/week_03/README.md` | `param_shavak/week_03/README.md` | 5,256 | `cb9ca2f0278d5e8f...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 36 | `weeks/week_03/retrieval/baseline_ms_to_ms_retrieval_jainam.py` | `param_shavak/week_03/retrieval/baseline_ms_to_ms_retrieval_jainam.py` | 11,448 | `b3a1885cf17b4fdf...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 37 | `weeks/week_03/retrieval/baseline_ms_to_sar_retrieval_jainam.py` | `param_shavak/week_03/retrieval/baseline_ms_to_sar_retrieval_jainam.py` | 11,123 | `a9ffc8a1cf79a718...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 38 | `weeks/week_03/retrieval/baseline_retrieval_jainam.py` | `param_shavak/week_03/retrieval/baseline_retrieval_jainam.py` | 14,484 | `cf871320b8592788...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 39 | `weeks/week_03/retrieval/baseline_sar_to_sar_retrieval_jainam.py` | `param_shavak/week_03/retrieval/baseline_sar_to_sar_retrieval_jainam.py` | 12,002 | `98ea55ebe5285162...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 40 | `weeks/week_03/visualization/generate_tsne_adarsh.py` | `param_shavak/week_03/visualization/generate_tsne_adarsh.py` | 6,407 | `fbd2c52df18d2241...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 41 | `weeks/week_03/visualization/label_distribution_adarsh.py` | `param_shavak/week_03/visualization/label_distribution_adarsh.py` | 2,589 | `f2023d79d5100d8d...` | `REQUIRED_SOURCE_W3` | `COPIED_VERIFIED` |
| 42 | `weeks/week_04/evaluation/monitor_overfitting_adarsh.py` | `param_shavak/week_04/evaluation/monitor_overfitting_adarsh.py` | 5,078 | `e096834dfbc4c0e7...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 43 | `weeks/week_04/models/resnet_meet.py` | `param_shavak/week_04/models/resnet_meet.py` | 2,418 | `7cb64a522147e60f...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 44 | `weeks/week_04/models/two_tower_adarsh.py` | `param_shavak/week_04/models/two_tower_adarsh.py` | 4,014 | `bee0ea604cd34bfa...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 45 | `weeks/week_04/models/two_tower_fateh.py` | `param_shavak/week_04/models/two_tower_fateh.py` | 3,006 | `88c30849dc53ea4e...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 46 | `weeks/week_04/models/two_tower_jainam.py` | `param_shavak/week_04/models/two_tower_jainam.py` | 6,162 | `d88b4c4112d15ee2...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 47 | `weeks/week_04/models/two_tower_meet.py` | `param_shavak/week_04/models/two_tower_meet.py` | 4,014 | `bee0ea604cd34bfa...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 48 | `weeks/week_04/README.md` | `param_shavak/week_04/README.md` | 9,887 | `511c8460d7077d40...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 49 | `weeks/week_04/training/train_jainam.py` | `param_shavak/week_04/training/train_jainam.py` | 14,089 | `1e5240dedc8f092d...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 50 | `weeks/week_04/visualization/plot_training_loss_adarsh.py` | `param_shavak/week_04/visualization/plot_training_loss_adarsh.py` | 3,442 | `d5c8369e9268d17a...` | `REQUIRED_SOURCE_W4` | `COPIED_VERIFIED` |
| 51 | `weeks/week_05/benchmarking/timing_meet.py` | `param_shavak/week_05/benchmarking/timing_meet.py` | 22,274 | `9edceeeabe7ab3f0...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 52 | `weeks/week_05/deployment/app_meet.py` | `param_shavak/week_05/deployment/app_meet.py` | 18,183 | `7f8e8adc36918728...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 53 | `weeks/week_05/deployment/run_pipeline_meet.py` | `param_shavak/week_05/deployment/run_pipeline_meet.py` | 10,763 | `7c99cd2ef7b0d56b...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 54 | `weeks/week_05/deployment/static/style.css` | `param_shavak/week_05/deployment/static/style.css` | 17,764 | `f2210d49994af2cc...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 55 | `weeks/week_05/deployment/templates/domain.html` | `param_shavak/week_05/deployment/templates/domain.html` | 1,792 | `e7ef663716c0700d...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 56 | `weeks/week_05/deployment/templates/search.html` | `param_shavak/week_05/deployment/templates/search.html` | 12,350 | `bc06a60adcfdb5f8...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 57 | `weeks/week_05/deployment/templates/subcategory.html` | `param_shavak/week_05/deployment/templates/subcategory.html` | 1,667 | `bd7874b6f225bd5d...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 58 | `weeks/week_05/embeddings/extract_test_embeddings_jainam.py` | `param_shavak/week_05/embeddings/extract_test_embeddings_jainam.py` | 9,694 | `4a811f85b5e65a47...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 59 | `weeks/week_05/embeddings/extract_test_embeddings_meet.py` | `param_shavak/week_05/embeddings/extract_test_embeddings_meet.py` | 9,796 | `eb793dce0f0b5008...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 60 | `weeks/week_05/embeddings/generate_trained_embeddings_adarsh.py` | `param_shavak/week_05/embeddings/generate_trained_embeddings_adarsh.py` | 11,947 | `20035f2977fa11a8...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 61 | `weeks/week_05/evaluation/create_week5_final_report_jainam.py` | `param_shavak/week_05/evaluation/create_week5_final_report_jainam.py` | 11,589 | `cf2e73fc90f64216...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 62 | `weeks/week_05/evaluation/evaluate_trained_cross_modal_jainam.py` | `param_shavak/week_05/evaluation/evaluate_trained_cross_modal_jainam.py` | 11,184 | `18f8d1cee4d265df...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 63 | `weeks/week_05/evaluation/evaluate_trained_cross_modal_meet.py` | `param_shavak/week_05/evaluation/evaluate_trained_cross_modal_meet.py` | 11,460 | `947a1763646c570b...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 64 | `weeks/week_05/evaluation/evaluate_trained_same_modal_jainam.py` | `param_shavak/week_05/evaluation/evaluate_trained_same_modal_jainam.py` | 16,298 | `f1b4672785202291...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 65 | `weeks/week_05/evaluation/evaluate_trained_same_modal_meet.py` | `param_shavak/week_05/evaluation/evaluate_trained_same_modal_meet.py` | 16,295 | `088f20f2cd1f09ca...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 66 | `weeks/week_05/evaluation/test_all_subcategories_meet.py` | `param_shavak/week_05/evaluation/test_all_subcategories_meet.py` | 14,065 | `5ac1417f20029fb7...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 67 | `weeks/week_05/README.md` | `param_shavak/week_05/README.md` | 14,432 | `b5d8ab4bb76391f6...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 68 | `weeks/week_05/retrieval/build_faiss_indices_jainam.py` | `param_shavak/week_05/retrieval/build_faiss_indices_jainam.py` | 7,460 | `bcd1c61b129f0e3e...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 69 | `weeks/week_05/retrieval/build_faiss_indices_meet.py` | `param_shavak/week_05/retrieval/build_faiss_indices_meet.py` | 7,323 | `9169192e12d35104...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 70 | `weeks/week_05/retrieval/identifier_mapper.py` | `param_shavak/week_05/retrieval/identifier_mapper.py` | 4,356 | `75f6c50234d72cfa...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 71 | `weeks/week_05/retrieval/retriever_meet.py` | `param_shavak/week_05/retrieval/retriever_meet.py` | 34,648 | `8f18c385b769e53e...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 72 | `weeks/week_05/retrieval/test_retriever_meet.py` | `param_shavak/week_05/retrieval/test_retriever_meet.py` | 14,508 | `7981c907fc8653b0...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 73 | `weeks/week_05/visualization/generate_trained_tsne_meet.py` | `param_shavak/week_05/visualization/generate_trained_tsne_meet.py` | 12,218 | `d81b5ea7c630c921...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 74 | `weeks/week_05/visualization/success_failure_cases_adarsh.py` | `param_shavak/week_05/visualization/success_failure_cases_adarsh.py` | 9,622 | `acbcdacd60b4618a...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 75 | `weeks/week_05/visualization/tsne_visualization_meet.py` | `param_shavak/week_05/visualization/tsne_visualization_meet.py` | 7,376 | `b658bac8c1b951bc...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 76 | `weeks/week_05/visualization/visual_retrieval_grid_adarsh.py` | `param_shavak/week_05/visualization/visual_retrieval_grid_adarsh.py` | 17,104 | `a43a49b800c658ad...` | `REQUIRED_SOURCE_W5` | `COPIED_VERIFIED` |
| 77 | `shared/config/config_jainam.yaml` | `param_shavak/shared/config/config_jainam.yaml` | 299 | `ae1301e1fd46f10d...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 78 | `shared/config/disaster_modality_config.json` | `param_shavak/shared/config/disaster_modality_config.json` | 3,798 | `d21fa2da10fb1f8c...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 79 | `shared/config/environment_hpc.yml` | `param_shavak/shared/config/environment_hpc.yml` | 539 | `93b3e5fb25295688...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 80 | `shared/config/requirements_hpc.txt` | `param_shavak/shared/config/requirements_hpc.txt` | 820 | `ca69381d490d57b3...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 81 | `shared/config/requirements_jainam.txt` | `param_shavak/shared/config/requirements_jainam.txt` | 186 | `a8543546fff7f497...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 82 | `shared/config/requirements_local.txt` | `param_shavak/shared/config/requirements_local.txt` | 826 | `22dd333a3ac6d3cc...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 83 | `shared/config/requirements_web.txt` | `param_shavak/shared/config/requirements_web.txt` | 31 | `006f5425895353fa...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 84 | `shared/config/settings.py` | `param_shavak/shared/config/settings.py` | 5,787 | `20afc3e60d5aa637...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 85 | `shared/config/settings_adarsh.py` | `param_shavak/shared/config/settings_adarsh.py` | 4,314 | `1fcc1c54db81de25...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 86 | `shared/config/settings_fateh.py` | `param_shavak/shared/config/settings_fateh.py` | 725 | `6460b7f50550ba5b...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 87 | `shared/config/settings_meet.py` | `param_shavak/shared/config/settings_meet.py` | 4,316 | `faa1193e5e8f9302...` | `REQUIRED_CONFIG` | `COPIED_VERIFIED` |
| 88 | `shared/utilities/config_jainam.py` | `param_shavak/shared/utilities/config_jainam.py` | 577 | `7f5aa7a004753384...` | `REQUIRED_SHARED_MODULE` | `COPIED_VERIFIED` |
| 89 | `shared/utilities/io_jainam.py` | `param_shavak/shared/utilities/io_jainam.py` | 1,659 | `2d1b4c6e6e727d29...` | `REQUIRED_SHARED_MODULE` | `COPIED_VERIFIED` |
| 90 | `shared/utilities/static_code_auditor.py` | `param_shavak/shared/utilities/static_code_auditor.py` | 25,244 | `2d607f04f741dfdf...` | `REQUIRED_SHARED_MODULE` | `COPIED_VERIFIED` |
| 91 | `shared/utilities/visualization_meet.py` | `param_shavak/shared/utilities/visualization_meet.py` | 10,041 | `23689c55ddf9c5e6...` | `REQUIRED_SHARED_MODULE` | `COPIED_VERIFIED` |
| 92 | `shared/models/best_model-v1.ckpt` | `param_shavak/models/best_model-v1.ckpt` | 628,042,368 | `83c11cbfa3865dff...` | `REQUIRED_MODEL_CHECKPOINT` | `COPIED_VERIFIED` |
| 93 | `shared/models/best_model-v1.ckpt` | `param_shavak/shared/models/best_model-v1.ckpt` | 628,042,368 | `83c11cbfa3865dff...` | `REQUIRED_MODEL_CHECKPOINT` | `COPIED_VERIFIED` |
| 94 | `results/local_pc/week_02/splits/subset_30k.csv` | `param_shavak/data/splits/subset_30k.csv` | 7,615,243 | `785660d6202f4506...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 95 | `results/local_pc/week_02/splits/test_split.csv` | `param_shavak/data/splits/test_split.csv` | 1,064,531 | `103436a01b7441f9...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 96 | `results/local_pc/week_02/splits/train_split.csv` | `param_shavak/data/splits/train_split.csv` | 4,936,307 | `8008195f2c675355...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 97 | `results/local_pc/week_02/splits/val_split.csv` | `param_shavak/data/splits/val_split.csv` | 1,101,827 | `01223bbe59f6af32...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 98 | `results/local_pc/week_02/normalization_stats/paired_dataloader_verification_summary.json` | `param_shavak/data/normalization_stats/paired_dataloader_verification_summary.json` | 642 | `6ce8daaa35b653ec...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 99 | `results/local_pc/week_02/normalization_stats/s1_stats.json` | `param_shavak/data/normalization_stats/s1_stats.json` | 738 | `3f45a9f9b4e33a57...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 100 | `results/local_pc/week_02/normalization_stats/s1_stats_regeneration_summary.json` | `param_shavak/data/normalization_stats/s1_stats_regeneration_summary.json` | 1,419 | `33c35a1b2e6632ce...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 101 | `results/local_pc/week_02/normalization_stats/s2_stats.json` | `param_shavak/data/normalization_stats/s2_stats.json` | 325 | `dd456c452abf6630...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 102 | `results/local_pc/week_02/normalization_stats/sar_preprocessing_test_results.json` | `param_shavak/data/normalization_stats/sar_preprocessing_test_results.json` | 416 | `80d6d7d426e84b26...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 103 | `results/local_pc/week_02/normalization_stats/statistics.json` | `param_shavak/data/normalization_stats/statistics.json` | 11,471 | `6eb6cefa5e7d3e53...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |
| 104 | `results/local_pc/week_05/trained_embeddings/test_patch_id_mapping.csv` | `param_shavak/data/test_patch_id_mapping.csv` | 605,941 | `5fe6c65bebcd794e...` | `REQUIRED_DATASET` | `COPIED_VERIFIED` |

---

## 5. Checkpoint & Critical Asset Verification

### 5.1 Model Checkpoint Verification
- **Source File:** `shared/models/best_model-v1.ckpt`
- **Primary Staged File:** `param_shavak/models/best_model-v1.ckpt`
- **Mirrored Staged File:** `param_shavak/shared/models/best_model-v1.ckpt`
- **File Size:** `628,042,368 bytes` (598.95 MB)
- **Source SHA-256:** `83C11CBFA3865DFF907B28B4EFC28817B40DECB7D5790F4A88B3DC03C0423A7C`
- **Primary Destination SHA-256:** `83C11CBFA3865DFF907B28B4EFC28817B40DECB7D5790F4A88B3DC03C0423A7C`
- **Mirrored Destination SHA-256:** `83C11CBFA3865DFF907B28B4EFC28817B40DECB7D5790F4A88B3DC03C0423A7C`
- **Bitwise Match:** 100% IDENTICAL.

### 5.2 Dataset Split CSV Verification
- `param_shavak/data/splits/train_split.csv`: 21,000 rows (Verified)
- `param_shavak/data/splits/val_split.csv`: 4,500 rows (Verified)
- `param_shavak/data/splits/test_split.csv`: 4,500 rows (Verified)
- **Disjointness:** Zero sample overlap across splits.

### 5.3 Bi-Directional Identifier Mapping Verification
- `param_shavak/data/test_patch_id_mapping.csv`: 4,500 rows (Verified)
- **Order Alignment:** 100% row-for-row alignment between canonical `patch_id`, native SAR `s1_name`, and native MS `s2v1_name`.

### 5.4 Normalization Statistics Verification
- `param_shavak/data/normalization_stats/s1_stats.json`: VV/VH training statistics (Verified)
- `param_shavak/data/normalization_stats/s2_stats.json`: B04/B03/B02 channel training statistics (Verified)

---

## 6. Known Unresolved Integration Issues (For Cluster Integration)

In accordance with strict staging instructions, source files were not modified during this staging operation. The following two known issues must be addressed when configuring the runtime environment on the cluster:

1. **Test Embedding Extractor Module Binding:**
   - `param_shavak/week_05/embeddings/extract_test_embeddings_jainam.py` and `meet.py` attempt `from weeks.week_02.dataset.paired_dataset import PairedBigEarthNetDataset`.
   - The module resides at `weeks.week_02.dataloader.paired_bigearthnet_dataset_meet.py`.
   - *Cluster Resolution:* Bind the import or utilize `generate_trained_embeddings_adarsh.py`.

2. **Atlas Web Portal Template Relative Path Binding:**
   - `param_shavak/week_05/deployment/app_meet.py` initializes Flask looking for `templates/` at repository root rather than inside `deployment/`.
   - *Cluster Resolution:* Update Flask `template_folder` and `static_folder` to `Path(__file__).resolve().parent / "templates"` and `static`.

---

## 7. Final Staging Decision

**`STAGING COMPLETE — ALL HASHES VERIFIED`**

All required Week 1–5 source code, shared utilities, configurations, split metadata, and the trained neural model checkpoint have been staged into `param_shavak/` with 100% bitwise SHA-256 verification. Zero file conflicts or hash mismatches occurred, all excluded scripts were omitted, and no local historical artifacts were modified.