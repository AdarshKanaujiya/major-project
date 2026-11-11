# RESEARCH_AND_AUDIT_LOG.md — Verification History & Cryptographic Audit Trail

> **DOCUMENT STATUS**: Formal Audit & Verification Log  
> **PROJECT**: SENTINEL-RS  
> **AUDITOR**: Senior ML Research Engineer & MLOps Architect  
> **DATE**: November 2026

---

## 1. Executive Audit Summary

This document preserves the step-by-step verification history, cryptographic audits, file reconciliation math, safety assessments, and investigation notes compiled during the restructuring of **SENTINEL-RS**.

Every statement in this document is explicitly labeled using the following standard:
- `[VERIFIED]`: Directly verified on disk via physical inspection or cryptographic hashing during this audit.
- `[PREVIOUSLY REPORTED]`: Historical claims or metrics recorded by the student team in previous logs/reports, not independently reproduced.
- `[PROPOSED]`: Recommendations or technical suggestions for future phases.
- `[UNVERIFIED]`: Hypotheses or parameters requiring confirmation on the HPC cluster.

---

## 2. Phase 1 Repository Scan & Inventory Audit

### 2.1 Four-Repository Scan Results
`[VERIFIED]` A physical traversal of the four original member repositories at `C:\Users\hp\out\major project\` was executed on November 10, 2026:

| Repository Name | Primary Author | Total Physical Files | Total Volume | Primary Content |
| :--- | :--- | :---: | :---: | :--- |
| `cross-modal-remote-sensing-retrieval` | Jainam (Person 1) | **64** | 7.44 MB | 30k split generation, training loop, evaluation reports |
| `deep-cross-final-yr-project` | Adarsh (Person 2) | **74** | 135.87 MB | Parquet metadata, S2 preprocessing, baseline embeddings, t-SNE |
| `majorProject` | Meet (Person 3) | **81** | 96.06 MB | SAR preprocessing, paired DataLoader, Flask Atlas portal, FAISS |
| `majorproject_week3` | Fateh (Person 4) | **40** | 113.86 MB | Two-Tower baseline design, contrastive loop, project logs |
| **Total Original Physical Files** | **All 4 Repositories** | **259** | **353.23 MB** | *(Excluding `.git/` internal Git database objects)* |

### 2.2 External Model Checkpoint Discovery
`[VERIFIED]` The master trained checkpoint was not stored inside `majorProject/models/` (which contained only a `.gitkeep` file). A filesystem scan located the canonical trained model at:
`C:\Users\hp\out\micro project\final ye project\models\best_model-v1.ckpt`  
- File Size: **598.95 MB** (628,041,728 bytes).
- Architecture: Complete PyTorch Lightning state dictionary containing ResNet-50 SAR Tower, ResNet-50 Multispectral Tower, and dual 512-D projection heads.

---

## 3. Mathematical Reconciliation of All 259 Original Files

`[VERIFIED]` The relationship between the 259 original repository files and the 189 consolidated destination files in `SENTINEL-RS` was cryptographically and mathematically proven:

$$\mathbf{259} \text{ (Physical files in 4 original member repos)}$$
$$- \mathbf{75} \text{ (Excluded duplicate, boilerplate, or placeholder files)}$$
$$= \mathbf{184} \text{ (Unique files consolidated from original repos)}$$

$$\mathbf{184} \text{ (Consolidated repo sources)}$$
$$+ \mathbf{3} \text{ (Root documents: Master Plan, PROJECT_AUDIT.md, WEEKLY_TASK_MAPPING.md)}$$
$$+ \mathbf{2} \text{ (Dual destination copies of trained model checkpoint)}$$
$$= \mathbf{189} \text{ (Total consolidated files on disk in SENTINEL-RS)}$$

### Progression from 182 to 189 Files Explained
- **Initial Consolidation Pass (182 destinations)**: Mapped 177 repository files + 3 root documents + 2 checkpoint destination copies = 182 files.
- **Second Audit Pass (+7 research files)**: A comprehensive review of unmapped files revealed 7 valuable research documents, notebooks, and reports that were previously overlooked:
  1. `majorProject/notebooks/dataset_exploration.ipynb` (11.5 KB) → `weeks/week_01/data_exploration/`
  2. `deep-cross/week5/results_analysis.md` (1.6 KB) → `results/local_pc/week_05/evaluation/`
  3. `deep-cross/week5/plot/cases_report.csv` (1.1 KB) → `results/local_pc/week_05/visualization/`
  4. `majorProject/outputs/week4_week5_dependency_report.json` (23.7 KB) → `results/local_pc/week_05/evaluation/`
  5. `majorProject/docs/README.md` (8.4 KB) → `shared/docs/README_sar_pipeline_meet.md`
  6. `majorProject/docs/WEEK3_README.md` (7.2 KB) → `shared/docs/WEEK3_README_meet.md`
  7. `majorProject/docs/WEEK5_README.md` (4.5 KB) → `shared/docs/WEEK5_README_meet.md`
- **Final Result**: Exactly $182 + 7 = \mathbf{189}$ physical files in `SENTINEL-RS`.

---

## 4. Full Catalog of the 75 Excluded Files & Cryptographic SHA-256 Audit

`[VERIFIED]` Every one of the 75 excluded files was checked and categorized. Thirty-seven pairs of candidate duplicate files were subjected to SHA-256 cryptographic hashing against their canonical counterpart in `SENTINEL-RS`:

```
================================================================================
CATEGORY 1: Git Repository Metadata & Placeholders (11 files)
================================================================================
1.  cross-modal-remote-sensing-retrieval/.gitignore
2.  cross-modal-remote-sensing-retrieval/configs/.gitkeep
3.  cross-modal-remote-sensing-retrieval/models/.gitkeep
4.  cross-modal-remote-sensing-retrieval/notebooks/.gitkeep
5.  cross-modal-remote-sensing-retrieval/outputs/.gitkeep
6.  cross-modal-remote-sensing-retrieval/outputs/images/.gitkeep
7.  cross-modal-remote-sensing-retrieval/scripts/.gitkeep
8.  cross-modal-remote-sensing-retrieval/tests/.gitkeep
9.  deep-cross-final-yr-project/.gitignore
10. majorProject/.gitignore
11. majorproject_week3/.gitignore

================================================================================
CATEGORY 2: Python Package Initializers / __init__.py Boilerplate (18 files)
================================================================================
12. cross-modal-remote-sensing-retrieval/src/__init__.py
13. cross-modal-remote-sensing-retrieval/src/models/__init__.py
14. cross-modal-remote-sensing-retrieval/src/preprocessing/__init__.py
15. cross-modal-remote-sensing-retrieval/src/utils/__init__.py
16. deep-cross-final-yr-project/config/__init__.py
17. majorProject/config/__init__.py
18. majorProject/dataset/__init__.py
19. majorProject/evaluation/__init__.py
20. majorProject/models/__init__.py
21. majorProject/preprocessing/__init__.py
22. majorProject/src/__init__.py
23. majorProject/src/models/__init__.py
24. majorProject/utils/__init__.py
25. majorproject_week3/src/__init__.py
26. majorproject_week3/src/config/__init__.py
27. majorproject_week3/src/dataset/__init__.py
28. majorproject_week3/src/models/__init__.py
29. majorproject_week3/src/preprocessing/__init__.py

================================================================================
CATEGORY 3: Empty Zero-Byte Placeholder Documents (2 files)
================================================================================
30. cross-modal-remote-sensing-retrieval/docs/PROJECT_LOG.md (0 bytes)
31. cross-modal-remote-sensing-retrieval/docs/references.md (0 bytes)

================================================================================
CATEGORY 4: Accidental Browser Download Duplicates (3 files)
================================================================================
32. majorproject_week3/s2_stats (1).json
    -> SHA-256: MATCH with s2_stats.json
33. majorproject_week3/outputs/embeddings/baseline_val_ms_embeddings (1).npy
    -> SHA-256: MATCH with baseline_val_ms_embeddings.npy
34. majorproject_week3/outputs/metadata/val_split (1).csv
    -> SHA-256: MATCH with val_split.csv

================================================================================
CATEGORY 5: Redundant Cross-Member Shared Copies in majorProject/teammate_inputs/ (8 files)
================================================================================
35. majorProject/teammate_inputs/baseline_val_ms_embeddings.npy
    -> SHA-256: MATCH with baseline_val_ms_embeddings.npy
36. majorProject/teammate_inputs/baseline_val_patch_ids.txt
    -> SHA-256: MATCH with baseline_val_ms_patch_ids.txt
37. majorProject/teammate_inputs/baseline_val_s1_embeddings.npy
    -> SHA-256: MATCH with baseline_val_s1_embeddings.npy
38. majorProject/teammate_inputs/baseline_val_s1_patch_ids.txt
    -> SHA-256: MATCH with baseline_val_s1_patch_ids.txt
39. majorProject/teammate_inputs/s2_stats.json
    -> SHA-256: MATCH with s2_stats.json
40. majorProject/teammate_inputs/test_split.csv
    -> SHA-256: MATCH with test_split.csv
41. majorProject/teammate_inputs/train_split.csv
    -> Semantic Match: 21,126 patch IDs 100% identical to canonical train_split.csv
42. majorProject/teammate_inputs/val_split.csv
    -> SHA-256: MATCH with val_split.csv

================================================================================
CATEGORY 6: Identical Duplicate Splits, Normalization Stats & Metrics (12 files)
================================================================================
43. deep-cross-final-yr-project/csvs/train_split.csv
    -> Semantic Match: 21,126 patch IDs 100% identical; hash differed only due to row sort order
44. deep-cross-final-yr-project/csvs/val_split.csv
    -> SHA-256: MATCH with val_split.csv
45. deep-cross-final-yr-project/csvs/test_split.csv
    -> SHA-256: MATCH with test_split.csv
46. deep-cross-final-yr-project/s1_stats.json
    -> SHA-256: MATCH with s1_stats.json
47. deep-cross-final-yr-project/s2_stats.json
    -> SHA-256: MATCH with s2_stats.json
48. deep-cross-final-yr-project/week4/metrics.csv
    -> SHA-256: MATCH with metrics.csv
49. deep-cross-final-yr-project/week4/training_loss.png
    -> SHA-256: MATCH with training_loss.png
50. majorProject/outputs/s1_stats.json
    -> SHA-256: MATCH with s1_stats.json
51. majorproject_week3/outputs/metadata/train_split.csv
    -> Semantic Match: 21,126 patch IDs 100% identical to canonical split
52. majorproject_week3/outputs/metadata/val_split.csv
    -> Early prototype: superseded by official val_split.csv
53. majorproject_week3/outputs/metadata/test_split.csv
    -> SHA-256: MATCH with test_split.csv
54. majorproject_week3/s2_stats.json
    -> Early prototype: computed on 18,067 images; superseded by official 21k s2_stats.json

================================================================================
CATEGORY 7: Identical Duplicate Text Evaluation Reports & Logs (8 files)
================================================================================
55. deep-cross-final-yr-project/outputs/reports/baseline_ms_to_ms_retrieval_results.txt
    -> SHA-256: MATCH with baseline_ms_to_ms_retrieval_results.txt
56. deep-cross-final-yr-project/outputs/reports/baseline_ms_to_sar_retrieval_results.txt
    -> SHA-256: MATCH with baseline_ms_to_sar_retrieval_results.txt
57. deep-cross-final-yr-project/outputs/reports/baseline_results_summary.txt
    -> SHA-256: MATCH with baseline_results_summary.txt
58. deep-cross-final-yr-project/outputs/reports/baseline_retrieval_results.txt
    -> SHA-256: MATCH with baseline_retrieval_results.txt
59. deep-cross-final-yr-project/outputs/reports/baseline_sar_to_sar_retrieval_results.txt
    -> SHA-256: MATCH with baseline_sar_to_sar_retrieval_results.txt
60. deep-cross-final-yr-project/outputs/reports/verification_report.txt
    -> SHA-256: MATCH with verification_report.txt
61. majorProject/outputs/reports/trained_cross_modal_results.csv
    -> Semantic Match: Recall@1, 5, 10 metrics identical; hash differed only on CPU query time
62. majorProject/outputs/reports/trained_same_modal_results.csv
    -> Semantic Match: Recall metrics identical; hash differed only on CPU query time

================================================================================
CATEGORY 8: Duplicate Baseline Embeddings & Patch ID Lists (6 files)
================================================================================
63. deep-cross-final-yr-project/week5/trained_test_ms_patch_ids.txt
    -> SHA-256: MATCH with trained_test_ms_patch_ids.txt
64. deep-cross-final-yr-project/week5/trained_test_sar_patch_ids.txt
    -> SHA-256: MATCH with trained_test_sar_patch_ids.txt
65. majorproject_week3/outputs/embeddings/baseline_val_ms_embeddings.npy
    -> Early prototype run: unstratified 4,500 patch run superseded by Adarsh's official embeddings
66. majorproject_week3/outputs/embeddings/baseline_val_patch_ids.txt
    -> Early prototype run: superseded by Adarsh's baseline_val_ms_patch_ids.txt
67. majorproject_week3/outputs/embeddings/baseline_val_s1_embeddings.npy
    -> SHA-256: MATCH with baseline_val_s1_embeddings.npy
68. majorproject_week3/outputs/metadata/baseline_val_s1_patch_ids.txt
    -> SHA-256: MATCH with baseline_val_s1_patch_ids.txt

================================================================================
CATEGORY 9: Duplicate Configuration & Requirement Files (2 files)
================================================================================
69. majorProject/config/disaster_modality_config.json
    -> SHA-256: Duplicate of config preserved in shared/config/disaster_modality_config.json
70. majorProject/requirements.txt
    -> Duplicate of requirements preserved in shared/config/requirements_jainam.txt

================================================================================
CATEGORY 10: Superseded Early-Draft Web Prototype Templates (Week 3) (4 files)
================================================================================
71. majorproject_week3/static/style.css
    -> Early prototype: superseded by Meet's full production Atlas CSS (17.7 KB)
72. majorproject_week3/templates/domain.html
    -> Early prototype: superseded by Meet's production domain template
73. majorproject_week3/templates/search.html
    -> Early prototype: superseded by Meet's production search template (12.3 KB)
74. majorproject_week3/templates/subcategory.html
    -> Early prototype: superseded by Meet's production subcategory template

================================================================================
CATEGORY 11: Cross-Member Duplicate Training Log (1 file)
================================================================================
75. deep-cross-final-yr-project/week4/Week_4_training_record_from_jainam.txt
    -> SHA-256: MATCH with Jainam's Week_4_training_record.txt
```

---

## 5. Auxiliary Files Audit (`needs_classification/`)

`[VERIFIED]` All 8 auxiliary files were inspected statically:

1. `check_required_s2_jainam.py`: Scans disk to verify required Sentinel-2 patch folders exist before dataset preprocessing.
2. `copy_required_s2_jainam.py`: Automates copying matching S2 patch folders from raw external storage into project data dir.
3. `build_s1_subset_meet.py`: Filters and extracts matching Sentinel-1 SAR patches corresponding to the 30k split.
4. `find_adarsh.py`: Filesystem search utility locating nested patch folders across deep directory trees.
5. `test_pipeline_meet.py`: End-to-end integration test verifying that DataLoader, TwoTower model forward pass, and FAISS querying run without errors.
6. `Procfile_fateh`: Heroku/Render process declaration (`web: python app.py`) for cloud web hosting.
7. `render_fateh.yaml`: Render.com Blueprint Infrastructure-as-Code specification for hosting the web interface.
8. `output_adarsh.txt`: Terminal stdout capture from Adarsh's initial data filtering and preprocessing verification pass.

---

## 6. Static Safety Assessment of `execute_consolidation.py`

`[VERIFIED]` Static analysis of the consolidation script confirms:
- **Zero Move Calls**: Contains only `shutil.copy2()`. Source repositories were never moved or renamed.
- **Zero Deletion Calls**: Contains no `os.remove()`, `os.unlink()`, or `shutil.rmtree()`.
- **Target Segregation**: Writes exclusively to `SENTINEL-RS/`. It cannot write to or overwrite anything in the four original member repositories or `param_shavak/`.
- **Reproducibility**: Serves as a deterministic, auditable script that reconstructs the consolidated file tree.

---

## 7. Environmental & Execution Anomalies Log

During previous turns, several tool and environment boundary anomalies were detected, investigated, and cleanly resolved:

1. **PowerShell Sandbox File Access**:  
   - *Anomaly*: Initial PowerShell commands targeting `C:\Users\hp\out\major project` returned `PermissionDenied` in standard sandbox mode.
   - *Resolution*: Switched to `BypassSandbox: true` as authorized for out-of-workspace directories.
2. **PowerShell Here-Strings & Variable Interpolation**:  
   - *Anomaly*: Multi-line PowerShell scripts failed with parse errors (`MissingVariableNameAfterForeach`, `AmpersandNotAllowed`) due to unescaped quotes.
   - *Resolution*: Replaced raw multi-line PowerShell scripts with Python one-liners (`python -c "..."`) and dedicated Python helper scripts.
3. **Background Task Timeouts**:  
   - *Anomaly*: Full directory recursive scans on `C:\Users\hp\out` timed out after 5,000 ms and were pushed into background tasks (`task-767`, `task-798`).
   - *Resolution*: Background tasks were cleanly terminated using `manage_task(Action='kill')` and replaced with targeted sub-second scans.
4. **Missing Python Global Packages**:  
   - *Anomaly*: Invoking `python -c "import numpy"` failed because global Windows Python lacked scientific packages.
   - *Resolution*: Performed all data comparisons, byte counts, and SHA-256 verifications using Python's built-in standard library (`hashlib`, `os`, `shutil`, `json`).

---

## 8. Phase 2 - Step 1: Automated Static AST Dependency & Path Audit

`[VERIFIED]` On November 10, 2026, the automated AST dependency and path audit utility (`shared/utilities/static_code_auditor.py`) was executed across `weeks/week_01/` through `weeks/week_05/`:

### 8.1 Files Discovered & Parsed
- **Total Files in `weeks/`**: **86 files**
  - **Python Scripts (`.py`)**: **81 scripts** (100% parsed successfully via `ast.parse`; **0 syntax errors**).
  - **Jupyter Notebooks (`.ipynb`)**: **1 notebook** (`week_01/data_exploration/dataset_exploration_meet.ipynb`).
  - **Web Non-Code Assets**: **4 files** (3 HTML templates + 1 CSS stylesheet under `weeks/week_05/deployment/`).
- **Execution Guards (`if __name__ == '__main__':`)**:
  - **54 runnable entry-point scripts** containing standalone execution guards.
  - **27 library-only scripts** (neural network definitions, dataset abstractions, preprocessing helper classes).

### 8.2 Dependency Ecosystem Inventory
- **Core Third-Party Dependencies (14 packages)**: `torch`, `torchvision`, `pytorch_lightning`, `lightning`, `faiss`, `rasterio`, `numpy`, `pandas`, `matplotlib`, `sklearn`, `tqdm`, `PIL`, `flask`, `werkzeug`.
- **Internal Package Import Chains**: Top-level imports targeting `src`, `config`, `utils`, `dataset`, `models`, `preprocessing`, `evaluation` were detected across 62 scripts. These require `PYTHONPATH` or standardized relative imports to function outside specific working directories.

### 8.3 Path Anomaly Summary
- **Absolute Windows Paths**: **20 critical instances** detected across 10 scripts (e.g. `C:\Users\hp\...`, `D:\Dataset\...`, `C:\BigEarthNet-S1-Required\...`).
- **Dataset / Artifact Path Literals**: **385 instances** (hardcoded references to `.tif`, `.parquet`, `.ckpt`, `.json`, `.csv`).
- **Working-Directory-Dependent Relative Paths**: **105 instances** (e.g. `outputs/`, `models/`, `data/`).
- **Companion Detailed Reports**:
  - Machine-readable: `shared/docs/project_plans_and_audits/phase2_dependency_and_path_audit.json`
  - Human-readable summary: `shared/docs/project_plans_and_audits/PHASE2_STEP1_STATIC_AUDIT_REPORT.md`

---

## 9. Phase 2 - Step 2: Week 1 Comparative Implementation Audit

`[VERIFIED]` On November 10, 2026, a read-only comparative source code audit was conducted across all 7 scripts and notebooks in `weeks/week_01/`:

### 9.1 Week 1 Script Inventory
1. `weeks/week_01/data_exploration/explore_metadata_adarsh.py` (19 lines)
2. `weeks/week_01/data_exploration/explore_metadata_jainam.py` (108 lines)
3. `weeks/week_01/data_exploration/filter_clean_data_adarsh.py` (42 lines)
4. `weeks/week_01/data_exploration/label_analysis_adarsh.py` (83 lines)
5. `weeks/week_01/data_exploration/dataset_exploration_meet.py` (228 lines)
6. `weeks/week_01/data_exploration/dataset_exploration_meet.ipynb` (Jupyter Notebook, 11.5 KB)
7. `weeks/week_01/inspection/inspect_dataset_meet.py` (314 lines)
8. `weeks/week_01/inspection/view_random_images_jainam.py` (98 lines)

### 9.2 Key Technical Findings & Comparative Evaluation
- **Metadata Parquet Exploration**:
  - `explore_metadata_adarsh.py` is minimal and relies on `metadata.parquet` existing in current working directory.
  - `explore_metadata_jainam.py` uses PyArrow for memory-efficient row group inspection, but has hardcoded `D:\Dataset\metadata.parquet` and duplicate Counter imports.
  - `filter_clean_data_adarsh.py` performs the critical verification proving zero overlap between clean patches and the 69,450 cloud/snow/shadow contaminated patches.
  - `label_analysis_adarsh.py` establishes the 19-class frequency distribution that directly informed the Week 2 stratified sampling strategy.
- **SAR Dataset Inspection**:
  - `inspect_dataset_meet.py` is production-quality: contains comprehensive logging, type annotations, `rasterio` with `PIL` fallback, and generates `dataset_inspection_report.md`.
- **Optical Band Visualization**:
  - `view_random_images_jainam.py` correctly resolves bands `_B04.tif`, `_B03.tif`, `_B02.tif`, but contains hardcoded `D:\Dataset` and calls `plt.show()` (which hangs on headless Linux servers). Needs `plt.savefig()` for HPC compatibility.

### 9.3 HPC Readiness Assessment (November 10, 2026)
`[VERIFIED via static inspection]` All 8 Week 1 scripts are **logically correct** for the problem statement. The science and logic are sound. All Week 1 EDA goals were met (SAR structure confirmed, metadata schema confirmed, cloud/snow exclusion verified, label distribution established, S2 RGB visualized). **None are HPC-ready as-is** due to operational blockers:

| Script | Blockers |
| :--- | :--- |
| `explore_metadata_adarsh.py` | Relative path `"metadata.parquet"` (CWD-dependent) |
| `explore_metadata_jainam.py` | Hardcoded `D:\Dataset\metadata.parquet`; duplicate `Counter` import |
| `filter_clean_data_adarsh.py` | Relative paths for both Parquet files |
| `label_analysis_adarsh.py` | Relative path `"metadata.parquet"` |
| `dataset_exploration_meet.py` | `config.settings` import assumes old repo root; `BASE_DIR/"outputs"` target |
| `inspect_dataset_meet.py` | `config.settings` import (graceful fallback exists); output path |
| `view_random_images_jainam.py` | Hardcoded `D:\Dataset`; `plt.show()` hangs on headless Linux; 3-level path nesting needs verification |

**Required Phase 3/4 fixes** (not to be done now):
1. Replace all hardcoded/relative paths via `shared/config/settings.py`.
2. Replace `plt.show()` with `plt.savefig()` + `plt.close()` in `view_random_images_jainam.py`.
3. Remove duplicate `Counter` import from `explore_metadata_jainam.py`.
4. Verify directory nesting assumption in `view_random_images_jainam.py`.
5. Route all outputs to `results/local_pc/week_01/` after config centralization.

---

## 10. Phase 2 - Step 2: Week 2 Comparative Implementation Audit

`[VERIFIED]` On November 10, 2026, a read-only comparative audit was conducted across all 28 Week 2 scripts:

### 10.1 Week 2 Script Inventory (28 scripts)
- `dataset_preparation/`: 5 scripts (Jainam split creation + Meet subset)
- `preprocessing/`: 14 scripts (4× S2, 4× SAR, stats generator, test/visualize helpers)
- `dataloader/`: 9 scripts (3× Dataset, 3× DataLoader, 1× paired standalone, 2× verification)

### 10.2 Critical Findings

**S2 Preprocessing**:
- `preprocess_s2_fateh.py` has inverted R/B channel order (loads B02→B03→B04 without reordering, then applies population stats computed for B04→B03→B02 order). `[VERIFIED]` correctness bug.
- `preprocess_s2_jainam.py` and `preprocess_s2_fateh.py` use per-patch Z-score normalization instead of population statistics — incompatible with training pipeline.
- `preprocess_s2_meet.py` is canonical candidate: correct channel order, population stats, NaN guard, shape validation.

**SAR Preprocessing**:
- The 45-line `sar_preprocessing_fateh.py` (actual file on disk) skips dB clipping entirely and uses per-patch normalization — not suitable for training.
- `sar_preprocessing_adarsh.py` and the longer version are functionally equivalent: clip[-25,0] + z-score + resize + NaN check.
- Canonical candidate: `sar_preprocessing_adarsh.py`.

**Dataset/DataLoader**:
- `dataset_adarsh.py` has broken import `from week4.sar_preprocessing` — will fail from `weeks/week_02/dataloader/`.
- `dataloader_adarsh.py` has broken import `from week4.dataset` — same issue.
- `paired_bigearthnet_dataset_meet.py` and `dataset_fateh.py` have correct import paths.
- `dataloader_meet.py` canonical DataLoader; default `num_workers=0` explicitly for Windows — must be raised to 8+ on PARAM SHAVAK.
- All three PairedBigEarthNetDataset classes are functionally identical (same pairing logic, CSV columns, S2 resolution, label encoder).

**PARAM SHAVAK Migration Checklist created**: `shared/docs/project_plans_and_audits/PARAM_SHAVAK_MIGRATION_CHECKLIST.md` — 32 items across 5 categories.

### 10.3 Canonical Candidates Summary

| Component | Canonical | Key Reason |
| :--- | :--- | :--- |
| S2 Preprocessing | `preprocess_s2_jainam.py` / `meet.py` | Direct B04-B03-B02 order, population Z-score using verified `s2_stats.json`, NaN guard |
| SAR Preprocessing | `sar_preprocessing_adarsh.py` | Clip [-25, 0] dB + z-score + NaN guard; self-contained |
| Paired Dataset | `paired_bigearthnet_dataset_meet.py` | Robust 4-candidate directory resolution; deterministic 19-class label encoding |
| DataLoader | `dataloader_meet.py` | Full batch preservation (`drop_last_train=False`), deterministic val shuffle |

---

## 11. Week 1 & 2 Empirical Revalidation and Approved Source Corrections (November 11, 2026)

### 11.1 Empirical Dataset & Raster Ground Truth
`[VERIFIED BY EXECUTION & DATA INSPECTION]`
- **GeoTIFF Formats (Directly Inspected with `rasterio`)**:
  - **Sentinel-1 SAR**: Stored as `float32` natively in **Decibels (dB)**. VH range $[-30.20, -6.05]\text{ dB}$; VV range $[-25.31, +5.20]\text{ dB}$. Outlier clipping to $[-25.0, 0.0]\text{ dB}$ verified mandatory before normalization. No linear-to-dB conversion permitted.
  - **Sentinel-2 Optical**: Stored as `uint16` in **raw Digital Numbers (DN)**. Mean range $341 - 638\text{ DN}$, matching `s2_stats.json` ($\mu = [587.03, 613.28, 438.19]$). No $/10000$ scaling permitted.
- **Split Integrity & Zero Leakage**:
  - `train_split.csv` ($21,000$), `val_split.csv` ($4,500$), `test_split.csv` ($4,500$).
  - Overlap between train, val, and test: **Strictly 0**.
  - All training patches originate strictly from `metadata.parquet['split'] == 'train'`.
  - Statistics `s1_stats.json` and `s2_stats.json` were accumulated globally over all $302,400,000$ pixels of `train_split.csv` with zero validation/test leakage.
- **Quarantined Patches**:
  - `metadata.parquet` (480,038 patches) has 0 seasonal snow and 0 cloud/shadow patches.
  - `metadata_for_patches_with_snow_cloud_or_shadow.parquet` (69,450 patches) is 100% disjoint from `metadata.parquet`.

### 11.2 Approved Source Code Corrections Applied
All 15 target source files were backed up in `weeks/week_02/.backup_pre_correction/` with SHA-256 hashes recorded before editing:
1. `preprocess_s2_fateh.py`: Inverted band order `("B02", "B03", "B04")` fixed to `("B04", "B03", "B02")`; per-patch normalization replaced with population Z-score using canonical `s2_stats.json`.
2. `sar_preprocessing_fateh.py`: Added $[-25.0, 0.0]\text{ dB}$ clipping; replaced per-patch normalization with population Z-score using canonical `s1_stats.json`.
3. `preprocess_s2_meet.py`: Removed silent fallback to divergent hardcoded stats; explicitly resolves canonical `s2_stats.json` and raises `FileNotFoundError` if missing; added `np.isfinite` guard.
4. `preprocess_s2_jainam.py`: Corrected `STATS_PATH` resolution to canonical `s2_stats.json`.
5. `preprocess_s2_adarsh.py`: Corrected `STATS_FILE` resolution to canonical `s2_stats.json`; added safe band lookup and std division guard.
6. `sar_preprocessing_adarsh.py`: Corrected `STATS_PATH` depth resolution to canonical `s1_stats.json`.
7. `sar_preprocessing_jainam.py`: Corrected `STATS_PATH` depth resolution to canonical `s1_stats.json`.
8. `sar_preprocessing_meet.py`: Provided self-contained constants and canonical stats path fallback; fixed `NORMALIZATION_MODE = "z-score"`.
9. `dataset_adarsh.py`: Corrected broken `week4` imports to local Week 2 preprocessor modules.
10. `dataloader_adarsh.py`: Corrected broken `week4.dataset` import to `dataset_adarsh`; updated default CSV paths to canonical `SPLITS_DIR`.
11. `paired_bigearthnet_dataset_meet.py`: Provided self-contained constants; updated imports to local Week 2 preprocessors.
12. `dataset_meet.py`: Provided self-contained constants; updated imports to local Week 2 preprocessors.
13. `dataloader_meet.py`: Updated dataset import from `dataset_meet`; updated default CSV paths to canonical `SPLITS_DIR`.
14. `dataset_fateh.py`: Provided self-contained constants; updated imports to local Week 2 preprocessors.
15. `dataloader_fateh.py`: Updated dataset import from `dataset_fateh`; updated default CSV paths to canonical `SPLITS_DIR`.

### 11.3 Multi-Level Validation Results
`[VERIFIED BY EXECUTION]`
1. **Compilation Check**: `python -m py_compile` passed across all 15 files with 0 errors.
2. **Import Smoke Test**: All 15 modules imported cleanly via standard Python import mechanism.
3. **Preprocessing Test on Real GeoTIFFs**:
   - All four S2 preprocessors produced: shape `(3, 224, 224)`, `finite=True`, range `[-0.68, 2.41]`.
   - All four SAR preprocessors produced: shape `(2, 224, 224)`, `finite=True`, range `[-2.34, 2.87]`.
4. **DataLoader Real-Batch Test**:
   - All three DataLoaders (`dataloader_meet`, `dataloader_adarsh`, `dataloader_fateh`) initialized 21,000 train / 4,500 val samples and yielded valid real batches:
     `SAR: torch.Size([4, 2, 224, 224])`, `MS: torch.Size([4, 3, 224, 224])`, `Labels: torch.Size([4, 19])`, `IDs: 4`.
5. **Model Interface Compatibility**:
   - Real batch passed through Two-Tower ResNet-50 and InfoNCE loss:
     `SAR Embedding: torch.Size([4, 512])`, `MS Embedding: torch.Size([4, 512])`, `Loss: 1.3863 (finite)`.
   - L2 normalization confirmed inside `InfoNCELoss.forward` via `F.normalize(..., dim=1)`. Raw projection head outputs unnormalized features.

### 11.4 Acceptance Status
- **Week 1**: Accepted with Qualifications (CWD path cleanup deferred to HPC preparation).
- **Week 2**: **SCIENTIFICALLY VALIDATED & ACCEPTED FOR CANONICAL PIPELINE**.
- **PARAM SHAVAK Staging**: Ready, awaiting explicit user authorization.

---

## 12. Final Numerical Consistency, Checkpoint Compatibility & PARAM SHAVAK Staging (November 11, 2026)

### 12.1 Final Numerical Cross-Contributor Equivalence
\[VERIFIED BY EXECUTION]\
- **Sentinel-2 Multispectral**:
  - Adarsh vs Meet (PyTorch \F.interpolate\): MaxAbsDiff = **0.000000e+00**, MeanAbsDiff = **0.000000e+00** (bitwise identical).
  - Jainam vs Fateh (PIL \Image.resize\): MaxAbsDiff = **0.000000e+00**, MeanAbsDiff = **0.000000e+00** (bitwise identical).
  - PyTorch vs PIL: Difference bounded by single-precision floating point coordinate rounding (< 1.0e-06).
- **Sentinel-1 SAR**:
  - Adarsh vs Meet (PyTorch \F.interpolate\): MaxAbsDiff = **0.000000e+00**, MeanAbsDiff = **0.000000e+00** (bitwise identical).
  - Jainam vs Fateh (PIL \Image.resize\): MaxAbsDiff <= 5.96e-07, MeanAbsDiff <= 9.77e-08 (float32 mantissa precision limit).

### 12.2 InfoNCE Loss Investigation & Mathematical Ground Truth
\[VERIFIED BY MATHEMATICAL DERIVATION & EXECUTION]\
- **Untrained Random Init Loss = 1.386601**:
  - Output projections are unnormalized in \TwoTowerNetwork\ (norms ~50-130).
  - \InfoNCELoss.forward\ applies \F.normalize(..., dim=1)\, projecting embeddings to the 512-D unit hypersphere.
  - On random weights, positive and negative pairs have near-identical cosine similarities (~ -0.013).
  - Temperature \	au = 0.07\ scales logits uniformly to ~ -0.18.
  - Softmax probability over batch B=4 is uniform \P = 1/4\.
  - Theoretical chance entropy: \-\ln(1/4) = \ln(4) = 1.386294...\. Difference is 3.06e-04.
  - Conclusion: 1.3863 is the exact chance baseline for B=4, not a trained retrieval loss.

### 12.3 Historical Checkpoint Compatibility (\est_model-v1.ckpt\)
\[VERIFIED BY STRICT LOAD & INFERENCE]\
- Checkpoint: \shared/models/best_model-v1.ckpt\ (598.95 MB, 644 parameter tensors).
- Stripping PyTorch Lightning \
etwork.\ prefix yielded **0 missing keys and 0 unexpected keys** against \TwoTowerNetwork\.
- Real batch inference (B=4):
  - Output embeddings: Shape \(4, 512)\, dtype \loat32\, completely finite.
  - **Positive pairs (diagonal)**: \[0.9329, 0.9287, 0.7922, 0.8586]\ (Mean: **0.8781**).
  - **Negative pairs (off-diagonal)**: Range \[-0.3509, +0.2152]\ (Mean: **-0.0952**).
  - \Diagonal > Off-diagonal: True\ for all pairs.
  - **Trained InfoNCE Loss**: **0.000048** (down from 1.3866, 99.996% reduction).

### 12.4 Resolved Report Issues & Recorded Audit Qualifications
1. **Off-Diagonal Cosine Similarity Mean**: Recalculated accurately from the 12 off-diagonal elements of the trained similarity matrix: sum = -1.1421, mean = **-0.0952** (corrected from previous run-variation value -0.0898).
2. **Geographic Tile Coverage**: The 8 tested patches represent **7 unique MGRS tiles** (\T29UNB\, \T29SNC\, \T35VNL\, \T29UPU\, \T29UPV\, \T34TFN\, \T35VPK\), as tile \T35VNL\ was tested on two separate patches (\43_10\ and \44_11\). All 8 samples produced 100% finite tensors.
3. **Embedding Divergence Qualification**: While PyTorch and PIL preprocessors match at the pixel level within <= 1.0e-06, full assertion of zero embedding divergence on trained models requires evaluating both tensors through the identical checkpoint.
4. **Local-Subset vs. Complete Dataset**: The existing normalization statistics (\s1_stats.json\, \s2_stats.json\) represent the 21,000-patch local PC training split. They must NOT be treated as full-dataset statistics. Full-dataset statistics must be recomputed on PARAM SHAVAK using the complete training split.

### 12.5 Provenance Verification & Staging Outcome
- Provenance confirmed: Staged files originated from canonical repository \C:\Users\hp\out\major project\SENTINEL-RS\weeks\week_01\ and \week_02\.
- Staged into \param_shavak/week_01/\: 7 audited source files. Marked as \AUDITED — HPC PATH ADAPTATION PENDING\.
- Staged into \param_shavak/week_02/\: 15 verified source files across Preprocessing and DataLoader. All SHA-256 hashes matched canonical sources.
- No historical local artifacts or original member repositories were modified or moved.

---

## 13. Local HPC-Readiness Preparation & Operational Pivot (November 11, 2026)

### 13.1 Operational Context & Constraint
`[RECORDED INVARIANT]`
- **PARAM SHAVAK Inaccessible**: Due to temporary infrastructure unavailability, remote SSH sessions, cluster commands, GPU benchmarks, and full-dataset transfers are paused.
- **Operational Strategy**: Complete the final, integrated, and scientifically validated source code across Weeks 1 to 5 locally, establish environment portability, and generate all deployment/SLURM scripts in advance of restored access.
- **HPC Reference Specification (Planning Only)**: Hostname `PS-BOSS-OS`, Linux OS, ~12 TB storage under `/home`, NVIDIA RTX 5000 Ada GPU (unverified live system parameters).

### 13.2 Centralized Portable Configuration
`[VERIFIED IN SOURCE CODE & TESTED]`
- Authored canonical `shared/config/settings.py`:
  - Dynamically resolves `PROJECT_ROOT`, `RESULTS_DIR`, `SPLITS_DIR`, `STATS_DIR`, `S1_DIR`, and `S2_DIR`.
  - Seamlessly toggles between local PC (`results/local_pc/`) and cluster mode (`param_shavak/results/`) via `SENTINEL_ENV` environment variable.
  - Enshrines canonical constants: `IMAGE_SIZE = (224, 224)`, `S1_CLIP_MIN = -25.0 dB`, `S1_CLIP_MAX = 0.0 dB`, `NUM_CLASSES = 19`, `EMBEDDING_DIM = 512`, `TEMPERATURE = 0.07`.
  - Authored `.env.example` at repository root.

### 13.3 Linux / HPC Cluster Environment Specifications
`[PREPARED & VERIFIED]`
- `shared/config/requirements_hpc.txt`: Explicit pip requirements with `faiss-gpu`, `rasterio`, and `torch>=2.2.0`.
- `shared/config/environment_hpc.yml`: Full Conda environment definition targeting CUDA 12.1 and GDAL on Linux x86_64.

### 13.4 Full-Dataset Automation & SLURM Workflow
`[PREPARED & STAGED IN param_shavak/scripts/]`
- `generate_full_dataset_splits_and_stats.py`: Automated pipeline to filter cloud patches, generate disjoint official splits (`train`, `val`, `test`), and accumulate global population mean/std strictly over the full training split.
- SLURM batch job scripts created for PARAM SHAVAK:
  1. `01_compute_full_dataset_stats.sh` (16 CPUs, 64 GB RAM, 12h)
  2. `02_train_two_tower_hpc.sh` (1 GPU RTX 5000 Ada, 16 CPUs, 64 GB RAM, 48h)
  3. `03_extract_embeddings_and_eval.sh` (1 GPU RTX 5000 Ada, 16 CPUs, 64 GB RAM, 12h)
  4. `run_all_hpc.sh`: Orchestrates automated job dependencies via `sbatch --dependency=afterok`.
- Pre-flight checklist established in `shared/docs/project_plans_and_audits/PARAM_SHAVAK_MIGRATION_CHECKLIST.md`.
