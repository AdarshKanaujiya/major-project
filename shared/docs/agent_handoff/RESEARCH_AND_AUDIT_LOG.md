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
| S2 Preprocessing | `preprocess_s2_meet.py` | Correct channel order, population stats, full validation |
| SAR Preprocessing | `sar_preprocessing_adarsh.py` | Clip + z-score + NaN guard; no broken imports |
| Paired Dataset | `paired_bigearthnet_dataset_meet.py` | No broken imports; correct preprocessing imports |
| DataLoader | `dataloader_meet.py` | No `week4` imports; canonical CSV paths |


### next step to solve issue from week2
refer this audit for week2 file **[`audit_report\week2_audit_report`](SENTINEL-RS\shared\docs\audit_report\week2_audit_report.md)**