# Phase 2 - Step 1: Static Code Dependency & Path Audit Report

> **AUDIT TYPE**: Read-Only Abstract Syntax Tree (AST) Analysis  
> **SCOPE**: `weeks/week_01/` through `weeks/week_05/`  
> **EXECUTION STATUS**: 100% Non-destructive (Zero project scripts executed or modified)  
> **DATE**: November 2026  

---

## 1. Executive Summary & File Accounting

| Audit Metric | Count | Verification Notes |
| :--- | :---: | :--- |
| **Total Files in `weeks/`** | **86** | Discovered across `week_01` through `week_05` |
| **Python Scripts (`.py`)** | **81** | Expected core scripts across 5 milestones |
| **Successfully Parsed** | **81** | 100% AST parse rate without syntax errors |
| **Parse Failures / Syntax Errors** | **0** | Zero broken Python syntax detected |
| **Jupyter Notebooks (`.ipynb`)** | **1** | `week_01/data_exploration/dataset_exploration_meet.ipynb` |
| **Non-Code Web Files** | **4** | 3 HTML templates (`domain`, `search`, `subcategory`) + 1 CSS stylesheet |
| **Runnable Entry Points (`__main__`)** | **54** | Scripts with standalone execution guards |
| **Pure Library / Module Files** | **27** | Model definitions, preprocessing classes, dataset modules |

### Explanation of Expected File Counts
- The initial estimate was **85 files**.
- Physical disk scan discovered **86 files** in `weeks/`.
- **Exact Breakdown**: 81 `.py` Python scripts, 1 `.ipynb` Jupyter notebook, and 4 web assets (3 HTML templates and 1 CSS file in `weeks/week_05/deployment/`).
- All 81 `.py` files were parsed and audited successfully.

---

## 2. Dependency Inventory by Category

### A. Third-Party Dependencies (HPC Environment Requirements)
These libraries must be installed on the PARAM SHAVAK Python virtual environment/container:

| Library | Category | Typical Usage in Codebase |
| :--- | :--- | :--- |
| `PIL` | Core Dependency | Image thumbnail conversion and visualization |
| `faiss` | Core Dependency | IndexFlatIP, fast vector similarity search |
| `flask` | Core Dependency | Atlas web portal server and routing |
| `lightning` | Core Dependency | General |
| `matplotlib` | Core Dependency | Training loss curves, t-SNE scatter plots |
| `numpy` | Core Dependency | Embedding arrays, matrix math, metrics |
| `pandas` | Core Dependency | CSV dataset splits, Parquet tables, results tables |
| `pyarrow` | Core Dependency | Parquet columnar dataset reading |
| `rasterio` | Core Dependency | GeoTIFF satellite band decoding, metadata extraction |
| `sklearn` | Core Dependency | t-SNE dimensionality reduction, train_test_split |
| `torch` | Core Dependency | PyTorch tensor operations, DataLoader, neural layers |
| `torchvision` | Core Dependency | Pretrained ResNet-50 backbone, transforms |
| `tqdm` | Core Dependency | Epoch and batch progress bars |
| `werkzeug` | Core Dependency | General |

### B. Standard Library Dependencies
Total unique standard library modules detected: **22**

```
argparse, ast, collections, concurrent, csv, datetime, functools, inspect, io, json, logging, os, pathlib, random, re, shutil, sys, time, typing, unittest, warnings, zipfile
```

### C. Internal Project Package Imports (Flagged for Path / Environment Dependency)
The following internal modules are referenced via top-level imports that expect their parent directory to be in `PYTHONPATH` or the current working directory:

| Internal Package | Occurrences / Context | Risk on Linux / PARAM SHAVAK |
| :--- | :--- | :--- |
| `config` | Referenced as top-level module (e.g. `from config...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `dataset` | Referenced as top-level module (e.g. `from dataset...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `evaluation` | Referenced as top-level module (e.g. `from evaluation...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `models` | Referenced as top-level module (e.g. `from models...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `preprocessing` | Referenced as top-level module (e.g. `from preprocessing...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `src` | Referenced as top-level module (e.g. `from src...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `utils` | Referenced as top-level module (e.g. `from utils...`) | Will fail with `ModuleNotFoundError` if run from different working directory unless `PYTHONPATH` is configured. |
| `__future__` | Local module or script-level import | Must be resolved to `shared/` or relative path. |
| `app` | Local module or script-level import | Must be resolved to `shared/` or relative path. |
| `inspect_dataset` | Local module or script-level import | Must be resolved to `shared/` or relative path. |
| `preprocess_s2` | Local module or script-level import | Must be resolved to `shared/` or relative path. |
| `psutil` | Local module or script-level import | Must be resolved to `shared/` or relative path. |
| `tests` | Local module or script-level import | Must be resolved to `shared/` or relative path. |
| `week4` | Local module or script-level import | Must be resolved to `shared/` or relative path. |

---

## 3. Path Anomalies & Hardcoded Filesystem Strings

| Path Anomaly Category | Total Detected Occurrences | Impact on PARAM SHAVAK (Linux) |
| :--- | :---: | :--- |
| **Absolute Windows Paths (`C:\...`)** | **20** | **CRITICAL BREAKING**: Fails immediately on Linux (`FileNotFoundError`). |
| **User-Specific Paths (`hp`, `Users`)** | **0** | **CRITICAL BREAKING**: Hardcoded to user profile `hp`. |
| **Dataset & Checkpoint File Literals** | **385** | **HIGH**: Paths pointing to local filenames (`s1_stats.json`, `.npy`, `.ckpt`). |
| **Working-Directory-Dependent Rel Paths** | **105** | **MEDIUM**: Breaks if script is invoked from a different working directory. |

### Script-by-Script Path Anomaly Registry (Sample Critical Instances)

| Script Path in `weeks/` | Line # | String Literal Detected | Category |
| :--- | :---: | :--- | :--- |
| `week_01\data_exploration\explore_metadata_jainam.py` | 8 | `D:\Dataset\metadata.parquet` | `absolute_windows_path` |
| `week_01\inspection\view_random_images_jainam.py` | 12 | `D:\Dataset` | `absolute_windows_path` |
| `week_02\dataloader\test_paired_batch_jainam.py` | 32 | `C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required` | `absolute_windows_path` |
| `week_02\dataloader\test_paired_batch_jainam.py` | 33 | `C:\BigEarthNet-S2-Required` | `absolute_windows_path` |
| `week_02\dataset_preparation\verify_pairs_jainam.py` | 10 | `D:/Dataset/metadata.parquet` | `absolute_windows_path` |
| `week_02\preprocessing\compute_stats_adarsh.py` | 16 | `C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2` | `absolute_windows_path` |
| `week_02\preprocessing\preprocess_s2_adarsh.py` | 18 | `C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2` | `absolute_windows_path` |
| `week_02\preprocessing\preprocess_s2_adarsh.py` | 224 | `C:\Users\hp\out\micro project\final...3T101031_N9999_R022_T33UUP_29_55` | `absolute_windows_path` |
| `week_02\preprocessing\preprocesser_adarsh.py` | 7 | `C:\Users\hp\out\micro project\final...3T101031_N9999_R022_T33UUP_29_55` | `absolute_windows_path` |
| `week_02\preprocessing\test_preprocessing_adarsh.py` | 11 | `C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2` | `absolute_windows_path` |
| `week_02\preprocessing\test_sar_preprocessing_jainam.py` | 24 | `D:\Dataset` | `absolute_windows_path` |
| `week_02\preprocessing\visualize_preprocessing_adarsh.py` | 19 | `C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2` | `absolute_windows_path` |
| `week_04\training\train_adarsh.py` | 46 | `C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required` | `absolute_windows_path` |
| `week_04\training\train_adarsh.py` | 50 | `C:\BigEarthNet-S2-Required` | `absolute_windows_path` |
| `week_04\training\train_jainam.py` | 46 | `C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required` | `absolute_windows_path` |
| `week_04\training\train_jainam.py` | 50 | `C:\BigEarthNet-S2-Required` | `absolute_windows_path` |
| `week_05\embeddings\extract_test_embeddings_jainam.py` | 28 | `C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required` | `absolute_windows_path` |
| `week_05\embeddings\extract_test_embeddings_jainam.py` | 32 | `C:\BigEarthNet-S2-Required` | `absolute_windows_path` |
| `week_05\embeddings\generate_trained_embeddings_adarsh.py` | 18 | `C:\Users\hp\out\micro project\final...\dataset\BigEarthNet-S1-Required` | `absolute_windows_path` |
| `week_05\embeddings\generate_trained_embeddings_adarsh.py` | 19 | `C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2` | `absolute_windows_path` |

*(See companion machine-readable file `phase2_dependency_and_path_audit.json` for all flagged instances with exact line numbers.)*

---

## 4. Entry-Point Inventory (`if __name__ == '__main__':`)

A total of **54 scripts** contain standalone execution guards (`__main__`), while **27 scripts** are purely library modules.

### Milestone Breakdown of Standalone Runnable Scripts:

| Week Milestone | Runnable Entry Points (`__main__`) | Pure Library Modules | Total Scripts |
| :--- | :---: | :---: | :---: |
| **week_01** | 2 | 5 | 7 |
| **week_02** | 21 | 7 | 28 |
| **week_03** | 8 | 7 | 15 |
| **week_04** | 5 | 5 | 10 |
| **week_05** | 18 | 3 | 21 |

---

## 5. Non-Destructive Audit Constraints Verification

- `[VERIFIED]` **Zero Code Execution**: All information was gathered strictly via `ast.parse` static syntax analysis. No project Python scripts or Jupyter code cells were executed.
- `[VERIFIED]` **Zero File Modifications**: No source code, outputs, or existing documentation files were altered, moved, or deleted.
- `[VERIFIED]` **PARAM SHAVAK Integrity**: The `param_shavak/` directory remains clean, segregated, and completely empty.
- `[VERIFIED]` **Audit Outputs Segregated**: The audit source code and outputs are stored exclusively in `shared/utilities/` and `shared/docs/`.

---
*Phase 2 - Step 1 Completed. Awaiting approval to proceed to Step 2 (Week-by-Week Comparative Audit & Canonical Selection).*