# DECISIONS_AND_RATIONALE.md — Architectural Invariants & Decision Registry

> **DOCUMENT STATUS**: Permanent Architectural Registry  
> **PROJECT**: SENTINEL-RS  
> **LAST UPDATED**: November 2026

---

## 1. Executive Summary & Purpose

This document serves as the permanent record of architectural decisions, file reconciliations, user questions and verified answers, and technical invariants established during the development and consolidation of the **SENTINEL-RS** codebase. Future AI agents and engineers must treat the final decisions recorded here as established facts and must not reopen settled architectural choices without explicit user direction.

---

## 2. Consolidation & Environmental Architecture Decisions

### Decision 2.1: Consolidation of Four Disparate Repositories into `SENTINEL-RS`
- **Context**: The project was independently developed by four students across four separate Git repositories (`cross-modal-remote-sensing-retrieval`, `deep-cross-final-yr-project`, `majorProject`, and `majorproject_week3`). Each repository held fragments of the pipeline, duplicated scripts, mismatched local paths, and conflicting dependencies.
- **Decision**: Merge all four codebases into a single unified repository at `SENTINEL-RS/`, organized strictly by the five-week research development roadmap (`weeks/week_01` to `week_05`).
- **Rationale**: A unified structure provides a single source of truth, eliminates cross-repo path dependencies, allows systemic code auditing, and makes deployment to the PARAM SHAVAK HPC cluster straightforward.
- **Status**: `[FINAL / VERIFIED]`

### Decision 2.2: Pristine Preservation of the Original Four Repositories
- **Context**: Merging code risk-evaluates overwriting individual member contributions, corrupting existing working prototypes, or losing experimental logs.
- **Decision**: The four original repositories at `C:\Users\hp\out\major project\` are treated as read-only backups. All consolidation actions were executed via read-and-copy operations (`shutil.copy2`). No original files were moved, renamed, or deleted.
- **Rationale**: Preserves 100% data provenance, provides an immediate rollback mechanism, and ensures student course-submission records remain untouched.
- **Status**: `[FINAL / VERIFIED]`

### Decision 2.3: Structural Separation between Source Code (`weeks/`) and Results (`results/local_pc/`)
- **Context**: In several member repos, output files (`.npy`, `.parquet`, `.png`, `.csv`, `.index`) were scattered inside source code directories (`scripts/`, `preprocessing/`, `src/`).
- **Decision**: Strictly separate executable logic from data artifacts:
  - Source code scripts and notebooks belong in `weeks/week_XX/<category>/`.
  - Generated datasets, splits, stats, embeddings, checkpoints, metrics, and plots belong in `results/local_pc/week_XX/<category>/`.
- **Rationale**: Standard MLOps practice. Prevents code bloat, simplifies version control, prevents accidental overwrites of expensive training outputs, and clarifies data lineage.
- **Status**: `[FINAL / VERIFIED]`

### Decision 2.4: Total Isolation between Local PC and PARAM SHAVAK Environments
- **Context**: Initial local experiments used a 30,000-patch subset on a local Windows PC. Subsequent experiments will run on PARAM SHAVAK, a high-performance Linux computing cluster with multi-GPU accelerators, potentially processing 120,000+ patches.
- **Decision**: Create an independent, mirrored working directory `param_shavak/` with its own `param_shavak/results/` hierarchy. Outputs from PARAM SHAVAK runs must write **exclusively** to `param_shavak/results/` and never touch `results/local_pc/`.
- **Rationale**: Cluster-scale outputs must never overwrite, contaminate, or be confused with the historical 30k baseline benchmark results.
- **Status**: `[FINAL / VERIFIED]`

### Decision 2.5: No Hardcoded Experiment Subfolders (e.g., `experiment_120k`)
- **Context**: A prior proposal suggested creating a dedicated `param_shavak/experiment_120k/` directory.
- **Decision**: Do **not** create `experiment_120k` or any experiment-size-specific subfolder. Keep `param_shavak/` organized purely by development milestone (`week_01/` to `week_05/`) and `param_shavak/results/` (`week_01/` to `week_05/`).
- **Rationale**: The dataset size for cluster training is dynamic and will be determined at runtime based on allocated GPU memory and cluster queue limits. Hardcoding dataset scales into directory names creates rigid, confusing structures.
- **Status**: `[FINAL / VERIFIED]`

### Decision 2.6: Empty Staging in `param_shavak/` until Code is Audited
- **Context**: The four member repos contain Windows-specific path strings (`C:\Users\hp\...`), Windows-specific spawn settings (`num_workers=0`), and duplicated modules.
- **Decision**: Keep `param_shavak/week_01` through `week_05` completely empty (0 files) for now. Code will only be staged into `param_shavak/` after being audited, canonicalized, and converted to use Linux environment variables.
- **Rationale**: Staging unverified or broken code to the HPC directory violates reproducibility and risks failed batch cluster submissions.
- **Status**: `[FINAL / VERIFIED]`

---

## 3. Definitive File Count Reconciliation & Proof

### 3.1 The Reconciliation Equations
During consolidation and verification, exact file accounting was established across physical disk files:

$$\mathbf{259} \text{ (Physical files in 4 original member repos)}$$
$$- \mathbf{75} \text{ (Excluded duplicate, boilerplate, or placeholder files)}$$
$$= \mathbf{184} \text{ (Unique files consolidated from original repos)}$$

$$\mathbf{184} \text{ (Consolidated repo sources)}$$
$$+ \mathbf{3} \text{ (Project root documents: Plan, Audit, Mapping)}$$
$$+ \mathbf{2} \text{ (Dual destination copies of trained model checkpoint)}$$
$$= \mathbf{189} \text{ (Total consolidated files on disk in SENTINEL-RS)}$$

### 3.2 Breakdown of the 75 Excluded Files
1. **Git Metadata & Placeholders (11 files)**: `.gitignore` and `.gitkeep` files from individual repos.
2. **Python Package Initializers (18 files)**: Empty or boilerplate `__init__.py` files.
3. **Empty Zero-Byte Documentation Stubs (2 files)**: `docs/PROJECT_LOG.md` (0 bytes) and `docs/references.md` (0 bytes).
4. **Duplicate Browser Downloads (3 files)**: Accidental download copies with `(1)` in filename: `s2_stats (1).json`, `baseline_val_ms_embeddings (1).npy`, `val_split (1).csv`.
5. **Redundant Cross-Member Inputs (8 files)**: `majorProject/teammate_inputs/` holding duplicate copies of `train_split.csv`, `test_split.csv`, `baseline_val_ms_embeddings.npy`, etc.
6. **Identical Duplicate Data Splits & Stats (12 files)**: Copies of `train_split.csv`, `val_split.csv`, `test_split.csv`, `s1_stats.json`, `s2_stats.json`, `metrics.csv`, and `training_loss.png` shared across teammates. Canonical versions are in `results/local_pc/`.
7. **Identical Duplicate Text Reports (8 files)**: Exact copies of `baseline_ms_to_ms_retrieval_results.txt`, `baseline_results_summary.txt`, etc.
8. **Duplicate Embeddings & Patch IDs (6 files)**: Redundant copies of `baseline_val_ms_embeddings.npy` and patch ID txt files.
9. **Duplicate Configs (2 files)**: Duplicate copies of `requirements.txt` and `disaster_modality_config.json`.
10. **Superseded Early-Draft Web Templates (4 files)**: Fateh's early prototype HTML/CSS from Week 3, superseded by Meet's full Atlas portal in `weeks/week_05/deployment/`.
11. **Duplicate Training Log (1 file)**: `Week_4_training_record_from_jainam.txt` in Adarsh's repo, which is an exact copy of Jainam's log.

---

## 4. Specific Technical Decisions

### Decision 4.1: Placement of Metadata Parquet Files
- **Decision**: Store `metadata.parquet` (3.45 MB) and `metadata_for_patches_with_snow_cloud_or_shadow.parquet` (0.68 MB) in `results/local_pc/week_01/metadata/`.
- **Rationale**: Parquet files are generated data tables containing precomputed BigEarthNet labels and cloud-exclusion masks, not executable code. Placing them under `results/` follows the code-data separation rule. The scripts that read them (`explore_metadata_adarsh.py`, `filter_clean_data_adarsh.py`) reside in `weeks/week_01/data_exploration/`.
- **Status**: `[FINAL / VERIFIED]`

### Decision 4.2: Preservation of Dual Checkpoint Copies
- **Decision**: Keep `best_model-v1.ckpt` (598.95 MB) in both `shared/models/best_model-v1.ckpt` and `results/local_pc/week_04/checkpoints/best_model-v1.ckpt`.
- **Rationale**: `shared/models/` acts as the shared model registry for live inference (used by the Flask Atlas portal), while `results/local_pc/week_04/checkpoints/` preserves the checkpoint as the primary training artifact of the Week 4 training milestone. Both copies were verified byte-for-byte identical.
- **Status**: `[FINAL / VERIFIED]`

### Decision 4.3: Retention of Auxiliary Files in `needs_classification/`
- **Decision**: Do not move the 8 auxiliary scripts (`find.py`, `check_required_s2.py`, `copy_required_s2.py`, `build_s1_subset.py`, `test_pipeline.py`, `Procfile`, `render.yaml`, `output.txt`) into main pipeline folders yet. Keep them in `needs_classification/`.
- **Rationale**: These are operational file-copying, testing, and deployment helpers. Keeping them isolated prevents cluttering the 5-week research timeline while keeping them accessible.
- **Status**: `[FINAL / VERIFIED]`

---

## 5. Previous User Questions & Established Answers

### Q1: Are all files placed in the `SENTINEL-RS` folder?
- **Established Answer**: **Yes.** All 184 unique source files from the four member repositories, plus 3 root documentation files and the master model checkpoint, are present inside `SENTINEL-RS/`.

### Q2: Are any files pending classification or proper placement?
- **Established Answer**: **No.** All 189 assets are classified and placed in their designated directory according to the 5-week plan and functional role. The 8 auxiliary scripts reside in `needs_classification/`, and `param_shavak/` is intentionally kept empty of unverified code.

### Q3: What is `execute_consolidation.py`?
- **Established Answer**: It is an automated, idempotent consolidation script that executed the physical copy of all 189 files with metadata preservation, verified existence on disk, and generated `PROJECT_FILE_MAPPING.md` and `README.md`. It strictly copies and contains zero deletion or modification commands.

### Q4: Why did the four original repositories contain 259 files while 184 unique files were consolidated?
- **Established Answer**: The 259 original files contained 75 files that were identical duplicates across teammate repositories, Python package boilerplate (`__init__.py`), Git metadata (`.gitignore`, `.gitkeep`), or zero-byte stubs. Exactly $259 - 75 = 184$ unique physical repository files were consolidated.

### Q5: What does the Parquet placement and provenance instruction mean?
- **Established Answer**: It means keeping executable code (`.py`) under `weeks/` and data artifacts (`.parquet`, `.csv`, `.npy`) under `results/local_pc/`, while documenting the author, original path, and development week of every file in `PROJECT_FILE_MAPPING.md`.

### Q6: Why are local PC and PARAM SHAVAK environments separate?
- **Established Answer**: To prevent data contamination. The local PC outputs reflect the historical 30,000-patch baseline experiment. PARAM SHAVAK is a multi-GPU Linux HPC cluster that will execute larger-scale experiments. Keeping `param_shavak/results/` separate ensures cluster outputs never overwrite or mix with local baseline results.

### Q7: Why should code be audited before migration to PARAM SHAVAK?
- **Established Answer**: The local PC scripts contain hardcoded Windows file paths (`C:\Users\hp\...`), Windows-specific DataLoader multiprocessing configurations (`num_workers=0`), and duplicated modules across team members. Auditing selects the single best implementation for each task and converts paths to use environment variables (`.env`), preventing cluster job failures.

### Q8: Where do the retrieved images and text labels in the Atlas portal come from?
- **Established Answer**: They come directly from the local test set directory (`test_split.csv`, 4,500 patches) and the precomputed FAISS indices (`ms_gallery.index`, `sar_gallery.index`). They do not come from Google or the live internet. The land-cover text labels (e.g. *Coniferous forest*, *Arable land*) are ground-truth Corine Land Cover labels from the BigEarthNet dataset.

### Q9: What happens when a user uploads a satellite image for retrieval in the portal?
- **Established Answer**:
  1. The user selects a GeoTIFF image and chooses the modality (SAR or Multispectral).
  2. Flask validates the file format and saves it to a temporary directory.
  3. Preprocessing converts the image into a standardized tensor: `(2, 224, 224)` for SAR (with dB clipping `[-25, 0]`) or `(3, 224, 224)` for MS (with `s2_stats.json` standardization).
  4. The corresponding encoder tower from `TwoTowerNetwork` produces a 2048-D representation, which the projection head maps to a 512-D L2-normalized vector.
  5. FAISS performs an Inner Product search (`IndexFlatIP`) against the gallery in ~1.2 ms, returning the top-10 nearest patch IDs and cosine similarity scores.
  6. The portal renders the top-10 retrieved scenes alongside similarity scores and land-cover labels.

