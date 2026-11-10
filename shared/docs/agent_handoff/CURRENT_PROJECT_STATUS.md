# CURRENT_PROJECT_STATUS.md — Living Status & Operational Task Log

> **DOCUMENT STATUS**: Living Status Board (Must be kept updated after every significant task)  
> **PROJECT**: SENTINEL-RS  
> **LAST UPDATED**: November 10, 2026

---

## 1. Project Phase & Overview

- **Active Development Phase**: **Phase 2 — Codebase Understanding & Dependency Audit**
- **Current Active Milestone**: **Step 2 (Week 1 Audit Complete; Week 2 Pending Approval)**
- **Previous Steps Completed**: 
  - Step 1: Automated Static AST Dependency & Path Audit `[VERIFIED COMPLETE]`
  - Step 2 (Week 1): Week 1 Comparative Implementation Audit `[VERIFIED COMPLETE]`
- **Overall Goal**: Consolidate four student member repositories, audit duplicate modules, fix known inference issues, and prepare a verified Linux execution environment for the PARAM SHAVAK HPC cluster.

---

## 2. Completed Work Log

| Milestone / Deliverable | Status | Verification Summary |
| :--- | :---: | :--- |
| **Four-Repository Discovery & Scan** | `[VERIFIED]` | Audited 259 physical files across all 4 member repositories (`cross-modal`: 64, `deep-cross`: 74, `majorProject`: 81, `majorproject_week3`: 40). |
| **Model Checkpoint Discovery** | `[VERIFIED]` | Located canonical trained model `best_model-v1.ckpt` (598.95 MB) at `C:\Users\hp\out\micro project\final ye project\models\`. |
| **Restructured Directory Creation** | `[VERIFIED]` | Created 65 skeleton directories across `shared/`, `weeks/week_01..05`, `results/local_pc/week_01..05`, and `param_shavak/`. |
| **Deterministic Consolidation** | `[VERIFIED]` | Consolidated 184 unique repository files + 3 root docs + 2 checkpoint copies = **189 physical files** in `SENTINEL-RS/`. |
| **Cryptographic Hash Verification** | `[VERIFIED]` | Evaluated 37 candidate duplicate pairs using SHA-256; confirmed all 75 excluded files were redundant or boilerplate. |
| **Original Repositories Safeguard** | `[VERIFIED]` | All four original repositories remain 100% untouched and preserved as pristine backups. |
| **PARAM SHAVAK Segregation** | `[VERIFIED]` | Created mirrored cluster directory tree with `param_shavak/` code folders clean & empty (0 files) and `param_shavak/results/` clean & empty (0 files). |
| **Exclusion of `experiment_120k`** | `[VERIFIED]` | Confirmed zero occurrences of `experiment_120k` anywhere in `SENTINEL-RS/`. |
| **Permanent AI Knowledge Base** | `[VERIFIED]` | Authored `AI_START_HERE.md`, `AGENT_HANDOFF.md`, `PROJECT_KNOWLEDGE_BASE.md`, `DECISIONS_AND_RATIONALE.md`, `RESEARCH_AND_AUDIT_LOG.md`, and `CURRENT_PROJECT_STATUS.md`. |

---

## 3. Current Project Structure Breakdown

`[VERIFIED]` Physical filesystem inspection of `C:\Users\hp\out\major project\SENTINEL-RS`:

| Root Subdirectory | File Count | Size (MB) | Purpose & Contents |
| :--- | :---: | :---: | :--- |
| `AI_START_HERE.md` | 1 file | 0.003 MB | Universal AI entry point and reading directive |
| `PROJECT_FILE_MAPPING.md`| 1 file | 0.05 MB | Exhaustive file-by-file provenance table (189 files) |
| `README.md` | 1 file | 0.01 MB | Project overview, technical specifications, and user guide |
| `shared/` | 28 files | 599.20 MB | Central docs, configurations, utilities, and `best_model-v1.ckpt` |
| `weeks/` | 85 files | 0.68 MB | Weekly source scripts (`week_01/` to `week_05/`) organized by function |
| `results/local_pc/` | 65 files | 752.61 MB | Historical 30k experiment outputs (splits, stats, embeddings, FAISS, metrics) |
| `param_shavak/` | **0 files** | 0.00 MB | Staging skeleton for verified Linux cluster code (`week_01/` to `week_05/`) |
| `param_shavak/results/` | **0 files** | 0.00 MB | Dedicated cluster output directory ensuring isolation from local PC results |
| `needs_classification/` | 8 files | 0.06 MB | Auxiliary data-copying, filesystem search, and deployment scripts |

---

## 4. Pending Work & Phased Roadmap

- **Phase 2: Codebase Understanding & Dependency Audit** `[CURRENT ACTIVE PHASE]`
  - Statically trace imports and build an internal module dependency graph across `weeks/week_01/` to `week_05/`.
  - Catalog all hardcoded Windows paths (`C:\Users\hp\...`) needing abstraction.
  - Side-by-side comparison of duplicate implementations (`two_tower`, preprocessing, DataLoaders).
- **Phase 3: Canonical Module Selection & Refactoring Plan** `[PENDING]`
  - Select the official canonical implementation for each pipeline role.
  - Formulate a clean configuration strategy via `shared/config/settings.py` and `.env`.
- **Phase 4: Local 30k Pipeline Validation & Bug Fixing** `[PENDING]`
  - Fix single-band channel duplication in `weeks/week_05/retrieval/retriever_meet.py`.
  - Fix modality routing and subcategory form submission in `weeks/week_05/deployment/app_meet.py`.
  - Validate local query latency and Flask UI responsiveness.
- **Phase 5: PARAM SHAVAK Linux Compatibility Preparation** `[PENDING]`
  - Configure multi-worker DataLoaders (`num_workers=8/16`, `pin_memory=True`).
  - Stage tested, canonical Python scripts into `param_shavak/week_01/` through `week_05/`.
- **Phase 6: Cluster Execution on PARAM SHAVAK** `[PENDING]`
  - Run large-scale training and indexing on PARAM SHAVAK; route outputs strictly to `param_shavak/results/`.
- **Phase 7: Final Evaluation & Research Publication** `[PENDING]`
  - Generate final Recall@K, mAP, and t-SNE figures; draft final IEEE-format project paper.

---

## 5. Current Blockers & Technical Debt

1. **Hardcoded Windows Paths**:  
   Multiple scripts in `weeks/` import or load files using hardcoded Windows paths like `C:\Users\hp\out\...`. This will immediately break on Linux (PARAM SHAVAK).  
   *Remediation Plan*: Centralize all paths in `shared/config/settings.py` backed by `.env` variables in Phase 3/4.
2. **Inference Channel Duplication in `retriever_meet.py`**:  
   An existing fallback mechanism in `retriever_meet.py` silently duplicates a single channel into 2 or 3 channels when multi-band TIFFs are missing.  
   *Remediation Plan*: Implement genuine band resolution (B04/B03/B02 for Sentinel-2, VV/VH for Sentinel-1) with strict validation errors in Phase 4.
3. **No Active Cluster Queue Access on Local Machine**:  
   PARAM SHAVAK cluster access requires Linux remote execution / SSH. Cluster runs must be staged and validated before submission.

---

## 6. Key Files and Artifacts Quick Reference

| Artifact | Location | Description |
| :--- | :--- | :--- |
| **Model Checkpoint** | `shared/models/best_model-v1.ckpt` | 598.95 MB trained Two-Tower model |
| **Model Checkpoint (W4)** | `results/local_pc/week_04/checkpoints/best_model-v1.ckpt` | Week 4 training output artifact |
| **Parquet Metadata** | `results/local_pc/week_01/metadata/metadata.parquet` | BigEarthNet v2.0 metadata (3.45 MB) |
| **Cloud Filter Parquet**| `results/local_pc/week_01/metadata/metadata_for_patches...parquet` | 69,450 excluded cloudy patches |
| **Training Split CSV** | `results/local_pc/week_02/splits/train_split.csv` | 21,000 paired training patches |
| **Validation Split CSV**| `results/local_pc/week_02/splits/val_split.csv` | 4,500 paired validation patches |
| **Test Split CSV** | `results/local_pc/week_02/splits/test_split.csv` | 4,500 paired test patches (gallery) |
| **S1 Stats JSON** | `results/local_pc/week_02/normalization_stats/s1_stats.json` | Sentinel-1 SAR VV/VH mean and std |
| **S2 Stats JSON** | `results/local_pc/week_02/normalization_stats/s2_stats.json` | Sentinel-2 MS RGB mean and std |
| **MS Test Embeddings** | `results/local_pc/week_05/trained_embeddings/trained_test_ms_embeddings.npy`| 4500 x 512 L2-normalized vectors |
| **SAR Test Embeddings**| `results/local_pc/week_05/trained_embeddings/trained_test_sar_embeddings.npy`| 4500 x 512 L2-normalized vectors |
| **MS FAISS Index** | `results/local_pc/week_05/faiss_indices/ms_gallery.index` | 9.2 MB FAISS IndexFlatIP gallery |
| **SAR FAISS Index** | `results/local_pc/week_05/faiss_indices/sar_gallery.index` | 9.2 MB FAISS IndexFlatIP gallery |
| **Atlas Flask Web App**| `weeks/week_05/deployment/app_meet.py` | Full-stack interactive retrieval portal |

---

## 7. Exact Next Task for the Incoming AI Agent

> **IMMEDIATE TASK FOR NEXT AGENT**:  
> **Perform a read-only, week-by-week audit of the existing codebase under `weeks/`, building a dependency map and identifying duplicate implementations before making any code changes or staging files to `param_shavak/`.**
> 
> *Guidelines*:
> 1. Do not modify or delete any existing source files.
> 2. Do not run training or preprocessing scripts.
> 3. Document all import chains, hardcoded path variables, and behavioral differences across members in `shared/docs/agent_handoff/RESEARCH_AND_AUDIT_LOG.md`.
> 4. Keep `param_shavak/` code directories empty until this audit is complete and approved by the user.

