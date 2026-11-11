# CURRENT_PROJECT_STATUS.md — Living Status & Operational Task Log

> **DOCUMENT STATUS**: Living Status Board (Must be kept updated after every significant task)  
> **PROJECT**: SENTINEL-RS  
> **LAST UPDATED**: November 11, 2026

---

## 1. Project Phase & Overview

> [!WARNING]
> **OPERATIONAL CONSTRAINT (November 11, 2026)**:  
> **PARAM SHAVAK is currently temporarily inaccessible.** Remote SSH access, cluster execution, GPU tests, and full-dataset transfers cannot be performed at this stage.  
> **Operational Pivot**: All engineering effort is focused locally on finalizing, integrating, and scientifically validating the end-to-end source code (Weeks 1 to 5), centralized configuration, environment definitions, and SLURM execution scripts before cluster access is restored.


- **Active Development Phase**: **Phase 3 — PARAM SHAVAK HPC Preparation & Staging**
- **Current Active Milestone**: **Week 1 Reference Staged & Week 2 Verified Staged in param_shavak/**
- **Previous Steps Completed**: 
  - Step 1: Automated Static AST Dependency & Path Audit [VERIFIED COMPLETE]
  - Step 2 (Week 1): Week 1 Final Revalidation & Quarantined Patch Audit [VERIFIED COMPLETE]
  - Step 2 (Week 2): Week 2 Empirical Verification, Scientific Corrections & Live Real-Data Batch Validation [VERIFIED COMPLETE]
  - Step 2 (Final): Final Numerical Cross-Contributor Audit & Checkpoint Compatibility [VERIFIED COMPLETE]
  - Controlled Staging: Week 1 & Week 2 Staged to param_shavak/ with Provenance Checked [VERIFIED COMPLETE]
- **Next Pending Milestone**: **Phase A — Complete Week 1 and Week 2 HPC Preparation (Path Adaptation, Linux Environment, Full-Dataset Normalization)**
- **Overall Goal**: Consolidate four student member repositories, audit duplicate modules, fix known inference issues, and prepare a verified Linux execution environment for the PARAM SHAVAK HPC cluster without contaminating historical local PC results.

---

## 2. Completed Work Log

| Milestone / Deliverable | Status | Verification Summary |
| :--- | :---: | :--- |
| **Four-Repository Discovery & Scan** | [VERIFIED] | Audited 259 physical files across all 4 member repositories (cross-modal: 64, deep-cross: 74, majorProject: 81, majorproject_week3: 40). |
| **Model Checkpoint Discovery** | [VERIFIED] | Located canonical trained model est_model-v1.ckpt (598.95 MB) at shared/models/. |
| **Restructured Directory Creation** | [VERIFIED] | Created 65 skeleton directories across shared/, weeks/week_01..05, 
esults/local_pc/week_01..05, and param_shavak/. |
| **Deterministic Consolidation** | [VERIFIED] | Consolidated 184 unique repository files + 3 root docs + 2 checkpoint copies = **189 physical files** in SENTINEL-RS/. |
| **Cryptographic Hash Verification** | [VERIFIED] | Evaluated 37 candidate duplicate pairs using SHA-256; confirmed all 75 excluded files were redundant or boilerplate. |
| **Original Repositories Safeguard** | [VERIFIED] | All four original repositories remain 100% untouched and preserved as pristine backups. |
| **PARAM SHAVAK Segregation** | [VERIFIED] | Segregated cluster tree: param_shavak/results/ isolated and empty. |
| **Exclusion of experiment_120k** | [VERIFIED] | Confirmed zero occurrences of experiment_120k anywhere in SENTINEL-RS/. |
| **Permanent AI Knowledge Base** | [VERIFIED] | Authored AI_START_HERE.md, AGENT_HANDOFF.md, PROJECT_KNOWLEDGE_BASE.md, DECISIONS_AND_RATIONALE.md, RESEARCH_AND_AUDIT_LOG.md, and CURRENT_PROJECT_STATUS.md. |
| **Week 1 Final Revalidation** | [VERIFIED] | Verified metadata.parquet (480,038 patches), zero snow/cloud corruption, 0 overlap with 69,450 quarantined patches, 100% paired, 19 classes. |
| **Week 2 Empirical Verification** | [VERIFIED] | Inspected real GeoTIFFs: S1 Float32 in dB (clip $[-25, 0]\text{ dB}$ mandatory), S2 uint16 raw DN (no $/10000$ scaling). Verified 0 split leakage. |
| **Week 2 Source Corrections** | [VERIFIED] | Corrected all 15 Week 2 files across Adarsh, Jainam, Meet, Fateh. Backups & SHA256 hashes recorded in weeks/week_02/.backup_pre_correction/. |
| **Final Numerical Audit** | [VERIFIED] | Verified bitwise equivalence (PyTorch: Adarsh==Meet; PIL: Jainam==Fateh). Proved random init loss .3866 \approx \ln(4)$ uniform baseline. Strict loaded est_model-v1.ckpt (0 missing/unexpected keys) with positive cosine sim 0.93 and trained loss 0.000048. |
| **Controlled Staging to param_shavak/** | [VERIFIED] | Staged 7 Week 1 files as reference (AUDITED — HPC PATH ADAPTATION PENDING); staged 15 verified Week 2 files. Validated SHA-256 matches. |

---

## 3. Current Project Structure Breakdown

[VERIFIED] Physical filesystem inspection of C:\Users\hp\out\major project\SENTINEL-RS:

| Root Subdirectory | File Count | Size (MB) | Purpose & Contents |
| :--- | :---: | :---: | :--- |
| AI_START_HERE.md | 1 file | 0.003 MB | Universal AI entry point and reading directive |
| PROJECT_FILE_MAPPING.md| 1 file | 0.05 MB | Exhaustive file-by-file provenance table (189 files) |
| README.md | 1 file | 0.01 MB | Project overview, technical specifications, and user guide |
| shared/ | 28 files | 599.20 MB | Central docs, configurations, utilities, and est_model-v1.ckpt |
| weeks/ | 85 files | 0.68 MB | Canonical historical weekly source scripts (week_01/ to week_05/) |
| 
esults/local_pc/ | 65 files | 752.61 MB | Historical 30k local experiment outputs (splits, stats, embeddings, FAISS, metrics) |
| param_shavak/ | **24 files** | 0.15 MB | Staged cluster code (week_01/: 7 code + README; week_02/: 15 code + README) |
| param_shavak/results/ | **0 files** | 0.00 MB | Dedicated cluster output directory ensuring isolation from local PC results |
| 
eeds_classification/ | 8 files | 0.06 MB | Auxiliary data-copying, filesystem search, and deployment scripts |

---

## 4. Local-Subset vs. Complete-Dataset Invariant

> [!IMPORTANT]
> - **Local PC Subset**: The 30,000-patch dataset on the local PC was used strictly for developing, comparing contributor code, fixing bugs, and verifying model interfaces.
> - **Full Dataset on PARAM SHAVAK**: The final training, embedding extraction, and retrieval experiments will use the complete dataset available on the HPC cluster.
> - **Statistics & Splits Separation**: Local normalization statistics (s1_stats.json, s2_stats.json) are historical local-subset artifacts. Full-dataset statistics must be recomputed on the cluster using only the complete training split.

---

## 5. Next-Agent Execution Order

### Phase A — Complete Week 1 and Week 2 HPC Preparation
1. Resolve remaining source provenance questions and audit qualifications.
2. Adapt Week 1 and Week 2 paths/configuration for Linux/SLURM environments.
3. Inspect available PARAM SHAVAK hardware, CUDA version, and Python virtual environment.
4. Resolve and install compatible Linux dependencies (	orch, 	orchvision, 
asterio, pandas, 
aiss-gpu).
5. Connect to the complete dataset mount on PARAM SHAVAK.
6. Generate full-dataset splits and recompute training-only normalization statistics (param_shavak/results/).
7. Run multi-sample preprocessing and paired DataLoader tests on the cluster.
8. Verify checkpoint loading on GPU.

### Phase B — Week 3 Baseline Evaluation
1. Audit and prepare baseline embedding extraction on the full dataset.
2. Run baseline retrieval and mAP evaluation; store in param_shavak/results/week_03/.

### Phase C — Week 4 Model Training
1. Audit training script and hyperparameters.
2. Validate strict train/val separation on full dataset.
3. Train Two-Tower network on full training split.
4. Save new HPC checkpoints in param_shavak/results/week_04/ without overwriting historical local checkpoints.

### Phase D — Week 5 Retrieval & Deployment
1. Generate final full-dataset embeddings.
2. Build FAISS indices (param_shavak/results/week_05/).
3. Evaluate cross-modal and same-modal retrieval metrics.
4. Generate visualizations, latency benchmarks, and validate the Atlas Flask portal.

---

## 6. Key Files and Artifacts Quick Reference

| Artifact | Location | Description |
| :--- | :--- | :--- |
| **Master Checkpoint** | shared/models/best_model-v1.ckpt | 598.95 MB trained Two-Tower model |
| **Local Checkpoint (W4)** | 
esults/local_pc/week_04/checkpoints/best_model-v1.ckpt | Historical Week 4 local training output |
| **Parquet Metadata** | 
esults/local_pc/week_01/metadata/metadata.parquet | BigEarthNet v2.0 metadata (480,038 patches) |
| **Cloud Filter Parquet**| 
esults/local_pc/week_01/metadata/metadata_for_patches...parquet | 69,450 excluded cloudy patches |
| **Local Train Split CSV** | 
esults/local_pc/week_02/splits/train_split.csv | 21,000 paired local training patches |
| **Local Val Split CSV** | 
esults/local_pc/week_02/splits/val_split.csv | 4,500 paired local validation patches |
| **Local Test Split CSV** | 
esults/local_pc/week_02/splits/test_split.csv | 4,500 paired local test patches |
| **Local S1 Stats JSON** | 
esults/local_pc/week_02/normalization_stats/s1_stats.json | Local SAR VV/VH mean and std |
| **Local S2 Stats JSON** | 
esults/local_pc/week_02/normalization_stats/s2_stats.json | Local MS RGB mean and std |
| **Staged W1 Source** | param_shavak/week_01/ | 7 audited exploration scripts (HPC path adaptation pending) |
| **Staged W2 Source** | param_shavak/week_02/ | 15 verified preprocessing and DataLoader scripts |
| **HPC Results Output** | param_shavak/results/ | Isolated destination for all future HPC outputs |

---

## 7. Exact Starting Point for Next AI Agent

> **STARTING INSTRUCTION FOR NEXT AGENT**:  
> **Begin with Phase A, Step 1 & 2: Review shared/docs/agent_handoff/AGENT_HANDOFF.md, verify cluster path bindings in param_shavak/week_01/ and param_shavak/week_02/, and prepare the environment configuration before running jobs on the cluster.**
> 
> *Strict Constraints*:
> 1. Do NOT modify the original four repositories.
> 2. Do NOT overwrite historical local results in 
esults/local_pc/.
> 3. Do NOT substitute local 30k stats for full-dataset stats on the cluster.
> 4. Keep param_shavak/results/ cleanly segregated.