# AGENT_HANDOFF.md — AI Agent Operating Manual & Handoff Directive

> **DOCUMENT PURPOSE**: This file is the primary handoff directive for any incoming AI agent or software engineer taking over work on the **SENTINEL-RS** project. It outlines the current state, strict environmental and data invariants, non-negotiable safety rules, the phased development roadmap, and the exact next task to execute.

---

## 1. Project Identity, Context & Mission

> [!WARNING]
> **TEMPORARY CLUSTER INACCESSIBILITY**:  
> PARAM SHAVAK is currently unavailable via SSH. Do NOT attempt SSH logins, remote jobs, or GPU executions. Prioritize completing the end-to-end integrated source code (Weeks 1 to 5), centralized configuration, and deployment scripts locally on Windows/CPU first.


### 1.1 What SENTINEL-RS Is
**SENTINEL-RS** is an applied machine learning and remote sensing retrieval research project developed by four undergraduate researchers. It tackles the **modality gap** between Synthetic Aperture Radar (SAR, Sentinel-1) and Optical/Multispectral imagery (MS, Sentinel-2). 
- In remote sensing, SAR operates through clouds, precipitation, and darkness by transmitting microwave radar pulses, capturing physical surface geometry, roughness, and dielectric properties.
- Multispectral imaging captures optical reflectance across distinct spectral bands (Red, Green, Blue, NIR, SWIR), providing rich visual, vegetative, and land-use information, but fails completely under cloud cover or night conditions.
- **The Core Objective**: Learn a shared, sensor-agnostic 512-dimensional metric embedding space where geographically and semantically matching SAR and Multispectral scenes map close together, enabling real-time cross-modal search:
  $$\text{Query: Sentinel-1 SAR Image} \longrightarrow \text{Retrieve: Semantically Corresponding Sentinel-2 MS Scenes}$$
  $$\text{Query: Sentinel-2 MS Image} \longrightarrow \text{Retrieve: Semantically Corresponding Sentinel-1 SAR Scenes}$$

### 1.2 What the User Is Trying to Achieve
The user has two goals:
1. **Consolidate, organize, and maintain** the code developed on a local Windows PC workstation across four team members into a clean, unified project repository following a 5-week research roadmap.
2. **Prepare a clean, verified execution environment** for **PARAM SHAVAK**, a college-hosted Linux High-Performance Computing (HPC) cluster with multi-GPU accelerators, where larger-scale experiments will be executed without contaminating or mixing with historical local PC results.

### 1.3 What Stage the Project Is Currently In
- **Phase 1 (Consolidation & Provenance Tracking)**: `[VERIFIED COMPLETE]`  
  All 4 member codebases have been audited and merged into `SENTINEL-RS/`. Historical 30k experimental outputs are safeguarded in `results/local_pc/`.
- **Phase 2 (Codebase Audits, Scientific Corrections & Local Validation)**: `[COMPLETE ✅]`  
  - **Week 1**: Audited and validated (metadata, zero leakage, 19 classes). `[COMPLETE]`
  - **Week 2**: Audited, 15 files corrected, numerical consistency and strict checkpoint compatibility verified. `[COMPLETE]`
  - **Week 3**: Audited, scientific clarification completed, 6 approved source corrections applied, 12-point validation suite passed. `[COMPLETE]`
  - **Week 4**: Two-Tower architecture audit, InfoNCE training, BatchNorm/Adam/AdamW decoupling, checkpoint saving — all implemented and validated. `[COMPLETE]`
  - **Week 5**: Phase A audit, Phase A.1 scientific reconciliation, Phase B controlled implementation — 22 files corrected, 0 compile errors, metric relabelling, identifier mapping, L2 embeddings, timing benchmark redesigned. `[COMPLETE]`

---

## 2. Current Verified State of the Project

The incoming AI agent can rely on the following verified facts:

| Component | Verified State | Verification Details |
| :--- | :---: | :--- |
| **Consolidation** | `[VERIFIED]` | 184 unique repo files + 3 root docs + 2 checkpoint copies = **189 destination files** in `SENTINEL-RS/`. |
| **Original Repositories** | `[VERIFIED]` | All 4 original member repositories are 100% untouched (`cross-modal`: 64, `deep-cross`: 74, `majorProject`: 81, `majorproject_week3`: 40 files). |
| **Local PC Results** | `[VERIFIED]` | All 30k local experiment outputs (splits, stats, embeddings, FAISS indices, evaluation CSVs, loss plots) reside safely in `results/local_pc/`. |
| **PARAM SHAVAK Code Folders** | `[VERIFIED]` | `param_shavak/week_01/` to `week_05/` exist as folder skeletons and are **completely empty (0 files)**. |
| **PARAM SHAVAK Results Folder** | `[VERIFIED]` | `param_shavak/results/` exists and is **completely empty (0 files)**. |
| **No 120k Folder** | `[VERIFIED]` | No `experiment_120k` directory exists. The cluster experiment scale will be determined at runtime. |
| **Trained Checkpoint** | `[VERIFIED]` | `best_model-v1.ckpt` (598.95 MB) is preserved in `shared/models/` and `results/local_pc/week_04/checkpoints/`. |
| **Parquet Files** | `[VERIFIED]` | `metadata.parquet` (3.45 MB) and `metadata_for_patches_with_snow_cloud_or_shadow.parquet` (0.68 MB) are in `results/local_pc/week_01/metadata/`. |
| **Auxiliary Utilities** | `[VERIFIED]` | 8 utility scripts are safely preserved in `needs_classification/`. |

---

## 3. Mandatory Directory Architecture

The repository enforces strict separation of concerns across five primary branches:

```
SENTINEL-RS/
├── shared/                                 # Centralized project-wide assets
│   ├── config/                             # Requirements, YAML configs, settings, disaster mappings
│   ├── docs/                               # Permanent AI Knowledge Base, audits, roadmaps, handoffs
│   ├── models/                             # Master trained model checkpoint (best_model-v1.ckpt)
│   └── utilities/                          # Shared I/O, configuration parsers, and visual helpers
│
├── weeks/                                  # Source code scripts organized by weekly milestone
│   ├── week_01/                            # Data exploration & inspection scripts (Adarsh, Jainam, Meet)
│   ├── week_02/                            # Preprocessing, 30k splits & paired DataLoader implementations
│   ├── week_03/                            # Baseline feature extractors, mAP engines & baseline t-SNE
│   ├── week_04/                            # Two-Tower architectures, InfoNCE loss & training loops
│   └── week_05/                            # Trained embeddings, FAISS galleries, Atlas Flask portal
│
├── results/                                # Output artifacts from experiments
│   └── local_pc/                           # Historical outputs from the 30k local PC run
│       ├── week_01/metadata/               # BigEarthNet Parquets, filtering and inspection reports
│       ├── week_02/normalization_stats/    # s1_stats.json, s2_stats.json, statistics.json
│       ├── week_02/splits/                 # train_split.csv, val_split.csv, test_split.csv (30k)
│       ├── week_03/baseline_embeddings/    # baseline_val_ms/s1_embeddings.npy (4500 x 2048)
│       ├── week_03/evaluation/             # Baseline Recall@K reports, baseline mAP results
│       ├── week_03/visualization/          # baseline_tsne.png, label_distribution.png
│       ├── week_04/checkpoints/            # best_model-v1.ckpt (598.95 MB), hparams.yaml
│       ├── week_04/metrics/                # metrics.csv, Week_4_training_record.txt
│       ├── week_04/visualization/          # training_loss.png, training_loss_by_ad.png
│       ├── week_05/trained_embeddings/     # trained_test_ms/sar_embeddings.npy (4500 x 512)
│       ├── week_05/faiss_indices/          # ms_gallery.index, sar_gallery.index (FAISS IndexFlatIP)
│       ├── week_05/evaluation/             # trained_cross_modal_results.csv, final comparison reports
│       ├── week_05/visualization/          # visual_retrieval_comparison.png, success/failure plots
│       └── week_05/benchmarking/           # timing_results.json (query latency profile)
│
├── param_shavak/                           # PARAM SHAVAK HPC Cluster Environment (Segregated)
│   ├── week_01/ to week_05/                # Target folders for verified, Linux-compatible cluster scripts
│   └── results/                            # Dedicated cluster results (week_01/ to week_05/)
│
├── needs_classification/                   # Specialized data copy, search, and deployment scripts
├── PROJECT_FILE_MAPPING.md                 # Complete provenance mapping of all 189 files
└── README.md                               # User-facing project overview and technical documentation
```

---

## 4. Eleven Non-Negotiable Rules for Any AI Agent

As an AI assistant working on this codebase, you must adhere strictly to these principles:

1. **Understand and Audit First**:  
   Always inspect existing code, comments, and documentation before suggesting changes. Do not assume code is broken until you have statically traced or tested it.
2. **Never Rewrite the Project from Scratch**:  
   Do not replace the architecture, throw away existing pipelines, or introduce massive third-party frameworks unless explicitly asked by the user.
3. **Preserve Historical Artifacts**:  
   Under no circumstances delete, overwrite, or mutate files in `results/local_pc/`. These represent weeks of GPU/CPU computation and baseline records.
4. **Environment Isolation**:  
   Outputs from PARAM SHAVAK cluster runs must write **exclusively** to `param_shavak/results/` and never touch `results/local_pc/`.
5. **Clean Staging for PARAM SHAVAK**:  
   Do not copy unverified, buggy, or Windows-hardcoded scripts into `param_shavak/`. Staged files must be tested and verified for Linux cluster execution.
6. **No Arbitrary Sizing Folders**:  
   Do not create an `experiment_120k` folder or hardcode dataset sizes. Sizing will be configured via environment variables and CLI arguments.
7. **Zero Touching of Original 4 Repositories**:  
   Never modify, delete, rename, or write into `cross-modal-remote-sensing-retrieval`, `deep-cross-final-yr-project`, `majorProject`, or `majorproject_week3`. They are read-only backups.
8. **Preserve File Provenance**:  
   When referencing scripts, preserve the author attribution suffix (e.g. `_jainam.py`, `_adarsh.py`, `_meet.py`, `_fateh.py`) so the team's individual contributions remain clear.
9. **Ask Before Destructive Changes**:  
   If an action could lose data, delete files, or overwrite models, stop and ask the user for explicit confirmation.
10. **Evidence-Based Reporting**:  
    Never claim that code "works", "is ready", or "has been verified" unless you have physically run a test or statically inspected the logic. Clearly distinguish verified facts from assumptions.
11. **Update Documentation Continuously**:  
    Follow the Mandatory Documentation Maintenance Rule (Section 6) whenever you take any action or discover new facts.

---

## 5. Phased Project Roadmap & Immediate Next Action

The project follows a disciplined seven-phase engineering roadmap:

```
[Phase 1: Consolidation] ──> [Phase 2: Codebase Audit] ──> [Phase 3: Canonical Selection]
     (COMPLETED)                     (ACTIVE)                       (PENDING)
                                         │
[Phase 6: HPC Experiments] <── [Phase 5: HPC Staging] <── [Phase 4: Local Pipeline Fix]
         │
[Phase 7: Final Report & Paper]
```

### Phase Breakdown
- **Phase 1: Consolidation & Restructuring** `[COMPLETED]`  
  Consolidate 4 repositories into `SENTINEL-RS`, isolate `results/local_pc/`, build `param_shavak/` skeleton, verify mathematical file accounting.
- **Phase 2: Codebase Understanding & Dependency Audit** `[CURRENT ACTIVE PHASE]`  
  Perform a read-only, week-by-week audit of the scripts in `weeks/`. Map internal dependencies, inspect imports, identify hardcoded Windows paths (`C:\Users\hp\...`), and document discrepancies between member implementations.
- **Phase 3: Duplicate Comparison & Canonical Module Selection** `[PENDING]`  
  Compare the four versions of `two_tower.py`, preprocessing scripts, and DataLoaders. Select the canonical, best-tested implementation for each role.
- **Phase 4: Local 30k Pipeline Validation & Bug Fixing** `[PENDING]`  
  Standardize settings via `shared/config/settings.py` and `.env`. Fix known issues (e.g., single-band duplication in inference retriever, Flask upload modality routing). Test end-to-end inference locally.
- **Phase 5: PARAM SHAVAK Linux Compatibility Preparation** `[PENDING]`  
  Adapt canonical scripts for Linux: enable multi-worker DataLoaders (`num_workers=8/16`, `pin_memory=True`), verify CUDA/multi-GPU settings, and stage clean scripts into `param_shavak/week_01/` through `week_05/`.
- **Phase 6: Larger-Scale Execution on PARAM SHAVAK** `[PENDING]`  
  Deploy to the PARAM SHAVAK cluster. Execute preprocessing, training, embedding extraction, and FAISS indexing at scale. Output strictly to `param_shavak/results/`.
- **Phase 7: Final Evaluation, Documentation & Paper Preparation** `[PENDING]`  
  Generate final cross-modal and same-modal Recall@K and mAP comparison tables, publication-quality t-SNE figures, and prepare the final IEEE report.

### Exact Immediate Next Task for the Agent
> **WEEK 1–5 LOCAL PIPELINE IS COMPLETE (November 12, 2026)**  
> All five weeks have been fully audited, corrected, and statically validated locally.  
>
> **NEXT TASK: PARAM SHAVAK Cluster Restoration & HPC Execution**
> 1. Restore SSH access to PARAM SHAVAK.
> 2. Inspect available hardware, CUDA version, Python environment.
> 3. Install `torch`, `torchvision`, `faiss-gpu`, `rasterio`, `matplotlib` on cluster.
> 4. Adapt `param_shavak/week_01/` and `param_shavak/week_02/` paths for Linux.
> 5. Connect to full BigEarthNet dataset mount (do NOT use local 30k subset stats).
> 6. Recompute full-dataset normalization statistics (training split only) → `param_shavak/results/week_02/`.
> 7. Run Week 4 training on full dataset → `param_shavak/results/week_04/`.
> 8. Run Week 5 embedding extraction + FAISS build + retrieval evaluation → `param_shavak/results/week_05/`.
>
> **Strict Constraints:**
> - Do NOT modify `results/local_pc/` (historical PC outputs are frozen).
> - Do NOT overwrite `weeks/` source code (already corrected and validated).
> - All new HPC outputs go to `param_shavak/results/` only.

---

## 6. Mandatory Documentation Maintenance Rule

To ensure that knowledge is never lost across conversation resets or different AI agents, **every AI agent working on this repository must obey this rule**:

> **Whenever you perform significant research, audit code, verify files, make an architectural decision, fix a bug, run an experiment, or resolve an issue, you MUST immediately update the corresponding documentation before ending your turn:**
> 1. **Record technical decisions and user Q&As** in `shared/docs/agent_handoff/DECISIONS_AND_RATIONALE.md`.
> 2. **Record verification steps, audit trails, and data checks** in `shared/docs/agent_handoff/RESEARCH_AND_AUDIT_LOG.md`.
> 3. **Update current phase, completed tasks, and the next immediate step** in `shared/docs/agent_handoff/CURRENT_PROJECT_STATUS.md`.
> 4. **Update permanent technical knowledge, equations, and specs** in `shared/docs/agent_handoff/PROJECT_KNOWLEDGE_BASE.md`.
> 5. **Update file mappings** in `PROJECT_FILE_MAPPING.md` if any new file is created.

*Never leave project knowledge solely in conversational chat memory.*
