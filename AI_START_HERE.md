# AI_START_HERE.md — Universal AI Agent Entry Point for SENTINEL-RS

> **CRITICAL INSTRUCTION FOR ANY AI AGENT / ASSISTANT**:  
> Read this document first before inspecting code, executing commands, or responding to the user. This repository contains a fully consolidated research project with strict environment segregation and historical experimental integrity that must not be broken.

---

## 1. Project Identification

- **Project Name**: **SENTINEL-RS** (Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data)
- **Domain**: Multimodal Earth Observation, Deep Metric Learning, Cross-Sensor Image Retrieval
- **Sensors**: Sentinel-1 Synthetic Aperture Radar (SAR: VV/VH bands) & Sentinel-2 Optical/Multispectral (MS: B04, B03, B02 RGB bands)
- **Core Technology**: Dual-ResNet-50 Two-Tower Neural Network, Symmetric InfoNCE Contrastive Loss, 512-D L2-Normalized Metric Latent Space, FAISS `IndexFlatIP` Similarity Search (<2ms latency), Flask-based Atlas Intelligence Web Portal.
- **Repository Location**: `C:\Users\hp\out\major project\SENTINEL-RS`

---

## 2. Mandatory Reading Order

Do **NOT** ask the user to re-explain the project, architecture, history, or previous decisions. Everything has been thoroughly documented and verified. Follow this exact reading sequence:

1. **[`shared/docs/agent_handoff/AGENT_HANDOFF.md`](shared/docs/agent_handoff/AGENT_HANDOFF.md)**  
   *Your primary operating manual.* Direct instructions to AI agents, non-negotiable safety constraints, environment isolation rules, file-handling mandates, and the future documentation maintenance rule.
2. **[`shared/docs/agent_handoff/CURRENT_PROJECT_STATUS.md`](shared/docs/agent_handoff/CURRENT_PROJECT_STATUS.md)**  
   *Current state of work.* Active phase, verified completed deliverables, pending milestones, blockers, and the exact next task.
3. **[`shared/docs/agent_handoff/PROJECT_KNOWLEDGE_BASE.md`](shared/docs/agent_handoff/PROJECT_KNOWLEDGE_BASE.md)**  
   *Deep technical knowledge.* Research problem, mathematical formulation, 5-week development plan, member contributions, experimental benchmarks, and full component inventory.
4. **[`shared/docs/agent_handoff/DECISIONS_AND_RATIONALE.md`](shared/docs/agent_handoff/DECISIONS_AND_RATIONALE.md)**  
   *Architectural memory.* Every past user question and verified answer, technical rationale, file count reconciliation equations, and established architectural invariants.
5. **[`shared/docs/agent_handoff/RESEARCH_AND_AUDIT_LOG.md`](shared/docs/agent_handoff/RESEARCH_AND_AUDIT_LOG.md)**  
   *Verification history & cryptographic audit.* Exact reconciliations, SHA-256 hash checks, duplicate file analysis, and safety evaluations.
6. **[`PROJECT_FILE_MAPPING.md`](PROJECT_FILE_MAPPING.md)**  
   *Provenance registry.* Exhaustive file-by-file table linking all 189 consolidated files to their original repository, author, development week, size, and function.

---

## 3. Non-Negotiable Safety Rules

You must strictly obey the following rules without exception:

1. **NEVER modify or write into the four original member repositories**:
   - `C:\Users\hp\out\major project\cross-modal-remote-sensing-retrieval` (64 files)
   - `C:\Users\hp\out\major project\deep-cross-final-yr-project` (74 files)
   - `C:\Users\hp\out\major project\majorProject` (81 files)
   - `C:\Users\hp\out\major project\majorproject_week3` (40 files)
   These are pristine historical backups and must remain 100% untouched.
2. **NEVER overwrite or contaminate historical local PC results**:
   `results/local_pc/` holds verified artifacts from the 30k local experiment (splits, stats, embeddings, FAISS indices, metrics, checkpoints). They must never be deleted, modified, or overwritten by cluster runs.
3. **PARAM SHAVAK environment must remain clean & segregated**:
   - `param_shavak/` code directories (`week_01/` to `week_05/`) must remain **completely empty** until code has been audited, standardized, and tested for Linux HPC compatibility.
   - Outputs from PARAM SHAVAK runs must go strictly to `param_shavak/results/` and **never** mix with `results/local_pc/`.
4. **DO NOT create an `experiment_120k` folder**:
   The final HPC dataset size is undetermined and will be decided at runtime. Keep `param_shavak/` clean and generic.
5. **DO NOT rewrite the project from scratch**:
   Respect existing architecture, checkpoints, preprocessing pipelines, and member attribution. Work incrementally.
6. **DO NOT claim code or experiments were tested unless you physically executed and verified them**:
   Clearly mark findings as `[VERIFIED]`, `[PREVIOUSLY REPORTED]`, `[UNVERIFIED]`, or `[PROPOSED]`.

---

## 4. Current Exact Next Task

The project has completed **Phase 1: Consolidation and Restructuring**.  
The current active task for the incoming agent is:

> **Phase 2: Perform a read-only, week-by-week audit of the existing codebase under `weeks/`, building a dependency map and comparing duplicate implementations across team members before making any code changes or staging files to `param_shavak/`.**

Proceed to **[`shared/docs/agent_handoff/AGENT_HANDOFF.md`](shared/docs/agent_handoff/AGENT_HANDOFF.md)** to begin.

