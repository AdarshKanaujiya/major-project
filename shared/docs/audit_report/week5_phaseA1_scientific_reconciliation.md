# SENTINEL-RS — Week 5 Phase A.1: Scientific Findings Reconciliation Report

**Date:** 2026-11-12  
**Auditor:** Senior ML Research Engineer, PyTorch Architect & Research Code Auditor  
**Project:** SENTINEL-RS — Cross-Modal Remote-Sensing Image Retrieval  
**Scope:** Targeted Scientific Reconciliation of 5 Core Audit Findings (AP@10, Recall@K, SAR Identifier Namespaces, Atlas/FAISS Evidence Classification, and Checkpoint Compatibility)  
**Status:** COMPLETE — STRICTLY READ-ONLY (No source code or historical artifacts modified)

---

## 1. Executive Summary & Purpose

Following the completion of the comprehensive Week 5 audit, this Phase A.1 reconciliation provides exact mathematical, code-level, and empirical clarifications for five specific scientific and technical areas. 

### Summary of Reconciled Findings

1. **Same-Modal AP@10 is NOT Standard Truncated mAP@10**:
   - The implemented formula normalizes Average Precision by the number of relevant items **retrieved in the top 10** ($\sum_{k=1}^{10} \text{rel}(k)$), **not** by $\min(\text{total relevant in gallery}, 10)$.
   - The historical values ($\text{SAR}\to\text{SAR} = 61.24\%$, $\text{MS}\to\text{MS} = 64.09\%$) represent **Retrieved-Normalized Average Precision @ 10** ($\text{AP}_{\text{ret}}@10$), also known as **Conditional Precision @ 10**.
2. **Same-Modal Recall@K is Strictly Hit Rate@K**:
   - The code tests `if any(top_k_relevance): recall_counts[k] += 1`.
   - This measures the fraction of queries with **at least one** relevant neighbor in the top $K$ (Hit Rate / Success Rate), **not** the fraction of all relevant gallery items retrieved (conventional multi-label Recall).
3. **SAR Manifests Intentionally Use Paired-Scene `patch_id`**:
   - Storing the Sentinel-2 `patch_id` in `trained_test_sar_patch_ids.txt` was not an accidental corruption, but a deliberate design choice establishing a 1-to-1 common paired key for cross-modal correspondence.
   - Existing manifests must be **preserved** (not overwritten), and supplemented with native `s1_name` and `s2v1_name` mappings.
4. **Atlas Portal & FAISS Latency Evidence Reclassified**:
   - `app_meet.py` is structurally complete as static code, but is **NOT currently operational** in the active environment (PyTorch, FAISS, and rasterio are absent; imports are broken).
   - FAISS search latency (~0.012 ms/query) is **Historical Execution Evidence** from precomputed CSV artifacts, not a live measurement in the current environment.
5. **Checkpoint Compatibility Explicitly Retains Unverified Status**:
   - While architectural defaults and state-dict key topologies align structurally, live binary checkpoint loading is **BLOCKED** and remains:  
     `ACTUAL CHECKPOINT LOAD NOT VERIFIED`.

---

## 2. Finding 1: Same-Modal AP@10 Mathematical Formulation

### 2.1 Code Inspection
- **Files Inspected**:
  - `weeks/week_05/evaluation/evaluate_trained_same_modal_jainam.py` (lines 316–358)
  - `weeks/week_05/evaluation/evaluate_trained_same_modal_meet.py` (lines 324–366)
- **Implemented Code**:
  ```python
  total_relevant_in_top_k = sum(relevance)

  if total_relevant_in_top_k == 0:
      average_precisions.append(0.0)
      continue

  precision_sum = 0.0
  relevant_seen = 0

  for rank, relevant in enumerate(relevance, start=1):
      if relevant:
          relevant_seen += 1
          precision_at_rank = relevant_seen / rank
          precision_sum += precision_at_rank

  ap_at_10 = precision_sum / total_relevant_in_top_k
  average_precisions.append(ap_at_10)
  ```

### 2.2 Comparison of Mathematical Definitions

Let $q$ be a query with gallery relevance indicators $r_k \in \{0, 1\}$ at ranks $k \in \{1, \dots, K\}$, where $K = 10$.  
Let $P@k = \frac{1}{k} \sum_{j=1}^k r_j$ be the precision at rank $k$.  
Let $R_{\text{total}}$ be the total number of relevant items in the entire gallery sharing at least one land-cover label with $q$.  
Let $R_{\text{ret@}K} = \sum_{k=1}^K r_k$ be the count of relevant items retrieved within the top $K$.

| Metric Definition | Formula | Divisor | Properties / Behavior |
|---|---|---|---|
| **A. Implemented Formula** | $$\text{AP}_{\text{ret}}@K = \frac{1}{\sum_{k=1}^K r_k} \sum_{k=1}^K P@k \cdot r_k$$ | $R_{\text{ret@}K} = \sum_{k=1}^K r_k$ | Normalizes strictly by **retrieved positive count**. If only 2 relevant items are retrieved at ranks 1 and 2, $\text{AP} = \frac{1.0 + 1.0}{2} = 1.0$, regardless of how many relevant items exist in the gallery. |
| **B. Standard Truncated AP@K** | $$\text{AP}_{\text{standard}}@K = \frac{1}{\min(R_{\text{total}}, K)} \sum_{k=1}^K P@k \cdot r_k$$ | $\min(R_{\text{total}}, K)$ | Standard TREC / IR metric. Penalizes systems for failing to retrieve up to $K$ relevant items when $R_{\text{total}} \ge K$. |
| **C. Global Untruncated AP** | $$\text{AP}_{\text{global}} = \frac{1}{R_{\text{total}}} \sum_{k=1}^{N_{\text{gallery}}} P@k \cdot r_k$$ | $R_{\text{total}}$ | Standard PASCAL VOC / COCO object detection & retrieval metric across the complete database. |

### 2.3 Scientific Assessment & Historical Values

> [!IMPORTANT]
> **Conclusion on Historical Values**:
> - **SAR $\to$ SAR mAP@10 = 61.24%**
> - **MS $\to$ MS mAP@10 = 64.09%**
> 
> These numbers represent **Retrieved-Normalized Average Precision @ 10** ($\text{AP}_{\text{ret}}@10$), **not** standard truncated $\text{mAP}@10$.
>
> In BigEarthNet, frequent land-cover classes (e.g., *Arable land*, *Coniferous forest*) have hundreds of matching patches in the 4,500 test set ($R_{\text{total}} \gg 10$). Standard truncated $\text{mAP}@10$ would divide by $\min(R_{\text{total}}, 10) = 10$ for virtually every query. The implemented formula instead divides by the actual number of relevant items found in the top 10 ($1 \le R_{\text{ret@}10} \le 10$).
>
> **Recommended Scientific Naming**:
> - Report as: **Retrieved-Normalized Average Precision @ 10 ($\text{AP}_{\text{ret}}@10$)** or **Conditional Mean Average Precision @ 10 (c-mAP@10)**.
> - Historical values must be preserved verbatim in documentation, accompanied by this explicit definition.

---

## 3. Finding 2: Same-Modal Recall@K Formulation

### 3.1 Code Inspection
- **Files Inspected**:
  - `weeks/week_05/evaluation/evaluate_trained_same_modal_jainam.py` (lines 308–314, 363–375)
  - `weeks/week_05/evaluation/evaluate_trained_same_modal_meet.py` (lines 316–322, 371–383)
- **Implemented Code**:
  ```python
  recall_counts = {1: 0, 5: 0, 10: 0}

  for k in recall_counts:
      top_k_relevance = relevance[:k]
      if any(top_k_relevance):
          recall_counts[k] += 1

  results = {
      "Recall@1":  (recall_counts[1] / total_queries) * 100.0,
      "Recall@5":  (recall_counts[5] / total_queries) * 100.0,
      "Recall@10": (recall_counts[10] / total_queries) * 100.0,
  }
  ```

### 3.2 Distinction: Hit Rate vs. Conventional Recall

1. **Hit Rate @ K ($\text{HR}@K$) / Success Rate @ K**:
   $$\text{HitRate}@K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\sum_{k=1}^K r_{i,k} \ge 1\right)$$
   This evaluates: *Did the model retrieve at least one relevant patch in the top $K$ candidates?*
   - This is **exactly** what the code computes (`any(relevance[:k])`).
2. **Conventional Multi-Label Recall @ K**:
   $$\text{Recall}@K = \frac{1}{N} \sum_{i=1}^N \frac{\sum_{k=1}^K r_{i,k}}{R_{i,\text{total}}}$$
   This evaluates: *What fraction of all matching gallery patches was captured in the top $K$?*
   - Because $R_{\text{total}}$ can be $>1,000$ for dominant classes, conventional Recall@10 would naturally be $<1.0\%$.

### 3.3 Scientific Assessment & Historical Values

> [!NOTE]
> **Conclusion on Historical Values**:
> - **SAR $\to$ SAR**: R@1 = 53.40%, R@5 = 85.98%, R@10 = 92.49%
> - **MS $\to$ MS**: R@1 = 57.84%, R@5 = 87.04%, R@10 = 93.47%
> 
> These numbers represent **Hit Rate @ K ($\text{HR}@K$)** (percentage of queries where at least one of the top $K$ retrievals shares a land-cover class with the query).
>
> In single-positive retrieval (such as cross-modal retrieval), Hit Rate@K is mathematically identical to Recall@K. In same-modal multi-label retrieval, it is Hit Rate@K.
> 
> **Recommended Scientific Naming**:
> - In technical reports, clarify as: **Same-Modal Hit Rate @ K ($\text{HR}@K$, reported historically as Recall@K)**.
> - Preserve all historical values intact without modification.

---

## 4. Finding 3: SAR Identifier Namespace Architecture

### 4.1 Manifest and Split Inspection
- **Files Inspected**:
  - `results/local_pc/week_02/splits/test_split.csv`
  - `results/local_pc/week_02/splits/val_split.csv`
  - `results/local_pc/week_05/trained_embeddings/trained_test_sar_patch_ids.txt`
  - `results/local_pc/week_05/trained_embeddings/trained_test_ms_patch_ids.txt`
  - `weeks/week_05/retrieval/retriever_meet.py` (lines 266–293, 330–367)

### 4.2 Analysis of Historical Intent
1. `test_split.csv` has three key identifier columns:
   - `patch_id`: e.g., `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66` (Sentinel-2 patch ID).
   - `s1_name`: e.g., `S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66` (Sentinel-1 raw folder name).
   - `s2v1_name`: e.g., `S2A_MSIL2A_20170613T101031_39_66` (legacy S2 naming format).
2. The PyTorch dataset (`PairedBigEarthNetDataset`) returns `(sar, ms, label, patch_id)`. The batch identifier yielded during embedding extraction is `patch_id`.
3. Consequently, `trained_test_sar_patch_ids.txt` records `patch_id` for each SAR embedding.
4. **Why this was intentional**:
   In cross-modal evaluation (`evaluate_trained_cross_modal_jainam.py`), SAR and MS embeddings share the row-level `patch_id`. Relevance between query and gallery is computed by testing `if query_patch_id == gallery_patch_id`. If the SAR gallery had stored `s1_name`, string equality would have evaluated to `False` for every pair, falsely yielding 0% Recall.
5. **How the Retriever Handles It**:
   In `retriever_meet.py`, the retriever indexes metadata by `patch_id`. When it needs to locate the raw SAR GeoTIFF on disk, it explicitly executes:
   ```python
   meta = self.metadata_lookup.get(patch_id, {})
   s1_name = meta.get("s1_name", patch_id)
   candidate_s1 = self._local_s1_patches.get(s1_name)
   ```

### 4.3 Non-Destructive Multi-Namespace Design Proposal

> [!IMPORTANT]
> **Architectural Decision**:
> - **DO NOT OVERWRITE** `trained_test_sar_patch_ids.txt`. It is the canonical paired-scene identifier manifest matching the embedding matrices row-for-row.
> - Implement a non-destructive multi-namespace structure for Phase B:
>   1. `patch_id`: Canonical paired-scene identifier (preserves exact matching in cross-modal evaluation and FAISS lookups).
>   2. `s1_name`: Native Sentinel-1 radar scene identifier (used for raw file system access to VV/VH GeoTIFFs).
>   3. `s2v1_name`: Native Sentinel-2 optical identifier (used for raw S2 band access).
> - Supplementary metadata artifact to introduce in Phase B:  
>   `results/local_pc/week_05/trained_embeddings/test_patch_id_mapping.csv` containing explicit columns:  
>   `[row_index, patch_id, s1_name, s2v1_name]`.

---

## 5. Finding 4: Atlas Portal & FAISS Evidence Classification

To ensure complete scientific accuracy, all claims regarding Atlas portal readiness and FAISS latency are reconciled and classified into strict evidence categories.

### Table 5.1: Strict Evidence Classification of Atlas and FAISS Claims

| Claim / Subsystem | Claim Statement | Evidence Classification | Technical Justification |
|---|---|---|---|
| **Atlas Architecture (`app_meet.py`)** | "Fully structured Flask application with domain routing, subcategory handling, and hybrid candidate merging." | `[VERIFIED — STATIC]` | Python syntax compiles with exit code 0 (`py_compile`). Flask routes, argument parsers, and template integration are verified via AST and static inspection. |
| **Atlas Live Operation** | "`app_meet.py` is operational / production-ready." | `[UNVERIFIED / BLOCKED]` | **Cannot be claimed as operational.** The local environment lacks `torch`, `faiss-cpu`, and `rasterio`. Imports (`src.inference.retriever`) fail without path modifications. No live web server was started or probed in this environment. |
| **Atlas Historical Verification** | "Atlas subcategories were verified on real test patches." | `[HISTORICAL EVIDENCE]` | Proven by `week5_subcategory_results.json` (18.6 KB), which records full retrieval outputs for all 6 subcategories executed on the contributor's system. |
| **FAISS Gallery Sizing** | "Indices contain 4,500 512-D float32 vectors totaling 9,216,045 bytes." | `[VERIFIED — DATA]` | Files `sar_gallery.index` and `ms_gallery.index` exist on disk and match the exact mathematical size: $4500 \times 512 \times 4 + 45\text{ B}$. |
| **FAISS Latency (~0.012 ms)** | "FAISS batch search requires ~0.012 ms per query." | `[HISTORICAL EVIDENCE]` | Verified from historical CSV records (`trained_cross_modal_results.csv`, `trained_same_modal_results.csv`). It is **NOT** a currently executed measurement since FAISS is not installed locally. |
| **Inference Latency (1.51 ms)** | "Two-Tower network achieves 1.51 ms CPU inference." | `[DISPROVEN / UNRELIABLE]` | Originates from `PlaceholderEncoder` (a 1-layer toy linear model on random arrays in `timing_meet.py`). Unscientific for ResNet-50. |

---

## 6. Finding 5: Checkpoint Compatibility Re-evaluation

The relationship between the Week 5 codebase and the historical checkpoint `best_model-v1.ckpt` must be delineated into distinct verification layers.

### Table 6.1: Checkpoint Compatibility Verification Layers

| Verification Layer | Status | Evidence Label | Reconciled Finding |
|---|---|---|---|
| **1. Architecture Defaults** | Aligned | `[VERIFIED — STATIC]` | `TwoTowerNetwork` in Week 4 defaults to `use_batchnorm=False`. Week 5 code instantiates `TwoTowerNetwork(pretrained=False)` without passing `use_batchnorm`. Both use the 3-layer `Linear -> ReLU -> Linear` projection head. |
| **2. Key Topology & Prefix Stripping** | Aligned | `[VERIFIED — STATIC]` | Week 5 scripts iterate through `checkpoint['state_dict']`, remove the Lightning wrapper prefix `network.`, and map directly to `sar_encoder`, `ms_encoder`, `sar_projection`, and `ms_projection`. |
| **3. State-Dict Tensor Count** | Aligned | `[HISTORICAL EVIDENCE]` | Historical training records document 644 tensors ($318 \times 2$ ResNet-50 + $4 \times 2$ projection head). |
| **4. Live Checkpoint Load Execution** | **NOT VERIFIED** | `[BLOCKED]` | The physical 598.95 MB binary is absent locally (only `best_model-v1.ckpt.readme.txt` is present). Host environment lacks PyTorch. |

> [!CAUTION]
> **Mandatory Scientific Standing**:
> The status:
> ```
> ACTUAL CHECKPOINT LOAD NOT VERIFIED
> ```
> is strictly retained. Structural key and architectural alignment provide necessary, but not sufficient, proof of operational checkpoint compatibility. Successful execution can only be asserted once the physical binary is mounted and loaded into memory with `strict=True`.

---

## 7. Corrections to the Original Week 5 Audit Report

The following specific statements in `shared/docs/audit_report/week5_comprehensive_read_only_audit.md` are hereby clarified and reconciled:

1. **Section 3.F (Same-modal mAP Formula)**:
   - *Original Statement*: Stated formula as $AP = \frac{1}{\min(R, K)} \sum P@k \cdot \text{rel}(k)$ while prose said "where the divisor is the number of relevant items in top-K".
   - *Reconciled Statement*: The divisor in code is strictly $\sum_{k=1}^{10} \text{rel}(k)$ ($R_{\text{ret@}10}$). The metric is Retrieved-Normalized Average Precision @ 10 ($\text{AP}_{\text{ret}}@10$).
2. **Section 3.F (Same-modal Recall@K)**:
   - *Original Statement*: Referred to same-modal metrics as Recall@1, Recall@5, Recall@10.
   - *Reconciled Statement*: The evaluator computes whether $\ge 1$ relevant item exists in top $K$ (`any(relevance[:k])`), which is strictly Hit Rate@K ($\text{HR}@K$).
3. **Section 3.D & 7.2 (SAR Patch ID Manifest)**:
   - *Original Statement*: Classified the presence of S2 patch IDs in `trained_test_sar_patch_ids.txt` as a defect/flaw.
   - *Reconciled Statement*: Reclassified as an intentional paired-scene identifier convention necessary for exact cross-modal evaluation, which requires supplementary native S1 mapping rather than overwriting.
4. **Section 3.H (Atlas Portal Status)**:
   - *Original Statement*: Described `app_meet.py` as an "operational production application".
   - *Reconciled Statement*: Clarified as statically complete architecture, with live execution currently blocked and unverified due to missing dependencies.

---

## 8. Summary of Recommendations for Phase B Implementation

When authorized to enter Phase B (Controlled Implementation):

1. **Preserve Existing Metrics & Naming**:
   - Keep historical results tables intact; annotate tables with exact metric definitions ($\text{AP}_{\text{ret}}@10$ and $\text{HitRate}@K$).
2. **Non-Destructive Manifest Enhancement**:
   - Retain `trained_test_sar_patch_ids.txt` for cross-modal evaluation compatibility.
   - Create an explicit mapping module / manifest linking `patch_id` $\leftrightarrow$ `s1_name` $\leftrightarrow$ `s2v1_name`.
3. **Canonical Imports & Settings Integration**:
   - Update canonical Week 5 scripts to import `TwoTowerNetwork` from `weeks.week_04.models.two_tower_jainam` and resolve paths via `shared.config.settings`.
4. **Validation Embedding Normalization**:
   - Correct `generate_trained_embeddings_adarsh.py` to apply $L_2$ normalization before saving validation arrays.
5. **Deprecate Placeholder Benchmark Code**:
   - Replace the toy `PlaceholderEncoder` in `timing_meet.py` with real `TwoTowerNetwork` benchmark instrumentation, clearly separating genuine FAISS search latency from neural inference latency.

---
*Report compiled and validated by Antigravity Senior ML Research Engineer.*

