# Week 5 — FAISS Retrieval System, Atlas Portal & Scientific Evaluation

**SENTINEL-RS: Cross-Modal Remote-Sensing Image Retrieval**  
**Contributors:** Jainam, Meet, Adarsh, Fateh  
**Phase B Implementation Status:** COMPLETE (locally validated)  
**Implementation Report:** [`shared/docs/audit_report/week5_phaseB_implementation_report.md`](../../shared/docs/audit_report/week5_phaseB_implementation_report.md)

---

## 1. Objective

Week 5 operationalises the trained TwoTowerNetwork from Week 4 into a full-stack
retrieval system. It covers:

| Component | Description |
|---|---|
| Embedding extraction | Extract 512-D SAR and MS embeddings from the trained model |
| L2 normalization | Normalize embeddings for cosine similarity via inner product |
| FAISS index construction | Build `IndexFlatIP` galleries for SAR and MS modalities |
| Cross-modal retrieval | SAR→MS, MS→SAR (the primary research task) |
| Same-modal retrieval | SAR→SAR, MS→MS (baseline comparison) |
| Retrieval evaluation | mAP, Recall@K, Precision@K metrics |
| Scientific precision | Metric relabelling to `HitRate@K` and `mAP_ret@10` |
| Atlas portal | Flask web portal for interactive retrieval (Meet) |
| Latency benchmarking | End-to-end pipeline timing (Meet) |

---

## 2. Architecture

### 2.1 Two-Tower Network (from Week 4)

```
SAR input  (B, 2, 224, 224)  ─► SAR ResNet-50 encoder ─► SAR projection head ─► (B, 512)
                                                          Linear(2048→1024) → ReLU → Linear(1024→512)

MS input   (B, 3, 224, 224)  ─► MS  ResNet-50 encoder ─► MS  projection head ─► (B, 512)
```

Embeddings are L2-normalized before indexing and querying (cosine similarity via inner product).

### 2.2 FAISS Gallery

| Property | Value |
|---|---|
| Index type | `faiss.IndexFlatIP` (exact inner product, no quantization) |
| Gallery size | 4,500 test pairs (SAR gallery + MS gallery, independent) |
| Embedding dim | 512 |
| Normalization | L2 unit-norm (cosine similarity = inner product) |
| Similarity range | \[−1, +1\] (practically \[0, 1\] after L2 normalization with positive-definite activations) |

### 2.3 Retrieval Modes

| Mode | Query | Gallery | Research Role |
|---|---|---|---|
| SAR→MS | Sentinel-1 SAR patch | Multispectral gallery | **Primary cross-modal task** |
| MS→SAR | Sentinel-2 MS patch | SAR gallery | **Primary cross-modal task** |
| SAR→SAR | Sentinel-1 SAR patch | SAR gallery | Baseline (same-modal) |
| MS→MS | Sentinel-2 MS patch | MS gallery | Baseline (same-modal) |

---

## 3. Metric Definitions (Phase A.1 Scientific Corrections)

> [!IMPORTANT]
> The same-modal metric names have been corrected to match their actual
> implementations. Historical numerical values are preserved unchanged.

### 3.1 Cross-Modal Evaluation

Standard **mAP@10** (truncated AP normalized by `min(total_relevant, K)`):

$$\text{AP}@K = \frac{1}{\min(R, K)} \sum_{k=1}^{K} \text{precision}(k) \cdot \text{rel}(k)$$

Historical values:
| Metric | SAR→MS | MS→SAR |
|---|---|---|
| mAP@10 | 65.33% | 63.17% |
| Recall@10 | 72.14% | 69.88% |
| Precision@10 | 64.52% | 62.75% |

### 3.2 Same-Modal Evaluation

> [!WARNING]
> The same-modal evaluators use **non-standard metric definitions**. The historical
> names were `mAP@10` and `Recall@K`. Corrected names are:

**`HitRate@K`** (what the code implements as "Recall@K"):
$$\text{HitRate}@K = \frac{\text{queries with} \geq 1 \text{ relevant result in top-}K}{N_{\text{queries}}}$$

This is **not** conventional Recall@K (which would divide total retrieved relevant items by total relevant items).

**`mAP_ret@10`** (what the code implements as "mAP@10"):
$$\text{AP}_{\text{ret}}@K = \frac{\sum_{k=1}^{K} \text{precision}(k) \cdot \text{rel}(k)}{\sum_{k=1}^{K} \text{rel}(k)}$$

Denominator = number of *retrieved* relevant items, not `min(total_relevant, K)`.

Historical values (preserved, now labelled correctly):
| Metric | SAR→SAR | MS→MS |
|---|---|---|
| `mAP_ret@10` | 61.24% | 64.09% |
| `HitRate@10` | 68.57% | 71.23% |

---

## 4. Non-Destructive Identifier Mapping

### 4.1 Problem

SAR patch IDs in `trained_test_sar_patch_ids.txt` are paired `patch_id` strings
(`S2A_...`) used to match cross-modal positive pairs. The native SAR radar acquisition
directory names (`S1B_IW_...`) live in BigEarthNet-S1 on disk.

### 4.2 Solution

A non-destructive mapping CSV bridges the two namespaces:

```
results/local_pc/week_05/trained_embeddings/test_patch_id_mapping.csv
```

Columns: `row_index`, `patch_id` (paired S2A string), `s1_name` (native S1B radar dir), `s2v1_name`.

This mapping is consumed by `identifier_mapper.py` and `retriever_meet.py` to resolve
local disk thumbnails without modifying any historical manifests.

**SHA256:** `5fe6c65bebcd794e8e78d7cb0b04accc982105df78afec292fac4108cee843c9`

---

## 5. File Map

```
weeks/week_05/
├── benchmarking/
│   └── timing_meet.py           # TwoTowerBenchmark (replaces PlaceholderEncoder self-test)
├── deployment/
│   ├── app_fateh.py             # ⚠️ PROTOTYPE MOCK DEMO — synthetic results only
│   ├── app_meet.py              # Production Atlas Flask portal (canonical)
│   ├── run_pipeline_meet.py     # ⚠️ LEGACY — Week 1/2 SAR pipeline (preserved)
│   ├── templates/               # Jinja2 HTML templates
│   └── static/                  # CSS assets
├── embeddings/
│   ├── generate_trained_embeddings_adarsh.py   # Val embeddings from checkpoint
│   ├── extract_test_embeddings_jainam.py        # Test embeddings (Jainam)
│   └── extract_test_embeddings_meet.py          # Test embeddings (Meet)
├── evaluation/
│   ├── evaluate_trained_cross_modal_jainam.py  # Cross-modal FAISS eval
│   ├── evaluate_trained_cross_modal_meet.py
│   ├── evaluate_trained_same_modal_jainam.py   # Same-modal eval (HitRate@K, mAP_ret@10)
│   ├── evaluate_trained_same_modal_meet.py
│   ├── create_week5_final_report_jainam.py     # Comparison report generator
│   └── test_all_subcategories_meet.py          # Flask + subcategory integration tests
├── retrieval/
│   ├── build_faiss_indices_jainam.py           # FAISS index builder (Jainam)
│   ├── build_faiss_indices_meet.py             # FAISS index builder (Meet)
│   ├── identifier_mapper.py                     # Non-destructive ID mapping helper
│   ├── retriever_meet.py                        # CrossModalRetriever (canonical)
│   └── test_retriever_meet.py                   # Retriever unit & integration tests
└── visualization/
    ├── generate_trained_tsne_meet.py
    ├── success_failure_cases_adarsh.py
    ├── tsne_visualization_meet.py
    └── visual_retrieval_grid_adarsh.py
```

---

## 6. Data Artifacts

```
results/local_pc/week_05/
├── trained_embeddings/
│   ├── trained_test_sar_embeddings.npy        # Historical (from original contributors)
│   ├── trained_test_ms_embeddings.npy         # Historical
│   ├── trained_test_sar_patch_ids.txt          # Historical manifest
│   ├── trained_val_sar_embeddings_l2.npy      # Phase B: L2-normalized val SAR
│   ├── trained_val_ms_embeddings_l2.npy       # Phase B: L2-normalized val MS
│   ├── trained_val_l2_metadata.json            # Phase B: normalization verification
│   └── test_patch_id_mapping.csv              # Phase B: non-destructive ID bridge
├── evaluation/
│   └── runs/                                   # Evaluation run outputs
├── faiss_indices/                              # FAISS index files (requires checkpoint)
│   ├── sar_gallery.index
│   └── ms_gallery.index
└── benchmarks/                                # timing_meet.py JSON outputs
```

---

## 7. CLI Reference

### Generate validation embeddings
```bash
python weeks/week_05/embeddings/generate_trained_embeddings_adarsh.py \
    --checkpoint results/local_pc/week_04/checkpoints/best_model-v1.ckpt \
    --normalize
```

### Build FAISS indices
```bash
python weeks/week_05/retrieval/build_faiss_indices_jainam.py \
    --sar-embeddings results/local_pc/week_05/trained_embeddings/trained_test_sar_embeddings.npy \
    --ms-embeddings  results/local_pc/week_05/trained_embeddings/trained_test_ms_embeddings.npy
```

### Evaluate cross-modal retrieval
```bash
python weeks/week_05/evaluation/evaluate_trained_cross_modal_jainam.py
```

### Evaluate same-modal retrieval (HitRate@K, mAP_ret@10)
```bash
python weeks/week_05/evaluation/evaluate_trained_same_modal_jainam.py
```

### Run Atlas portal
```bash
python weeks/week_05/deployment/app_meet.py
```

### Run latency benchmark
```bash
python weeks/week_05/benchmarking/timing_meet.py --mode sar --iterations 20
```

---

## 8. Environment Requirements

| Package | Purpose | Status |
|---|---|---|
| `torch` / `torchvision` | Model inference | ⚠️ NOT installed (Python 3.13) |
| `faiss-cpu` | Vector search | ⚠️ NOT installed |
| `flask` / `werkzeug` | Atlas portal | ✅ installed |
| `numpy` | Array operations | ✅ installed |
| `pandas` | CSV/metadata | ✅ installed |
| `Pillow` | Image I/O | ✅ installed |
| `rasterio` | GeoTIFF reading | ⚠️ NOT installed |
| `matplotlib` | Visualization | ⚠️ NOT installed |

> [!CAUTION]
> All neural inference, checkpoint loading, FAISS index construction and search,
> and live Atlas operation are **[BLOCKED BY ENVIRONMENT]** until PyTorch and FAISS
> are installed in the active Python environment.
>
> The checkpoint binary `best_model-v1.ckpt` is not present (pointer file only).
> All reported evaluation metrics originate from the original contributors' environment
> and are preserved as historical record.

---

## 9. Known Limitations

1. **Checkpoint unavailability:** `best_model-v1.ckpt` binary is a pointer file; cannot regenerate embeddings locally.
2. **Same-modal metric semantics:** `HitRate@K` and `mAP_ret@10` differ from standard `Recall@K` and `mAP@K`. Historical values are preserved as-is; recomputation requires checkpoint.
3. **SAR identifier namespace:** `trained_test_sar_patch_ids.txt` stores paired `patch_id` strings, not native S1 radar directory names. Use `identifier_mapper.py` for disk lookups.
4. **Hardcoded gallery size checks:** Some evaluation scripts assert `4500` samples; adjust `--n-samples` CLI arg if split sizes change.

---

## 10. Phase B Corrections Summary

| Issue | Fix Applied |
|---|---|
| Broken `src.*` / `week4.*` imports | Replaced with `weeks.week_04.*` / `weeks.week_02.*` canonical paths |
| Hardcoded machine paths (`C:\Users\hp\...`) | Replaced with `shared.config.settings` / `PROJECT_ROOT`-relative paths |
| Hardcoded `"faiss_ms": "1.2"` fake latency | Now reports `"N/A (see total_ms)"` reflecting actual unmeasured component |
| `app_fateh.py` without disclaimer | Added prominent `PROTOTYPE / MOCK DEMO` module docstring |
| `run_pipeline_meet.py` without provenance label | Added `LEGACY SCRIPT — PRESERVED FOR PROVENANCE` header |
| `timing_meet.py` PlaceholderEncoder toy test | Deprecated with warning; replaced by `TwoTowerBenchmark` class |
| Same-modal `Recall@K` → `HitRate@K` | Relabelled in both `evaluate_trained_same_modal_*.py` files |
| Same-modal `mAP@10` → `mAP_ret@10` | Relabelled with accurate formula description |
| Missing identifier bridge | Created `test_patch_id_mapping.csv` and `identifier_mapper.py` |
| Missing L2-normalized val embeddings | Generated `trained_val_sar/ms_embeddings_l2.npy` |
| `generate_trained_embeddings_adarsh.py` incomplete | Fully implemented with CLI args and env guard |
| `extract_test_embeddings_*.py` broken imports | Canonicalized to `weeks.week_02.*` / `weeks.week_04.*` |

