# PARAM SHAVAK Migration Checklist
> **Status**: Living document — accumulates items from all Phase 2 audit weeks.  
> **Last Updated**: November 10, 2026 (Week 2 Audit)  
> **Purpose**: Every change, configuration, installation, or verification required before SENTINEL-RS can run on the PARAM SHAVAK Linux HPC cluster.  
> **Labels**: `[VERIFIED]` = confirmed in source code | `[UNVERIFIED]` = needs server-side check | `[PROPOSED]` = recommended but not yet decided

---

## A. Filesystem & Paths

| # | Item | Affected File(s) | Line(s) | Current Behavior | Required Change | Priority | Evidence |
|:--|:-----|:----------------|:--------|:-----------------|:----------------|:--------:|:---------|
| A1 | Hardcoded `C:\Users\hp\...` Windows path | `preprocess_s2_adarsh.py` | 17-19 | `DATASET_ROOT = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2")` | Replace with env-var or `settings.py` `S2_DIR` | 🔴 Critical | `[VERIFIED]` |
| A2 | Hardcoded `C:\Users\hp\...` Windows path | `compute_stats_adarsh.py` | 15-17 | `DATASET_ROOT = Path(r"C:\Users\hp\out\micro project\...")` | Replace with `settings.py` `S2_DIR` | 🔴 Critical | `[VERIFIED]` |
| A3 | Hardcoded `C:\Users\hp\...` Windows path in `__main__` | `preprocess_s2_adarsh.py` | 223-225 | Specific patch path hardcoded in demo block | Wrap in `if __name__ == "__main__"` and use a CLI arg or env-var | 🟡 High | `[VERIFIED]` |
| A4 | Hardcoded `C:\BigEarthNet-S1-Required\...` Windows path | Multiple scripts (Phase 1) | Various | Dataset accessed via Windows path | Use `S1_DIR` env variable pointing to cluster NFS mount | 🔴 Critical | `[PREVIOUSLY REPORTED]` |
| A5 | Relative path `"metadata.parquet"` (CWD-dependent) | `compute_stats_adarsh.py` | 39 | `pd.read_parquet("metadata.parquet")` — requires running from `results/local_pc/week_01/metadata/` | Replace with `settings.py` `METADATA_PATH` absolute variable | 🔴 Critical | `[VERIFIED]` |
| A6 | Relative path `"csvs/train_split.csv"` | `compute_stats_adarsh.py` | 49 | `pd.read_csv("csvs/train_split.csv")` — CWD-dependent | Replace with `settings.py` `SPLITS_DIR / "train_split.csv"` | 🔴 Critical | `[VERIFIED]` |
| A7 | Relative path `"s2_stats.json"` for stats write | `compute_stats_adarsh.py` | 376 | Saves `s2_stats.json` to CWD | Save to `results/local_pc/week_02/normalization_stats/s2_stats.json` via `settings.py` | 🟡 High | `[VERIFIED]` |
| A8 | Relative path `"s2_stats.json"` for stats load | `preprocess_s2_adarsh.py` | 22-30 | `open("s2_stats.json")` — CWD-dependent | Load via `settings.py` `S2_STATS_PATH` | 🔴 Critical | `[VERIFIED]` |
| A9 | Stats path `parent / "s2_stats.json"` (script-relative) | `preprocess_s2_jainam.py` | 18 | `STATS_PATH = Path(__file__).resolve().parent / "s2_stats.json"` — expects stats file alongside script | Redirect to canonical `results/local_pc/week_02/normalization_stats/s2_stats.json` | 🟡 High | `[VERIFIED]` |
| A10 | Stats path `parent.parent / "s1_stats.json"` (script-relative) | `sar_preprocessing_adarsh.py` | 23 | `STATS_PATH = Path(__file__).resolve().parent.parent / "s1_stats.json"` — expects stats 2 dirs up | Redirect to canonical `results/local_pc/week_02/normalization_stats/s1_stats.json` | 🟡 High | `[VERIFIED]` |
| A11 | Stats path `parent / "s1_stats.json"` (script-relative) | `sar_preprocessing_fateh.py` | 23 | `STATS_PATH = Path(__file__).resolve().parent / "s1_stats.json"` — same script directory | Redirect to canonical stats path | 🟡 High | `[VERIFIED]` |
| A12 | S1 stats loaded from `S1_STATS_PATH` via `config.settings` | `sar_preprocessing_meet.py` | 37-43 | Imports `S1_STATS_PATH` from `config.settings` | Ensure `config.settings` is centralized in `shared/config/settings.py` with correct path on cluster | 🟡 High | `[VERIFIED]` |
| A13 | Cache file `.s1_patch_cache.json` written to `BASE_DIR` | `dataset_meet.py`, `dataset_adarsh.py`, `dataset_fateh.py` | ~163, ~317 | Cache written to old repo root `BASE_DIR` | On cluster, `BASE_DIR` must resolve to a writable scratch directory, not shared read-only NFS | 🟡 High | `[VERIFIED]` |
| A14 | DataLoader default CSV paths point to `teammate_inputs/` or `csvs/` | `dataloader_meet.py` L133-134; `dataloader_adarsh.py` L139-140 | 133-134 | Meet: `BASE_DIR/"teammate_inputs"/train_split.csv`; Adarsh: `BASE_DIR/"csvs"/train_split.csv` | Unify to `results/local_pc/week_02/splits/` via `settings.py` `SPLITS_DIR` | 🟡 High | `[VERIFIED]` |
| A15 | Preprocesser demo uses full hardcoded Windows patch path | `preprocesser_adarsh.py` | 7 | `patch_folder = Path(r"C:\Users\hp\out\...")` | Script is a one-off demo; not in pipeline. Exclude from PARAM SHAVAK staging. | 🟢 Low | `[VERIFIED]` |
| A16 | `plt.show()` in `preprocesser_adarsh.py` line 99 | `preprocesser_adarsh.py` | 99 | Will hang on headless server | Replace with `plt.savefig()` + `plt.close()` | 🔴 Critical | `[VERIFIED]` |

---

## B. Python Environment

| # | Item | Affected File(s) | Current Behavior | Required Change | Priority | Evidence |
|:--|:-----|:----------------|:-----------------|:----------------|:--------:|:---------|
| B1 | `torch`, `torchvision` | All training scripts | Required for tensor ops, ResNet-50 | Install compatible version for cluster CUDA | 🔴 Critical | `[VERIFIED]` |
| B2 | `pytorch_lightning` / `lightning` | Week 4+ training | Used for training loop abstraction | Confirm version compatibility with cluster PyTorch | 🟡 High | `[PREVIOUSLY REPORTED]` |
| B3 | `rasterio` / GDAL | `sar_preprocessing_meet.py`, `preprocess_s2_adarsh.py`, `compute_stats_adarsh.py` | GeoTIFF reading — `rasterio` with PIL fallback in some scripts | Install `rasterio` via `conda install -c conda-forge rasterio` on cluster (GDAL dependency complex on Linux) | 🔴 Critical | `[VERIFIED]` |
| B4 | `faiss-gpu` | Week 5 retrieval | FAISS `IndexFlatIP` for embedding search | Use `faiss-gpu` on PARAM SHAVAK (not `faiss-cpu`); verify CUDA compatibility | 🔴 Critical | `[PREVIOUSLY REPORTED]` |
| B5 | `PIL` (Pillow) | `sar_preprocessing_adarsh.py`, `sar_preprocessing_fateh.py`, `preprocess_s2_jainam.py`, `preprocess_s2_fateh.py` | Used as primary image reader (no rasterio) | Install Pillow; note Float32 mode `"F"` required for SAR — verify no uint8 truncation | 🟡 High | `[VERIFIED]` |
| B6 | `numpy`, `pandas` | All scripts | Standard data operations | Install; verify numpy ABI compatibility with PyTorch version | 🟡 High | `[VERIFIED]` |
| B7 | `pyarrow` | Week 1 `explore_metadata_jainam.py` | PyArrow Parquet reading | Install `pyarrow` | 🟢 Low | `[VERIFIED]` |
| B8 | `flask`, `werkzeug` | Week 5 `app_meet.py` | Web portal — not needed for cluster training | Do NOT install on training node. Web portal runs separately. | 🟢 Low | `[PREVIOUSLY REPORTED]` |
| B9 | `sklearn` (scikit-learn) | Evaluation scripts | Metrics computation | Install `scikit-learn` | 🟡 High | `[PREVIOUSLY REPORTED]` |
| B10 | `tqdm` | Training loops | Progress bars | Install; works fine headless on Linux | 🟢 Low | `[PREVIOUSLY REPORTED]` |
| B11 | `config.settings` module | `sar_preprocessing_meet.py`, `dataset_adarsh.py`, `dataset_fateh.py`, `dataloader_meet.py`, `dataloader_adarsh.py` | Imports `IMAGE_SIZE`, `NORMALIZATION_MODE`, `S1_STATS_PATH`, `VV_BAND_SUFFIX`, etc. | The `config/settings.py` from each member's original repo root must be replaced by one centralized `shared/config/settings.py` with all paths as env-var overridable constants | 🔴 Critical | `[VERIFIED]` |
| B12 | Duplicate `config.settings` modules | `dataset_adarsh.py` vs `dataset_fateh.py` | Both import from `config.settings` but via different `sys.path` manipulation | Unify into one canonical `shared/config/settings.py` | 🔴 Critical | `[VERIFIED]` |

---

## C. Linux Compatibility

| # | Item | Affected File(s) | Line(s) | Current Behavior | Required Change | Priority | Evidence |
|:--|:-----|:----------------|:--------|:-----------------|:----------------|:--------:|:---------|
| C1 | Windows backslash paths (`r"C:\..."`) | `preprocess_s2_adarsh.py`, `compute_stats_adarsh.py`, `preprocesser_adarsh.py` | Various | Windows raw string paths — invalid on Linux | Use `Path()` with forward slashes or env-variables | 🔴 Critical | `[VERIFIED]` |
| C2 | `plt.show()` — hangs on headless server | `preprocesser_adarsh.py` L99; `view_random_images_jainam.py` (Week 1) | 99 | Opens GUI window — will block indefinitely | Replace with `plt.savefig(output_path); plt.close()` | 🔴 Critical | `[VERIFIED]` |
| C3 | `num_workers=0` hardcoded for Windows DataLoader stability | `dataloader_meet.py` L138; `dataloader_adarsh.py` L144 | 138, 144 | Comment in code: "default 0 for Windows stability" | On PARAM SHAVAK, set `num_workers=8` (or higher) with `persistent_workers=True` and `pin_memory=True` | 🔴 Critical | `[VERIFIED]` |
| C4 | Windows-only comment in `dataloader_adarsh.py` | `dataloader_adarsh.py` | 29 | `# pls get the config settings from centralized settings.py...` comment left in | Note only; remove before cluster deployment | 🟢 Low | `[VERIFIED]` |
| C5 | `torch` imported first "to prevent DLL conflicts on Windows" | `paired_bigearthnet_dataset_meet.py` | 14-16 | Comment says Windows-specific DLL ordering required | On Linux this ordering is unnecessary; harmless but should be noted | 🟢 Low | `[VERIFIED]` |
| C6 | `PIL.Image.fromarray(array, mode="F")` for Float32 SAR | `sar_preprocessing_meet.py`, `sar_preprocessing_adarsh.py`, `sar_preprocessing_fateh.py` | ~226, ~189, ~25 | Float32 mode `"F"` — PIL supports this on Linux | Verify no silent uint8 truncation occurs; test on Linux with actual data | 🟡 High | `[VERIFIED]` |
| C7 | `Image.Resampling.BILINEAR` requires Pillow ≥ 9.1 | `preprocess_s2_jainam.py`, `preprocess_s2_fateh.py`, `sar_preprocessing_adarsh.py`, `sar_preprocessing_fateh.py` | Various | Uses `Image.Resampling.BILINEAR` (new API) | Confirm Pillow ≥ 9.1 installed on cluster; older Pillow uses `Image.BILINEAR` | 🟡 High | `[VERIFIED]` |
| C8 | `os.scandir()` used for directory scanning | `dataset_meet.py`, `dataset_adarsh.py`, `dataset_fateh.py`, `inspect_dataset_meet.py` | Various | Uses `os.scandir` — Linux-compatible | ✅ No change needed; `os.scandir` is portable | ✅ None | `[VERIFIED]` |
| C9 | `folder.rglob("*")` in Jainam/Fateh S2 preprocessing | `preprocess_s2_jainam.py` L78-83; `preprocess_s2_fateh.py` L13-14 | 78-83 | `rglob("*")` on Linux is case-sensitive | BigEarthNet band filenames (e.g. `_B04.tif`) must match exactly; verify no `.TIF` vs `.tif` issue | 🟡 High | `[VERIFIED]` |
| C10 | `multiprocessing start method` default | `dataloader_meet.py`, `dataloader_adarsh.py` | — | Windows uses `spawn`; Linux default is `fork` | Ensure Dataset `__getitem__` is fork-safe (no open file handles held at module level). Meet's `sar_preprocessing_meet.py` uses `functools.lru_cache` — verify safe under `fork` | 🟡 High | `[UNVERIFIED]` |

---

## D. Hardware & Execution

| # | Item | Affected File(s) | Current Behavior | Required Change | Priority | Evidence |
|:--|:-----|:----------------|:-----------------|:----------------|:--------:|:---------|
| D1 | `pin_memory = torch.cuda.is_available()` auto-detect | `dataloader_meet.py` L69, `dataloader_adarsh.py` L75 | Auto-detects GPU at DataLoader creation | On PARAM SHAVAK, explicitly set `pin_memory=True` for multi-GPU runs | 🟡 High | `[VERIFIED]` |
| D2 | `num_workers` must be raised from 0 to 8+ on cluster | Both DataLoaders | Default 0 (Windows); comments acknowledge this | Set `num_workers=8` or `16` depending on cluster node config | 🔴 Critical | `[VERIFIED]` |
| D3 | `persistent_workers=True` not set | Both DataLoaders | Workers respawn each epoch | Add `persistent_workers=True` when `num_workers > 0` for HPC efficiency | 🟡 High | `[PROPOSED]` |
| D4 | `prefetch_factor` not configured | Both DataLoaders | Default prefetch (2 batches) | Tune `prefetch_factor=4` for large-scale runs | 🟢 Low | `[PROPOSED]` |
| D5 | GPU/CUDA version requirements | Training scripts | Assumes CUDA-capable GPU with PyTorch | `[UNVERIFIED]` — PARAM SHAVAK GPU type, CUDA version and driver version not yet confirmed | 🔴 Critical | `[UNVERIFIED]` |
| D6 | SLURM/PBS job script requirements | Cluster execution | Local run via `python train.py` | `[UNVERIFIED]` — PARAM SHAVAK scheduler type not confirmed; likely SLURM but requires cluster admin confirmation | 🔴 Critical | `[UNVERIFIED]` |
| D7 | Multi-GPU `DistributedDataParallel` strategy | Training scripts | Single-GPU PyTorch Lightning | `[UNVERIFIED]` — whether PARAM SHAVAK allocates 1 or multiple GPUs per job; `strategy="ddp"` only if multi-GPU | 🔴 Critical | `[UNVERIFIED]` |
| D8 | Checkpoint resume from `best_model-v1.ckpt` | Week 5 scripts | Local checkpoint at `shared/models/best_model-v1.ckpt` | Copy checkpoint to cluster scratch; update `CHECKPOINT_PATH` env-variable | 🟡 High | `[VERIFIED path exists]` |
| D9 | Batch size for large-scale run | Training scripts | Local 30k run used batch_size=64 | `[UNVERIFIED]` — cluster VRAM per GPU determines maximum batch size; do not hardcode | 🟡 High | `[UNVERIFIED]` |

---

## E. Data & Artifact Management

| # | Item | Affected File(s) | Current Behavior | Required Change | Priority | Evidence |
|:--|:-----|:----------------|:-----------------|:----------------|:--------:|:---------|
| E1 | BigEarthNet-S1 dataset must be on cluster storage | All SAR scripts | Accessed from `C:\BigEarthNet-S1-Required\` or `D:\Dataset\` | Transfer to PARAM SHAVAK NFS/scratch; set `S1_DIR` env-var | 🔴 Critical | `[VERIFIED]` |
| E2 | BigEarthNet-S2 dataset must be on cluster storage | All S2 scripts | Accessed from `C:\Users\hp\out\...\BigEarthNet-S2` | Transfer to PARAM SHAVAK NFS/scratch; set `S2_DIR` env-var | 🔴 Critical | `[VERIFIED]` |
| E3 | `metadata.parquet` and cloud-filter Parquet required | Preprocessing | Stored in `results/local_pc/week_01/metadata/` | Copy to cluster; set `METADATA_PATH` env-var | 🔴 Critical | `[VERIFIED]` |
| E4 | `train_split.csv`, `val_split.csv`, `test_split.csv` required | Dataset/DataLoader | Stored in `results/local_pc/week_02/splits/` | Copy to cluster; set `SPLITS_DIR` env-var | 🔴 Critical | `[VERIFIED]` |
| E5 | `s1_stats.json`, `s2_stats.json` required before training | Preprocessing | Stored in `results/local_pc/week_02/normalization_stats/` | Copy to cluster; set `S1_STATS_PATH`, `S2_STATS_PATH` env-vars | 🔴 Critical | `[VERIFIED]` |
| E6 | PARAM SHAVAK results must go to `param_shavak/results/` only | All output scripts | Currently all results go to `results/local_pc/` | Configure `OUTPUT_DIR` env-var to `param_shavak/results/` on cluster; never write to `results/local_pc/` from cluster | 🔴 Critical | `[VERIFIED]` |
| E7 | `.s1_patch_cache.json` written to `BASE_DIR` | Dataset classes | Cache file written to old repo root | On cluster, write to a writable scratch path; not to shared/read-only repo directory | 🟡 High | `[VERIFIED]` |
| E8 | `s2_stats.json` channel order must match preprocessing | `compute_stats_adarsh.py`, all S2 preprocess scripts | `compute_stats_adarsh.py` saves stats as `[B04_Red, B03_Green, B02_Blue]` channel order | Fateh's `preprocess_s2_fateh.py` loads bands as `B02, B03, B04` — **stats channel mismatch** (see Section F below) | 🔴 Critical | `[VERIFIED]` |
| E9 | BigEarthNet-S2 nesting: tile → patch directory structure | `compute_stats_adarsh.py`, S2 Dataset | `DATASET_ROOT / tile_name / patch_id` | Verify PARAM SHAVAK uses identical directory structure | 🟡 High | `[VERIFIED]` |

---

## F. Critical Incompatibility — Channel Order Mismatch

> [!CAUTION]
> **This is a correctness bug, not just a portability issue. It must be resolved before any training run.**

The S2 normalization statistics (`s2_stats.json`) were computed by **`compute_stats_adarsh.py`** in channel order: **`[B04 (Red), B03 (Green), B02 (Blue)]`** (index 0=Red, 1=Green, 2=Blue).

However, **`preprocess_s2_fateh.py`** loads bands in order **`B02, B03, B04`** (index 0=Blue, 1=Green, 2=Red) — the reverse — and then applies the population statistics from `s2_stats.json` as-is, applying **Red stats to the Blue channel and Blue stats to the Red channel**.

| Script | Band Load Order | Stats Applied | Match? |
|:-------|:---------------|:-------------|:------:|
| `compute_stats_adarsh.py` (stats generator) | B02, B03, B04 → reordered to [B04, B03, B02] | — | Reference |
| `preprocess_s2_adarsh.py` | B02, B03, B04 → reordered to [B04, B03, B02] | Population stats [0]=B04, [1]=B03, [2]=B02 | ✅ |
| `preprocess_s2_meet.py` | B04, B03, B02 (direct) | Population stats [0]=B04, [1]=B03, [2]=B02 | ✅ |
| `preprocess_s2_jainam.py` | B04, B03, B02 (default bands arg) **but uses per-patch stats** | Per-patch mean/std (no population stats) | ⚠️ |
| `preprocess_s2_fateh.py` | B02, B03, B04 (default bands arg, **not reordered**) | Population stats [0]=B04 applied to index 0=B02 | ❌ **MISMATCH** |

This is a **Phase 4 fix item** — do not change code now.

---

*Document initialized: November 10, 2026 | Next update: After Week 3 Audit*

