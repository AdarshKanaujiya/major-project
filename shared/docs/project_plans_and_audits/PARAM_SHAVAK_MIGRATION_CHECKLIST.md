
---

## F. Critical Incompatibility — Channel Order Mismatch (STATUS: RESOLVED)

> [!NOTE]
> **Resolution Status**: `[RESOLVED & VERIFIED IN SOURCE CODE]` (November 11, 2026).
> All 15 Week 2 source files were backed up and corrected. `preprocess_s2_fateh.py` was updated to `("B04", "B03", "B02")` order with population Z-score normalization via `s2_stats.json`. Bitwise and single-precision equivalence across all 4 contributors was confirmed empirically on real GeoTIFF patches.

---

## G. Pre-Flight Deployment & Execution Checklist (For Restored HPC Access)

| Step | Action | Command / Target | Verification Criteria | Status |
| :--- | :--- | :--- | :--- | :---: |
| **G1** | **SSH Connectivity** | `ssh <user>@PS-BOSS-OS` | Interactive shell access established | Pending Access |
| **G2** | **Hardware Verification** | `nvidia-smi`, `lscpu`, `free -h`, `df -h /home` | Confirm NVIDIA RTX 5000 Ada, ~12 TB storage | Pending Access |
| **G3** | **Conda Environment Setup** | `conda env create -f shared/config/environment_hpc.yml` | `conda activate sentinel_rs` works | Script Ready |
| **G4** | **PyTorch & CUDA Sanity** | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | Returns `True` and `NVIDIA RTX 5000 Ada` | Script Ready |
| **G5** | **Dataset Mount Binding** | `export S1_DIR=...; export S2_DIR=...` in `.env` | Both paths resolve and contain GeoTIFFs | Script Ready |
| **G6** | **Full Splits & Stats** | `sbatch param_shavak/scripts/01_compute_full_dataset_stats.sh` | Generates official splits & training stats | Script Ready |
| **G7** | **Two-Tower Training** | `sbatch param_shavak/scripts/02_train_two_tower_hpc.sh` | Saves new checkpoint in `param_shavak/results/week_04/` | Script Ready |
| **G8** | **Embedding & Retrieval** | `sbatch param_shavak/scripts/03_extract_embeddings_and_eval.sh` | Builds FAISS indices, outputs cross-modal mAP | Script Ready |
| **G9** | **Results Isolation** | Check `ls -lh param_shavak/results/` | 100% of cluster outputs contained; 0 writes to `local_pc/` | Enforced |
