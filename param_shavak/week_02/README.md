# Week 02 — Staged Verified Preprocessing & Data Loading for PARAM SHAVAK

**Status**: \AUDITED & LOCALLY VERIFIED — HPC FULL-DATASET ADAPTATION PENDING\

## Staged Implementations
15 corrected Week 2 implementations from \weeks/week_02/\ have been staged:
- **Preprocessing**: \preprocess_s2_adarsh.py\, \preprocess_s2_meet.py\, \preprocess_s2_jainam.py\, \preprocess_s2_fateh.py\, \sar_preprocessing_adarsh.py\, \sar_preprocessing_meet.py\, \sar_preprocessing_jainam.py\, \sar_preprocessing_fateh.py\.
- **DataLoader / Dataset**: \dataset_adarsh.py\, \dataloader_adarsh.py\, \paired_bigearthnet_dataset_meet.py\, \dataset_meet.py\, \dataloader_meet.py\, \dataset_fateh.py\, \dataloader_fateh.py\.

## Important Rules for PARAM SHAVAK Full Dataset
1. **Local-Subset vs. Complete Dataset**: The existing normalization statistics (\s1_stats.json\, \s2_stats.json\) were computed on the 21,000-patch local PC training split. They are NOT final full-dataset statistics.
2. **Recompute Statistics on HPC**: The next agent must recompute global population statistics across the complete full training split on PARAM SHAVAK.
3. **Preserve Processing Logic**: The verified logic (S2: B04/B03/B02 channel order, raw DN Z-score; S1: VV/VH channel order, [-25, 0] dB clipping, Z-score; paired 19-class multi-hot labeling) is scientifically validated and must be preserved.
