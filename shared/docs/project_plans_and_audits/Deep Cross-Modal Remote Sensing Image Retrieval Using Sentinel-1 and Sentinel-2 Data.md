**\# Cross-Modal Satellite Image Retrieval — Complete Project Plan**

**\#\# Team**

\| Person \| Name \| Primary Role \|

\|--------\|------\|-------------\|

\| Person 1 \| Jainam \| Project lead, data pipeline, FAISS retrieval evaluation \|

\| Person 2 \| Adarsh \| Metadata, S2 preprocessing, mAP evaluation, report \|

\| Person 3 \| Meet \| SAR pipeline, paired DataLoader, t-SNE \|

\| Person 4 \| Fateh \| Model architecture, training, embedding generation \|

\---

**\#\# Project Goal (from projectdetail.md)**

Build a retrieval system that:

\- Accepts a query satellite image from one modality (SAR or Multispectral)

\- Returns a **\*\*ranked list of top-5 and top-10\*\*** most relevant images from a gallery that may contain images from the same or different modality

\- Achieves this by learning a **\*\*common embedding space\*\*** where semantically similar images are close regardless of sensor modality

\- Reports Recall@1, Recall@5, Recall@10, mAP, and average retrieval time per query

\---

**\#\# Daily Standup Rule**

Every evening, each member sends **\*\*three lines\*\*** in the group chat:

1\. What I completed today

2\. What number or output I produced

3\. What I am doing tomorrow

No exceptions. This catches blockers early.

\---

**\#\# Week 1 — Environment, Data, and First Look**

**\*\*Goal: Everyone has the same working environment and can see the data.\*\***

**\*\*Jainam (Person 1\) — Project lead and environment setup\*\***

Day 1-2: Create the GitHub repository with a clear folder structure. Set up the shared virtual environment. Create a requirements.txt with all dependencies: torch, torchvision, pytorch-lightning, faiss-cpu, rasterio, numpy, pandas, matplotlib, scikit-learn, tqdm. Write a setup guide in README so all 4 members can replicate the environment exactly. Push to GitHub.

Day 3-4: Start downloading BigEarthNet-S2 (58GB Multispectral) first since it is larger and takes more time. While it downloads, read the BigEarthNet v2.0 Description Document. Understand the folder structure, naming conventions, and patch format before touching any data.

Day 5-7: Once S2 download is complete, write a simple patch viewer script that loads one GeoTIFF file using rasterio, extracts the RGB bands (B04, B03, B02 from Sentinel-2), and displays the image using matplotlib. Run it on 10 random patches. Visually confirm what the data looks like before writing any processing code.

**\*\*Adarsh (Person 2\) — Dataset documentation and metadata exploration\*\***

Day 1-2: Download the metadata.parquet file from the BigEarthNet website. This is small (a few MB) and downloads instantly. Open it with pandas and explore every column: patch\_id, labels, split, country, s1\_name, s2v1\_name, contains\_seasonal\_snow, contains\_cloud\_or\_shadow. Print the first 20 rows and share with the team.

Day 3-4: Load the metadata\_for\_patches\_with\_snow\_cloud\_or\_shadow.parquet file. Count how many patches are excluded (69,450). Confirm that metadata.parquet is already pre-filtered and does not overlap with the excluded set. Record this observation clearly.

Day 5-7: Analyze the label distribution across clean patches. BigEarthNet uses multi-label classification. Find the top 10 most common labels and the least frequent ones. Note the class imbalance (Beaches: 1,316 patches vs Arable land: 188,025). Write a one-page summary and share with the team. This informs the subset sampling strategy in Week 2\.

**\*\*Meet (Person 3\) — SAR download and format understanding\*\***

Day 1-2: Start downloading BigEarthNet-S1 (51GB SAR). While it downloads, read about Sentinel-1 SAR data format. Understand that SAR values can be in decibel scale (can be negative), that S1 has 2 channels (VV and VH polarization), stored as GeoTIFF files.

Day 3-4: Once download is complete or partially complete, write a script that loads one SAR GeoTIFF using rasterio and prints: shape, data type, min value, max value, and mean value for each channel. Run on 20 random SAR patches and record the statistics.

Day 5-7: Write a draft SAR preprocessing function. Clip values to the reasonable dB range (-25 to 0 dB for VV). Normalize to zero mean unit variance using the statistics collected. Test on 50 patches and visualize the VV channel as a grayscale image. Share result images with the team.

**\*\*Fateh (Person 4\) — Documentation and paper reading\*\***

Day 1-2: Set up the project documentation structure. Create a shared Google Doc titled "SENTINEL-RS Project Log" where all 4 members will record daily progress: what they did, what worked, what failed, and what numbers they got.

Day 3-5: Read and summarize two papers. First: BigEarthNet original paper (Sumbul et al., IGARSS 2019\) — dataset construction, patch format, benchmarks. Second: CMIR-NET (Pattern Recognition Letters 2020\) — Two-Tower architecture, contrastive loss, results. Write one-page summary of each.

Day 6-7: Write a minimal code snippet that loads one batch of paired SAR and Multispectral patches. Understand what a paired dataset looks like. Share what you learned with the team.

\---

**\#\# Week 2 — Preprocessing Pipeline and DataLoader**

**\*\*Goal: Clean, paired DataLoader returning correct batches. All 4 members can run it.\*\***

**\*\*Jainam (Person 1\) — Pair verification and subset sampling\*\***

Day 1-2: BigEarthNet guarantees that S1 and S2 patches are paired by naming convention. Write a verification script that checks every SAR patch in the clean metadata has a corresponding Multispectral patch with the same name. Count and report any missing pairs.

Day 3-4: Implement the subset sampling strategy. From the clean paired patches, sample 30,000 pairs that are class-balanced across the top 10 land-cover labels identified by Adarsh in Week 1\. Use stratified sampling so rare classes are not excluded. Save to subset\_30k.csv. Push to GitHub.

Day 5-7: Split the 30,000 pairs into train (21,000 \= 70%), validation (4,500 \= 15%), and test (4,500 \= 15%) sets respecting the official BigEarthNet split boundaries. Save train\_split.csv, val\_split.csv, test\_split.csv. Verify zero overlap across splits. Push to GitHub.

**\*\*Adarsh (Person 2\) — Multispectral preprocessing pipeline\*\***

Day 1-2: Write the complete S2 preprocessing function. Extract bands B04 (Red), B03 (Green), B02 (Blue) from the 13-channel GeoTIFF. Resize from 120x120 to 224x224 using bilinear interpolation for ResNet50 compatibility.

Day 3-4: Compute channel-wise mean and standard deviation for S2 RGB bands using ONLY the 21,000 training patches listed in train\_split.csv. Do NOT use the full 237,871 training set — only the 21K subset training patches to avoid leakage. Save to s2\_stats.json. Format: {"R": {"mean": X, "std": Y}, "G": ..., "B": ...}. Push to GitHub.

Day 5-7: Test the preprocessing on 100 random training patches. Verify: correct output shape (3, 224, 224), no NaN values, no all-black patches. Print actual min and max of normalized output — it should be approximately \[-3, \+4\] (standardized), NOT \[0, 1\]. Document this correctly.

**\*\*Meet (Person 3\) — SAR preprocessing pipeline and PyTorch Dataset\*\***

Day 1-2: Finalize the SAR preprocessing function. Compute training set mean and standard deviation for SAR VV and VH channels across all 21,000 training SAR patches from train\_split.csv. Save to s1\_stats.json. Format: {"VV": {"mean": X, "std": Y}, "VH": ...}.

Day 3-4: Write the PairedBigEarthNetDataset PyTorch Dataset class that accepts a CSV split file, loads paired SAR and Multispectral patches, applies correct preprocessing to each modality, and returns (sar\_tensor, ms\_tensor, label\_tensor, patch\_id). SAR shape: (2, 224, 224). MS shape: (3, 224, 224).

Day 5-7: Build PyTorch DataLoaders with batch\_size=64, num\_workers=0 (Windows), shuffle=True for training and False for val/test. Run one full epoch of the training DataLoader without errors. Measure and report time per epoch.

**\*\*Fateh (Person 4\) — Baseline model setup and FAISS introduction\*\***

Day 1-2: Write the baseline feature extractor using pretrained ResNet50. Remove the final FC layer (output: 2048-dim). For the SAR extractor: replace first Conv2d(3,64,...) with Conv2d(2,64,...), copying pretrained weights for first 2 channels. Test with synthetic torch.randn() inputs — no real data needed.

Day 3-4: Install FAISS (pip install faiss-cpu). Write a minimal FAISS test: create 1000 random 512-dim vectors, build a flat L2 index, query with one vector, return top-10 results, measure query time in milliseconds. Confirm it works.

Day 5-7: Write the embedding generation script. Get the 4,500 val S2 patch folders from Jainam (\~500MB via Google Drive — NOT the full 58GB). Run all val patches through the S2 baseline extractor. Save as baseline\_val\_ms\_embeddings.npy and baseline\_val\_labels.npy. Record total time.

\---

**\#\# Week 3 — Baseline Evaluation and Visual Proof of the Problem**

**\*\*Goal: Baseline numbers in hand. Visual proof of modality gap. Team aligned on what the model must fix.\*\***

**\*\*Jainam (Person 1\) — Baseline FAISS retrieval and Recall@K\*\***

Day 1-2: Load baseline\_val\_ms\_embeddings.npy generated by Fateh. Build a FAISS flat L2 index on the 4,500 Multispectral gallery embeddings. For each SAR embedding in the query set, search top-10 nearest Multispectral embeddings. Record retrieved patch IDs.

Day 3-4: Implement Recall@K evaluation. For each query SAR patch, ground truth is its paired MS patch (same location). Compute Recall@1, Recall@5, Recall@10. Average across all 4,500 validation queries. Record as the "before contrastive training" baseline numbers.

Day 5-7: Run same-modal evaluation. SAR-to-SAR: retrieve top-10 SAR patches sharing the same land-cover label. MS-to-MS: same. Compute Recall@K for both. Compare same-modal vs cross-modal — the gap is the problem being solved.

**\*\*Adarsh (Person 2\) — mAP computation and baseline analysis\*\***

Day 1-2: Fix s2\_stats.json: rerun compute\_stats.py using only the 21,000 patch IDs from train\_split.csv (not the full 237,871). Push corrected s2\_stats.json to GitHub. Confirm the new stats differ from the old ones and update person2task.md.

Day 3-4: Implement mean Average Precision (mAP) computation. For each query, rank retrieved results by similarity. Check if retrieved label overlaps with query label (multi-label). Compute Average Precision per query. Average across all queries.

Day 5-7: Analyze 20 specific failure cases where the baseline retrieves wrong images. For each: show query SAR image and wrongly retrieved MS image. Write one-paragraph analysis of failure patterns. Build the baseline results table (rows: SAR-to-SAR, MS-to-MS, SAR-to-MS, MS-to-SAR; columns: Recall@1, Recall@5, Recall@10, mAP, avg retrieval time ms).

**\*\*Meet (Person 3\) — t-SNE visualization and retrieval time measurement\*\***

Day 1-2: Generate t-SNE of baseline embeddings. Take 2,000 embeddings (1,000 SAR \+ 1,000 MS) covering 5 land-cover classes. Run t-SNE: reduce 2048 → 2 dimensions. Plot: SAR \= red, MS \= blue, marker shape \= land-cover class.

Day 3-4: The t-SNE will show two completely separate clusters (SAR cluster and MS cluster) even for the same locations. This is the visual proof of the domain gap. Save as outputs/plots/baseline\_tsne.png. This is the "before" picture for the report and presentation.

Day 5-7: Measure retrieval time per query. Time each FAISS query individually. Compute mean, median, 95th percentile in milliseconds. Also time embedding generation separately. Record all numbers.

**\*\*Fateh (Person 4\) — Two-Tower architecture and InfoNCE loss\*\***

Day 1-2: Implement TwoTowerNetwork in src/models/two\_tower.py. SAR encoder (2-channel ResNet50) \+ MS encoder (standard ResNet50) \+ projection heads (2048 → 1024 → 512). Test with synthetic tensors: input (4, 2, 224, 224\) and (4, 3, 224, 224\) → output (4, 512\) and (4, 512).

Day 3-4: Implement InfoNCELoss (NT-Xent). L2-normalize embeddings. Build (B x B) cosine similarity matrix. Divide by temperature=0.07. Cross-entropy loss in both directions (SAR→MS and MS→SAR), average. Sanity check: loss on random embeddings ≈ log(batch\_size) ≈ 4.16 for B=64.

Day 5-7: Write PyTorch Lightning RetrievalModule. Wraps TwoTowerNetwork \+ InfoNCELoss. Implement training\_step, validation\_step, configure\_optimizers (Adam lr=1e-4, CosineAnnealingLR). Sanity check: run 3 epochs on 200 synthetic pairs. Confirm loss decreases. Push all code to GitHub.

\---

**\#\# End of Week 3 — What You Must Have**

Every item below must exist and be pushed to GitHub before Week 4 starts.

\- subset\_30k.csv, train\_split.csv, val\_split.csv, test\_split.csv — verified, zero overlap

\- s2\_stats.json recomputed from 21K train patches (corrected)

\- s1\_stats.json computed from 21K train SAR patches

\- PairedBigEarthNetDataset running without error, correct shapes confirmed

\- Fateh's baseline extractor code pushed

\- baseline\_val\_ms\_embeddings.npy saved (4500 x 2048\)

\- FAISS index built and queried successfully

\- Baseline Recall@K and mAP numbers recorded

\- baseline\_tsne.png saved showing SAR/MS as two separate clusters

\- TwoTowerNetwork \+ InfoNCELoss implemented and sanity-checked

\- PyTorch Lightning training loop confirmed to decrease loss over 3 epochs

\---

**\#\# Week 4 — Training the Two-Tower Model**

**\*\*Goal: Trained model checkpoint saved. Training loss confirmed to decrease. Validation metrics trending upward.\*\***

**\*\*Jainam (Person 1\) — Training infrastructure and checkpoint management\*\***

Day 1-2: Set up the full training script (scripts/train.py) using PyTorch Lightning Trainer. Configure: max\_epochs=20, batch\_size=64, GPU if available (accelerator="gpu" or "auto"). Set checkpoint callback to save best model by val/loss. Set up logging with TensorBoard or CSV logger.

Day 3-4: Launch training on the 21,000 training pairs. Monitor loss curves every epoch. Expected: loss starts around 4.1 (log 64\) and decreases to below 2.0 by epoch 10 if training is working. If loss is not decreasing by epoch 3, stop and debug. Common issues: learning rate too high, batch size too small, incorrect normalization.

Day 5-7: Run training to completion. Save the best model checkpoint (best\_model.ckpt). Record: final training loss, final validation loss, total training time in hours. Push checkpoint path and training log to GitHub.

**\*\*Adarsh (Person 2\) — Training monitoring and analysis\*\***

Day 1-2: Set up training metrics logging. Write a script that reads the training log (CSV or TensorBoard) and plots train/val loss curves in real time. Share the loss curve plot with the team at the end of each day.

Day 3-4: Monitor for overfitting signs. If val loss stops decreasing while train loss continues, training is overfitting. Note the epoch where this happens. If overfitting occurs before epoch 15, suggest adding dropout to projection heads (dropout=0.1) and restarting training.

Day 5-7: Once training completes, generate the training summary. Plot final train/val loss curves. Record the epoch at which the best val loss was achieved. Write a one-paragraph training analysis for the report.

**\*\*Meet (Person 3\) — Trained embedding generation\*\***

Day 1-2: Once best\_model.ckpt is available from Jainam, write scripts/generate\_trained\_embeddings.py. Load the trained TwoTowerNetwork (not the baseline extractor). Run all 4,500 validation SAR patches through the SAR encoder. Run all 4,500 validation MS patches through the MS encoder. Save as trained\_val\_sar\_embeddings.npy and trained\_val\_ms\_embeddings.npy.

Day 3-4: Also generate embeddings for the full test set (4,500 pairs). Save as trained\_test\_sar\_embeddings.npy and trained\_test\_ms\_embeddings.npy. These are the final embeddings used for all evaluation in Week 5\.

Day 5-7: Generate the "after" t-SNE plot. Use 2,000 trained embeddings (1,000 SAR \+ 1,000 MS) same classes as the baseline t-SNE. Run t-SNE. Plot same way (SAR=red, MS=blue). Save as outputs/plots/trained\_tsne.png. Compare visually with baseline\_tsne.png — the clusters should now be interleaved (SAR and MS mixed together by class).

**\*\*Fateh (Person 4\) — Model architecture experiments\*\***

Day 1-4: While base training runs, implement one architecture improvement: add hard negative mining to the InfoNCE loss. Instead of using all N-1 negatives equally, mine the hardest negatives (the most similar incorrect pairs in each batch). This improves training signal. Implement HardNegativeInfoNCELoss and compare with standard InfoNCELoss.

Day 5-7: Run a second training run with HardNegativeInfoNCELoss on the same dataset. Compare val loss curves between standard and hard-negative versions. If hard-negative version gives lower val loss, keep it. Report comparison to team.

\---

**\#\# Week 5 — Full Evaluation and Retrieval System**

**\*\*Goal: Final retrieval numbers on test set. "After" t-SNE ready. Full comparison table complete.\*\***

**\*\*Jainam (Person 1\) — Final FAISS retrieval and test set evaluation\*\***

Day 1-2: Build FAISS flat index on the 4,500 trained test MS embeddings (gallery). For each of the 4,500 SAR test queries, search top-10 nearest MS embeddings. Record all results. This is the cross-modal SAR→MS retrieval evaluation on the held-out test set.

Day 3-4: Compute Recall@1, Recall@5, Recall@10 on the test set for all four retrieval modes: SAR→MS, MS→SAR, SAR→SAR, MS→MS. Compare with the Week 3 baseline numbers. The improvement is the core result of the project.

Day 5-7: Measure final retrieval time per query on the test set. Compute mean, median, 95th percentile. Build a FAISS IVF index (approximate nearest neighbour) and compare query time with the flat index. Report whether the speed improvement is worth the accuracy tradeoff.

**\*\*Adarsh (Person 2\) — mAP on test set and final results table\*\***

Day 1-2: Compute mAP on the test set using the trained embeddings. Follow the same procedure as Week 3 but using test\_split.csv and trained embeddings. Record the final mAP number.

Day 3-4: Build the final comparison table. Two sections: Baseline (before training) and Trained (after contrastive learning). Rows: SAR-to-SAR, MS-to-MS, SAR-to-MS, MS-to-SAR. Columns: Recall@1, Recall@5, Recall@10, mAP, avg retrieval time (ms). This table is the single most important result in the report.

Day 5-7: Write a 3-paragraph result analysis for the report. Paragraph 1: what the baseline numbers showed. Paragraph 2: how the trained model improved each metric and why. Paragraph 3: where the system still fails and what future work could address it.

**\*\*Meet (Person 3\) — Visual retrieval results and qualitative analysis\*\***

Day 1-2: Build a visual retrieval demo. For 10 query SAR images, show: the query SAR image, top-5 retrieved MS images (baseline), top-5 retrieved MS images (trained). Visualize as a grid. This is the qualitative evidence that the system works.

Day 3-4: Pick 5 success cases and 5 failure cases from the test set. For successes: show query \+ top-1 correct retrieval. For failures: show query \+ top-1 wrong retrieval \+ the correct image that was missed. Write a short caption for each explaining why it succeeded or failed.

Day 5-7: Produce the final version of both t-SNE plots (baseline and trained) as publication-quality figures. Same color scheme, same number of points, same classes. Put them side by side in a single figure. Save as outputs/plots/tsne\_comparison.png.

**\*\*Fateh (Person 4\) — Retrieval demo and model export\*\***

Day 1-2: Build an interactive demo script (scripts/demo.py) that accepts a SAR image path as input, runs it through the trained SAR encoder, searches the pre-built FAISS index, and prints/displays the top-5 retrieved MS images with their similarity scores.

Day 3-5: Export the trained model to a usable format. Save the FAISS index as faiss\_index.bin. Save the trained encoder weights separately for SAR and MS. Write a requirements file for the demo. Test that the demo works cleanly on a fresh environment.

Day 6-7: Profile the full end-to-end pipeline. Time each step: image loading → preprocessing → embedding generation → FAISS search → display. Report total latency and identify the bottleneck step.

\---

**\#\# Week 6 — Report, Presentation, and Submission**

**\*\*Goal: IEEE-format paper submitted. Presentation slides ready. Code repository clean and documented.\*\***

**\*\*Jainam (Person 1\) — Abstract, Introduction, Experiments section\*\***

Day 1-2: Write the Abstract (250 words). Must include: problem statement, dataset used, method, key results (final Recall@10 and mAP), conclusion. Write the Introduction (1 page). Cover: why cross-modal retrieval is important, what the modality gap problem is, what this project does differently.

Day 3-4: Write the Experiments section. Include: dataset description (BigEarthNet, 30K subset), train/val/test split sizes, evaluation metrics used (Recall@K, mAP, retrieval time), and hardware used (CPU/GPU, RAM).

Day 5-7: Final code cleanup. Ensure all scripts have docstrings and comments. Ensure README has complete setup and run instructions. Tag the final commit as v1.0 on GitHub. Ensure all output files (.npy, .csv, .json, .png, .ckpt) are listed with download links or generation instructions.

**\*\*Adarsh (Person 2\) — Related Work and Results section, report editing\*\***

Day 1-2: Write the Related Work section (1 page). Cover: BigEarthNet dataset paper, contrastive learning (SimCLR or MoCo for context), cross-modal retrieval in remote sensing, FAISS for efficient search. Cite at least 8 papers.

Day 3-4: Write the Results section. Insert the final comparison table. Insert the t-SNE comparison figure. Insert the visual retrieval results grid. Write explanatory text for each figure and table. This is the most important section of the paper.

Day 5-7: Do a full editorial pass over the entire paper. Fix grammar, ensure consistent notation, check all figure captions, verify all numbers in the paper match the actual computed results. This is the person who ensures the paper is submission-ready.

**\*\*Meet (Person 3\) — Methodology section and presentation slides\*\***

Day 1-2: Write the Methodology section (1.5 pages). Cover: preprocessing pipeline for S2 and S1, Two-Tower network architecture diagram, InfoNCE loss formulation, FAISS indexing strategy. Include an architecture diagram figure.

Day 3-4: Build the presentation slides (15-20 slides). Structure: title → problem statement → dataset → method → baseline results (with t-SNE "before") → trained results (with t-SNE "after") → comparison table → demo → conclusion. Use the figures and tables already produced.

Day 5-7: Practice the presentation as a team. Each person presents their section. Time the total: aim for 12-15 minutes. Identify weak slides and improve them.

**\*\*Fateh (Person 4\) — Conclusion, Future Work, and demo video\*\***

Day 1-2: Write the Conclusion (half page). Summarize: what was built, what results were achieved, what the system can do. Write the Future Work section (half page). Suggest: using Vision Transformers instead of ResNet50, adding optical-to-SAR retrieval with a third modality, scaling to the full 480K dataset.

Day 3-4: Record a 2-3 minute demo video showing the retrieval system in action. Show: a query SAR image input → top-5 retrieved MS images output → repeat for 3 different query types. This goes in the final submission and GitHub README.

Day 5-7: Final submission packaging. Zip the code, ensure all dependencies are in requirements.txt, write a one-page README for the examiner explaining how to run the demo in under 5 minutes. Submit.

\---

**\#\# Master Deliverables Checklist**

**\#\#\# Data & Preprocessing**

\- \[ \] subset\_30k.csv — 30K class-balanced paired patches

\- \[ \] train\_split.csv, val\_split.csv, test\_split.csv — no overlap verified

\- \[ \] s2\_stats.json — computed from 21K train patches only (corrected)

\- \[ \] s1\_stats.json — computed from 21K train SAR patches

\- \[ \] PairedBigEarthNetDataset verified on val set

**\#\#\# Baseline (Week 3\)**

\- \[ \] baseline\_val\_ms\_embeddings.npy

\- \[ \] baseline\_tsne.png — showing modality gap

\- \[ \] Baseline Recall@1/5/10, mAP, retrieval time recorded

**\#\#\# Training (Week 4\)**

\- \[ \] best\_model.ckpt — best checkpoint by val loss

\- \[ \] Training loss curves saved

\- \[ \] trained\_val\_sar\_embeddings.npy, trained\_val\_ms\_embeddings.npy

\- \[ \] trained\_test\_sar\_embeddings.npy, trained\_test\_ms\_embeddings.npy

\- \[ \] trained\_tsne.png — showing modality gap is closed

**\#\#\# Evaluation (Week 5\)**

\- \[ \] Final Recall@1/5/10, mAP, retrieval time on test set

\- \[ \] Full comparison table (baseline vs trained, all 4 retrieval modes)

\- \[ \] Visual retrieval grid (10 queries, top-5 results)

\- \[ \] tsne\_comparison.png — baseline vs trained side by side

\- \[ \] faiss\_index.bin built and saved

\- \[ \] demo.py working end-to-end

**\#\#\# Report & Submission (Week 6\)**

\- \[ \] IEEE-format paper — all sections complete

\- \[ \] Presentation slides (15-20 slides)

\- \[ \] Demo video (2-3 minutes)

\- \[ \] GitHub repository cleaned, v1.0 tagged

\- \[ \] README with 5-minute setup and run instructions

\---

**\#\# Critical Path — What Blocks What**

\`\`\`

Week 1: Download data (Person 1, 3\) → everything else

Week 2: train\_split.csv (Person 1\) → s2\_stats.json (Person 2\) and s1\_stats.json (Person 3\)

Week 2: Paired Dataset (Person 3\) \+ Baseline Extractor (Person 4\) → embeddings

Week 3: Embeddings → FAISS eval (Person 1\) → Recall@K, mAP

Week 3: TwoTower \+ InfoNCE (Person 4\) → Training (Week 4\)

Week 4: best\_model.ckpt → trained embeddings (Person 3\) → final eval (Week 5\)

Week 5: All results → Report \+ Slides (Week 6\)

\`\`\`

The single most dangerous bottleneck is **\*\*Week 2 Person 3 (Meet)\*\*** — the PairedBigEarthNetDataset. If it is late, training cannot start.

