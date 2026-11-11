#!/bin/bash
# ===========================================================================
# Master Execution Pipeline for PARAM SHAVAK
# Executes or submits Steps 01 -> 02 -> 03 sequentially
# ===========================================================================

set -e

echo "============================================================"
echo "SENTINEL-RS: PARAM SHAVAK Master HPC Execution Pipeline"
echo "============================================================"

# Submit with SLURM dependencies
JOB1=$(sbatch --parsable param_shavak/scripts/01_compute_full_dataset_stats.sh)
echo "Submitted Step 01 (Splits & Stats): Job ID ${JOB1}"

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} param_shavak/scripts/02_train_two_tower_hpc.sh)
echo "Submitted Step 02 (Two-Tower Training): Job ID ${JOB2} (depends on ${JOB1})"

JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} param_shavak/scripts/03_extract_embeddings_and_eval.sh)
echo "Submitted Step 03 (Embeddings & Evaluation): Job ID ${JOB3} (depends on ${JOB2})"

echo "All jobs successfully scheduled in SLURM queue."
echo "Monitor status with: squeue -u $(whoami)"

