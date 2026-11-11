#!/bin/bash
# ===========================================================================
# SLURM Job Script: 03 - Feature Extraction, FAISS Indexing & Evaluation
# Cluster: PARAM SHAVAK (PS-BOSS-OS)
# Node: GPU Compute Node (NVIDIA RTX 5000 Ada)
# ===========================================================================

#SBATCH --job-name=sentinel_eval
#SBATCH --output=param_shavak/results/logs/eval_%j.out
#SBATCH --error=param_shavak/results/logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

echo "=== SENTINEL-RS: Step 03 - Embeddings, FAISS & Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

nvidia-smi

export SENTINEL_ENV="param_shavak"
export SENTINEL_RS_ROOT="$(pwd)"
export S1_DIR="${S1_DIR:-/home/data/BigEarthNet-S1-Required}"
export S2_DIR="${S2_DIR:-/home/data/BigEarthNet-S2}"
export RESULTS_DIR="${SENTINEL_RS_ROOT}/param_shavak/results"

mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/week_05/trained_embeddings"
mkdir -p "${RESULTS_DIR}/week_05/faiss_indices"
mkdir -p "${RESULTS_DIR}/week_05/evaluation"

source activate sentinel_rs || conda activate sentinel_rs

python param_shavak/week_05/retrieval/run_full_evaluation_hpc.py

echo "Step 03 completed successfully at $(date)"

