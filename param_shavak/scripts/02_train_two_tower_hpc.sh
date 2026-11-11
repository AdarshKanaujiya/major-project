#!/bin/bash
# ===========================================================================
# SLURM Job Script: 02 - Two-Tower Contrastive Training on PARAM SHAVAK
# Cluster: PARAM SHAVAK (PS-BOSS-OS)
# Node: GPU Compute Node (NVIDIA RTX 5000 Ada)
# ===========================================================================

#SBATCH --job-name=sentinel_train
#SBATCH --output=param_shavak/results/logs/train_%j.out
#SBATCH --error=param_shavak/results/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -e

echo "=== SENTINEL-RS: Step 02 - Two-Tower Contrastive Training ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Check GPU availability
nvidia-smi

export SENTINEL_ENV="param_shavak"
export SENTINEL_RS_ROOT="$(pwd)"
export S1_DIR="${S1_DIR:-/home/data/BigEarthNet-S1-Required}"
export S2_DIR="${S2_DIR:-/home/data/BigEarthNet-S2}"
export RESULTS_DIR="${SENTINEL_RS_ROOT}/param_shavak/results"
export BATCH_SIZE=64
export NUM_WORKERS=8
export PIN_MEMORY=1

mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/week_04/checkpoints"

source activate sentinel_rs || conda activate sentinel_rs

python param_shavak/week_04/training/train_two_tower_hpc.py

echo "Step 02 completed successfully at $(date)"

