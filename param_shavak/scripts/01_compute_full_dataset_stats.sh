#!/bin/bash
# ===========================================================================
# SLURM Job Script: 01 - Compute Full Dataset Splits & Population Stats
# Cluster: PARAM SHAVAK (PS-BOSS-OS)
# Node: GPU Compute Node (NVIDIA RTX 5000 Ada)
# ===========================================================================

#SBATCH --job-name=sentinel_stats
#SBATCH --output=param_shavak/results/logs/stats_%j.out
#SBATCH --error=param_shavak/results/logs/stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

echo "=== SENTINEL-RS: Step 01 - Full Dataset Splits & Stats ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Environment settings
export SENTINEL_ENV="param_shavak"
export SENTINEL_RS_ROOT="$(pwd)"

# Adjust dataset paths to cluster storage mount
export S1_DIR="${S1_DIR:-/home/data/BigEarthNet-S1-Required}"
export S2_DIR="${S2_DIR:-/home/data/BigEarthNet-S2}"
export RESULTS_DIR="${SENTINEL_RS_ROOT}/param_shavak/results"

mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/week_02/splits"
mkdir -p "${RESULTS_DIR}/week_02/normalization_stats"

# Activate environment
source activate sentinel_rs || conda activate sentinel_rs

python param_shavak/scripts/generate_full_dataset_splits_and_stats.py

echo "Step 01 completed successfully at $(date)"

