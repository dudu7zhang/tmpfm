#!/bin/bash
# =============================================================================
# Run scDFM experiments (3 runs)
# Conda environment: cmp_methods
# =============================================================================
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

CELLFLOW_DIR="/home/zhangshibo24s/cell_flow"
COMPARISON_DIR="$CELLFLOW_DIR/comparison_methods"
DATE=$(date +%Y%m%d_%H%M)
LOG_DIR="$CELLFLOW_DIR/results/logs"
mkdir -p "$LOG_DIR"

# GPU assignment (edit to match your machine)
GPU_SCDFM_ADDITIVE=${GPU_SCDFM_ADDITIVE:-4}
GPU_SCDFM_HOLDOUT=${GPU_SCDFM_HOLDOUT:-5}
GPU_SCDFM_LOCO=${GPU_SCDFM_LOCO:-6}

echo "=========================================="
echo "Starting scDFM experiments (3 runs)"
echo "Conda env: cmp_methods"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cmp_methods

# =============================================================================
# scDFM
# =============================================================================
echo ""
echo "=== Starting scDFM ==="

echo "Starting: scdfm_norman_additive (GPU $GPU_SCDFM_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_SCDFM_ADDITIVE nohup python "$COMPARISON_DIR/scripts/scdfm_norman_additive.py" \
    > "$LOG_DIR/scdfm_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scdfm_norman_holdout (GPU $GPU_SCDFM_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_SCDFM_HOLDOUT nohup python "$COMPARISON_DIR/scripts/scdfm_norman_holdout.py" \
    > "$LOG_DIR/scdfm_norman_holdout.log" 2>&1 &
echo "  PID: $!"

# echo "Starting: scdfm_loco (GPU $GPU_SCDFM_LOCO)"
# CUDA_VISIBLE_DEVICES=$GPU_SCDFM_LOCO nohup python "$COMPARISON_DIR/scripts/scdfm_loco.py" \
#     > "$LOG_DIR/scdfm_loco.log" 2>&1 &
# echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 scDFM experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/scdfm_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
