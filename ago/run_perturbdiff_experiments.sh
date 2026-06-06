#!/bin/bash
# =============================================================================
# Run PerturbDiff experiments (3 runs)
# Conda environment: cmp_methods
# =============================================================================
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

MYFLOW_DIR="/home/zhangshibo24s/cell_flow"
COMPARISON_DIR="$MYFLOW_DIR/comparison_methods"
DATE=$(date +%Y%m%d_%H%M)
LOG_DIR="$MYFLOW_DIR/results/logs"
mkdir -p "$LOG_DIR"

# GPU assignment (edit to match your machine)
GPU_PERTURBDIFF_ADDITIVE=${GPU_PERTURBDIFF_ADDITIVE:-0}
GPU_PERTURBDIFF_HOLDOUT=${GPU_PERTURBDIFF_HOLDOUT:-1}
GPU_PERTURBDIFF_LOCO=${GPU_PERTURBDIFF_LOCO:-2}

echo "=========================================="
echo "Starting PerturbDiff experiments (3 runs)"
echo "Conda env: cmp_methods"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cmp_methods

# =============================================================================
# PerturbDiff
# =============================================================================
echo ""
echo "=== Starting PerturbDiff ==="

echo "Starting: perturbdiff_norman_additive (GPU $GPU_PERTURBDIFF_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_PERTURBDIFF_ADDITIVE nohup python "$COMPARISON_DIR/scripts/perturbdiff_norman_additive.py" \
    > "$LOG_DIR/perturbdiff_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: perturbdiff_norman_holdout (GPU $GPU_PERTURBDIFF_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_PERTURBDIFF_HOLDOUT nohup python "$COMPARISON_DIR/scripts/perturbdiff_norman_holdout.py" \
    > "$LOG_DIR/perturbdiff_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: perturbdiff_loco (GPU $GPU_PERTURBDIFF_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_PERTURBDIFF_LOCO nohup python "$COMPARISON_DIR/scripts/perturbdiff_loco.py" \
    > "$LOG_DIR/perturbdiff_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 PerturbDiff experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/perturbdiff_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
