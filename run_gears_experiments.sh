#!/bin/bash
# =============================================================================
# Run GEARS experiments (3 runs)
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
GPU_GEARS_ADDITIVE=${GPU_GEARS_ADDITIVE:-0}
GPU_GEARS_HOLDOUT=${GPU_GEARS_HOLDOUT:-1}
GPU_GEARS_LOCO=${GPU_GEARS_LOCO:-2}

echo "=========================================="
echo "Starting GEARS experiments (3 runs)"
echo "Conda env: cmp_methods"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cmp_methods

# =============================================================================
# GEARS
# =============================================================================
echo ""
echo "=== Starting GEARS ==="

echo "Starting: gears_norman_additive (GPU $GPU_GEARS_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_GEARS_ADDITIVE nohup python "$COMPARISON_DIR/scripts/gears_norman_additive.py" \
    > "$LOG_DIR/gears_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: gears_norman_holdout (GPU $GPU_GEARS_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_GEARS_HOLDOUT nohup python "$COMPARISON_DIR/scripts/gears_norman_holdout.py" \
    > "$LOG_DIR/gears_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: gears_loco (GPU $GPU_GEARS_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_GEARS_LOCO nohup python "$COMPARISON_DIR/scripts/gears_loco.py" \
    > "$LOG_DIR/gears_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 GEARS experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/gears_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
