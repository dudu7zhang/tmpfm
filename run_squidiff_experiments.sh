#!/bin/bash
# =============================================================================
# Run Squidiff experiments (3 runs)
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
GPU_SQUIDIFF_ADDITIVE=${GPU_SQUIDIFF_ADDITIVE:-0}
GPU_SQUIDIFF_HOLDOUT=${GPU_SQUIDIFF_HOLDOUT:-1}
GPU_SQUIDIFF_LOCO=${GPU_SQUIDIFF_LOCO:-2}

echo "=========================================="
echo "Starting Squidiff experiments (3 runs)"
echo "Conda env: cmp_methods"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cmp_methods

# =============================================================================
# Squidiff
# =============================================================================
echo ""
echo "=== Starting Squidiff ==="

echo "Starting: squidiff_norman_additive (GPU $GPU_SQUIDIFF_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_SQUIDIFF_ADDITIVE nohup python "$COMPARISON_DIR/scripts/squidiff_norman_additive.py" \
    > "$LOG_DIR/squidiff_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: squidiff_norman_holdout (GPU $GPU_SQUIDIFF_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_SQUIDIFF_HOLDOUT nohup python "$COMPARISON_DIR/scripts/squidiff_norman_holdout.py" \
    > "$LOG_DIR/squidiff_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: squidiff_loco (GPU $GPU_SQUIDIFF_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_SQUIDIFF_LOCO nohup python "$COMPARISON_DIR/scripts/squidiff_loco.py" \
    > "$LOG_DIR/squidiff_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 Squidiff experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/squidiff_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
