#!/bin/bash
# =============================================================================
# Run CellFlow experiments (3 runs) - Our Method with Graph Fusion
# Conda environment: flow
# =============================================================================
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

CELLFLOW_DIR="/home/zhangshibo24s/cell_flow"
RUN_ID=${CELLFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
LOG_DIR="$CELLFLOW_DIR/results/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# GPU assignment (edit to match your machine)
GPU_CELLFLOW_ADDITIVE=${GPU_CELLFLOW_ADDITIVE:-0}
GPU_CELLFLOW_HOLDOUT=${GPU_CELLFLOW_HOLDOUT:-1}
GPU_CELLFLOW_LOCO=${GPU_CELLFLOW_LOCO:-2}

echo "=========================================="
echo "Starting CellFlow experiments (3 runs)"
echo "Conda env: flow"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate flow

# =============================================================================
# CellFlow (Our Method) - With Graph Fusion
# =============================================================================
echo ""
echo "=== Starting CellFlow (Our Method) ==="

echo "Starting: cellflow_norman_additive (GPU $GPU_CELLFLOW_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_ADDITIVE nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_additive.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_additive_${RUN_ID}" \
    > "$LOG_DIR/cellflow_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: cellflow_norman_holdout (GPU $GPU_CELLFLOW_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_HOLDOUT nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_holdout.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_holdout_${RUN_ID}" \
    > "$LOG_DIR/cellflow_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: cellflow_loco (GPU $GPU_CELLFLOW_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_LOCO nohup python "$CELLFLOW_DIR/train_cellflow_loco_new.py" \
    --output-dir "results/outputs/outputs_loco_${RUN_ID}" \
    > "$LOG_DIR/cellflow_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 CellFlow experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/cellflow_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
