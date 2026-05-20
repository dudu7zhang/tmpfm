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
GPU_CELLFLOW_HOLDOUT=${GPU_CELLFLOW_HOLDOUT:-4}
GPU_CELLFLOW_LOCO=${GPU_CELLFLOW_LOCO:-5}

echo "=========================================="
echo "Starting MyFlow experiments (3 runs)"
echo "Conda env: flow"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate flow

# =============================================================================
# MyFlow (Our Method) - With Graph Fusion
# =============================================================================
echo ""
echo "=== Starting MyFlow (Our Method) ==="

# echo "Starting: myflow_norman_additive (GPU $GPU_CELLFLOW_ADDITIVE)"
# CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_ADDITIVE nohup python "$CELLFLOW_DIR/train_cellflow_norman_additive.py" \
#     --output-dir "results/outputs/outputs_myflow_norman_additive_${RUN_ID}" \
#     > "$LOG_DIR/myflow_norman_additive.log" 2>&1 &
# echo "  PID: $!"

# echo "Starting: myflow_norman_holdout (GPU $GPU_CELLFLOW_HOLDOUT)"
# CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_HOLDOUT nohup python "$CELLFLOW_DIR/train_cellflow_norman_holdout.py" \
#     --output-dir "results/outputs/outputs_myflow_norman_holdout_${RUN_ID}" \
#     > "$LOG_DIR/myflow_norman_holdout.log" 2>&1 &
# echo "  PID: $!"

echo "Starting: myflow_loco (GPU $GPU_CELLFLOW_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_LOCO nohup python "$CELLFLOW_DIR/train_cellflow_loco_new.py" \
    --train-cell-fraction 0.15 \
    --num-iterations 15000 \
    --output-dir "results/outputs/outputs_myflow_loco_${RUN_ID}" \
    > "$LOG_DIR/myflow_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 CellFlow experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/myflow_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
