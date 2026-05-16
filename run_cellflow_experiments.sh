#!/bin/bash
# =============================================================================
# Run CellFlow experiments (6 experiments)
# Conda environment: flow
# =============================================================================
# Methods:
#   1. CellFlow (with graph fusion)
#   2. CellFlow baseline (without graph fusion)
#
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

CELLFLOW_DIR="/home/zhangshibo24s/cell_flow"
DATE=$(date +%Y%m%d_%H%M)
LOG_DIR="$CELLFLOW_DIR/logs_all_experiments_${DATE}"
mkdir -p "$LOG_DIR"

# Per-experiment GPU assignment (edit these to match your machine)
GPU_CELLFLOW_ADDITIVE=0
GPU_CELLFLOW_HOLDOUT=1
GPU_CELLFLOW_LOCO=2
GPU_BASELINE_ADDITIVE=3
GPU_BASELINE_HOLDOUT=0
GPU_BASELINE_LOCO=1

echo "=========================================="
echo "Starting CellFlow experiments (6 runs)"
echo "Conda env: flow"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source activate flow || conda activate flow

# =============================================================================
# CellFlow (Our Method) - With Graph Fusion
# =============================================================================
echo ""
echo "=== Starting CellFlow (Our Method) ==="

echo "Starting: cellflow_norman_additive (GPU $GPU_CELLFLOW_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_ADDITIVE nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_additive.py" \
    > "$LOG_DIR/cellflow_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: cellflow_norman_holdout (GPU $GPU_CELLFLOW_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_HOLDOUT nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_holdout.py" \
    > "$LOG_DIR/cellflow_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: cellflow_loco (GPU $GPU_CELLFLOW_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW_LOCO nohup python "$CELLFLOW_DIR/train_cellflow_loco_new.py" \
    > "$LOG_DIR/cellflow_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# CellFlow Baseline - Without Graph Fusion
# =============================================================================
echo ""
echo "=== Starting CellFlow Baseline (No Graph Fusion) ==="

echo "Starting: cellflow_baseline_norman_additive (GPU $GPU_BASELINE_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_ADDITIVE nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_additive.py" \
    --no-x-graph-fusion-enabled \
    --run-name norman_baseline_additive \
    --output-dir outputs_norman_baseline_additive_${DATE} \
    > "$LOG_DIR/cellflow_baseline_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: cellflow_baseline_norman_holdout (GPU $GPU_BASELINE_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_HOLDOUT nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_holdout.py" \
    --no-x-graph-fusion-enabled \
    --run-name norman_baseline_holdout \
    --output-dir outputs_norman_baseline_holdout_${DATE} \
    > "$LOG_DIR/cellflow_baseline_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: cellflow_baseline_loco (GPU $GPU_BASELINE_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_LOCO nohup python "$CELLFLOW_DIR/train_cellflow_loco_new.py" \
    --no-x-graph-fusion-enabled \
    --run-name loco_baseline \
    --output-dir outputs_loco_baseline_${DATE} \
    > "$LOG_DIR/cellflow_baseline_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 6 CellFlow experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/cellflow_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
