#!/bin/bash
# =============================================================================
# Run MyFlow experiments (6 experiments)
# Conda environment: flow
# =============================================================================
# Methods:
#   1. MyFlow (with graph fusion)
#   2. MyFlow baseline (without graph fusion)
#
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

MYFLOW_DIR="/home/zhangshibo24s/cell_flow"
DATE=$(date +%Y%m%d_%H%M)
RUN_ID=${MYFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export MYFLOW_RUN_ID=$RUN_ID
LOG_DIR="$MYFLOW_DIR/results/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# Per-experiment GPU assignment (edit these to match your machine)
GPU_MYFLOW_ADDITIVE=0
GPU_MYFLOW_HOLDOUT=1
GPU_MYFLOW_LOCO=2
GPU_BASELINE_ADDITIVE=3
GPU_BASELINE_HOLDOUT=0
GPU_BASELINE_LOCO=1

echo "=========================================="
echo "Starting MyFlow experiments (6 runs)"
echo "Conda env: flow"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source activate flow || conda activate flow

# =============================================================================
# MyFlow (Our Method) - With Graph Fusion
# =============================================================================
echo ""
echo "=== Starting MyFlow (Our Method) ==="

echo "Starting: myflow_norman_additive (GPU $GPU_MYFLOW_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_MYFLOW_ADDITIVE nohup python "$MYFLOW_DIR/train_myflow_norman_scdfm_additive.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_additive_${RUN_ID}" \
    > "$LOG_DIR/myflow_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: myflow_norman_holdout (GPU $GPU_MYFLOW_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_MYFLOW_HOLDOUT nohup python "$MYFLOW_DIR/train_myflow_norman_scdfm_holdout.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_holdout_${RUN_ID}" \
    > "$LOG_DIR/myflow_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: myflow_loco (GPU $GPU_MYFLOW_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_MYFLOW_LOCO nohup python "$MYFLOW_DIR/train_myflow_loco_new.py" \
    --output-dir "results/outputs/outputs_loco_${RUN_ID}" \
    > "$LOG_DIR/myflow_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# MyFlow Baseline - Without Graph Fusion
# =============================================================================
echo ""
echo "=== Starting MyFlow Baseline (No Graph Fusion) ==="

echo "Starting: myflow_baseline_norman_additive (GPU $GPU_BASELINE_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_ADDITIVE nohup python "$MYFLOW_DIR/train_myflow_norman_scdfm_additive.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_additive \
    --output-dir "results/outputs/outputs_norman_baseline_additive_${RUN_ID}" \
    > "$LOG_DIR/myflow_baseline_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: myflow_baseline_norman_holdout (GPU $GPU_BASELINE_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_HOLDOUT nohup python "$MYFLOW_DIR/train_myflow_norman_scdfm_holdout.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_holdout \
    --output-dir "results/outputs/outputs_norman_baseline_holdout_${RUN_ID}" \
    > "$LOG_DIR/myflow_baseline_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: myflow_baseline_loco (GPU $GPU_BASELINE_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_LOCO nohup python "$MYFLOW_DIR/train_myflow_loco_new.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name loco_baseline \
    --output-dir "results/outputs/outputs_loco_baseline_${RUN_ID}" \
    > "$LOG_DIR/myflow_baseline_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 6 MyFlow experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/myflow_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
