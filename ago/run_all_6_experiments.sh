#!/bin/bash
# =============================================================================
# Run ALL 6 MyFlow experiments (3 Baseline + 3 Our Method)
# Conda environment: flow
# =============================================================================
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
#
# GPU Layout (8x available: 4x 4090 + 4x 3090):
#   Baseline:  GPU 0/1/2
#   Our Method: GPU 3/4/5
# =============================================================================

set -e

MYFLOW_DIR="/home/zhangshibo24s/cell_flow"
RUN_ID=${MYFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export MYFLOW_RUN_ID=$RUN_ID
LOG_DIR="$MYFLOW_DIR/results/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# GPU assignment
GPU_BASELINE_ADDITIVE=${GPU_BASELINE_ADDITIVE:-0}
GPU_BASELINE_HOLDOUT=${GPU_BASELINE_HOLDOUT:-1}
GPU_BASELINE_LOCO=${GPU_BASELINE_LOCO:-2}
GPU_METHOD_ADDITIVE=${GPU_METHOD_ADDITIVE:-3}
GPU_METHOD_HOLDOUT=${GPU_METHOD_HOLDOUT:-4}
GPU_METHOD_LOCO=${GPU_METHOD_LOCO:-5}

echo "=========================================="
echo "Starting ALL 6 MyFlow experiments"
echo "Conda env: flow"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate flow

# =============================================================================
# Part 1: MyFlow Baseline (No Graph Fusion) - 3 experiments
# =============================================================================
echo ""
echo "=== [1/2] Starting MyFlow Baseline (3 experiments) ==="

echo "Starting: baseline_norman_additive (GPU $GPU_BASELINE_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_ADDITIVE nohup python "$MYFLOW_DIR/scripts/train_myflow_norman_scdfm_additive.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_additive \
    --output-dir "results/outputs/outputs_norman_baseline_additive_${RUN_ID}" \
    > "$LOG_DIR/baseline_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: baseline_norman_holdout (GPU $GPU_BASELINE_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_HOLDOUT nohup python "$MYFLOW_DIR/scripts/train_myflow_norman_scdfm_holdout.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_holdout \
    --output-dir "results/outputs/outputs_norman_baseline_holdout_${RUN_ID}" \
    > "$LOG_DIR/baseline_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: baseline_loco (GPU $GPU_BASELINE_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_LOCO nohup python "$MYFLOW_DIR/scripts/train_myflow_loco_new.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name loco_baseline \
    --output-dir "results/outputs/outputs_loco_baseline_${RUN_ID}" \
    > "$LOG_DIR/baseline_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# Part 2: MyFlow Our Method (With Graph Fusion) - 3 experiments
# =============================================================================
echo ""
echo "=== [2/2] Starting MyFlow Our Method (3 experiments) ==="

echo "Starting: method_norman_additive (GPU $GPU_METHOD_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_METHOD_ADDITIVE nohup python "$MYFLOW_DIR/scripts/train_myflow_norman_scdfm_additive.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_additive_${RUN_ID}" \
    > "$LOG_DIR/method_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: method_norman_holdout (GPU $GPU_METHOD_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_METHOD_HOLDOUT nohup python "$MYFLOW_DIR/scripts/train_myflow_norman_scdfm_holdout.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_holdout_${RUN_ID}" \
    > "$LOG_DIR/method_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: method_loco (GPU $GPU_METHOD_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_METHOD_LOCO nohup python "$MYFLOW_DIR/scripts/train_myflow_loco_new.py" \
    --output-dir "results/outputs/outputs_loco_${RUN_ID}" \
    > "$LOG_DIR/method_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 6 experiments launched!"
echo "=========================================="
echo ""
echo "Monitor all logs:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Monitor baseline only:"
echo "  tail -f $LOG_DIR/baseline_*.log"
echo ""
echo "Monitor method only:"
echo "  tail -f $LOG_DIR/method_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo "=========================================="
