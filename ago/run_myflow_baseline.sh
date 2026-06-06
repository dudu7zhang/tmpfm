#!/bin/bash
# =============================================================================
# Run MyFlow Baseline experiments (3 runs) - Without Graph Fusion
# Conda environment: flow
# =============================================================================
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

MYFLOW_DIR="/home/zhangshibo24s/cell_flow"
RUN_ID=${MYFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export MYFLOW_RUN_ID=$RUN_ID
LOG_DIR="$MYFLOW_DIR/results/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# GPU assignment (edit to match your machine)
GPU_BASELINE_ADDITIVE=${GPU_BASELINE_ADDITIVE:-4}
GPU_BASELINE_HOLDOUT=${GPU_BASELINE_HOLDOUT:-6}
GPU_BASELINE_LOCO=${GPU_BASELINE_LOCO:-7}

echo "=========================================="
echo "Starting MyFlow Baseline experiments (3 runs)"
echo "Conda env: flow"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate flow

# =============================================================================
# MyFlow Baseline - Without Graph Fusion
# =============================================================================
echo ""
echo "=== Starting MyFlow Baseline (No Graph Fusion) ==="

echo "Starting: myflow_baseline_norman_additive (GPU $GPU_BASELINE_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_ADDITIVE nohup python "$MYFLOW_DIR/scripts/train_myflow_norman_additive.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_additive \
    --output-dir "results/outputs/outputs_norman_baseline_additive_${RUN_ID}" \
    > "$LOG_DIR/myflow_baseline_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: myflow_baseline_norman_holdout (GPU $GPU_BASELINE_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_HOLDOUT nohup python "$MYFLOW_DIR/scripts/train_myflow_norman_holdout.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_holdout \
    --output-dir "results/outputs/outputs_norman_baseline_holdout_${RUN_ID}" \
    > "$LOG_DIR/myflow_baseline_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: myflow_baseline_loco (GPU $GPU_BASELINE_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_BASELINE_LOCO nohup python "$MYFLOW_DIR/scripts/train_myflow_loco_new.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name loco_baseline \
    --output-dir "results/outputs/outputs_loco_baseline_${RUN_ID}" \
    > "$LOG_DIR/myflow_baseline_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
echo ""
echo "=========================================="
echo "All 3 MyFlow Baseline experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/myflow_baseline_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
