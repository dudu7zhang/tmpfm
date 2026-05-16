#!/bin/bash
# =============================================================================
# Run all 18 experiments: 6 methods × 3 data splits
# =============================================================================
# Methods:
#   1. CellFlow (our method, with graph fusion)
#   2. CellFlow-baseline (without graph fusion)
#   3. GEARS
#   4. scDFM
#   5. PerturbDiff
#   6. Squidiff
#
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e  # Exit on error

# Configuration
CELLFLOW_DIR="/home/zhangshibo24s/cell_flow"
COMPARISON_DIR="$CELLFLOW_DIR/comparison_methods"
DATE=$(date +%Y%m%d_%H%M)
RUN_ID=${CELLFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
LOG_DIR="$CELLFLOW_DIR/results/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# Default GPU (can be overridden by environment variable)
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Starting all 18 experiments"
echo "GPU: $GPU_ID"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# =============================================================================
# CellFlow (Our Method) - With Graph Fusion
# =============================================================================
echo ""
echo "=== Starting CellFlow (Our Method) ==="

# CellFlow Norman Additive
echo "Starting: cellflow_norman_additive"
nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_additive.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_additive_${RUN_ID}" \
    > "$LOG_DIR/cellflow_norman_additive.log" 2>&1 &
echo "  PID: $!"

# CellFlow Norman Holdout
echo "Starting: cellflow_norman_holdout"
nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_holdout.py" \
    --output-dir "results/outputs/outputs_norman_scdfm_holdout_${RUN_ID}" \
    > "$LOG_DIR/cellflow_norman_holdout.log" 2>&1 &
echo "  PID: $!"

# CellFlow LOCO
echo "Starting: cellflow_loco"
nohup python "$CELLFLOW_DIR/train_cellflow_loco_new.py" \
    --output-dir "results/outputs/outputs_loco_${RUN_ID}" \
    > "$LOG_DIR/cellflow_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# CellFlow Baseline - Without Graph Fusion
# =============================================================================
echo ""
echo "=== Starting CellFlow Baseline (No Graph Fusion) ==="

# CellFlow Baseline Norman Additive
echo "Starting: cellflow_baseline_norman_additive"
nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_additive.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_additive \
    --output-dir "results/outputs/outputs_norman_baseline_additive_${RUN_ID}" \
    > "$LOG_DIR/cellflow_baseline_norman_additive.log" 2>&1 &
echo "  PID: $!"

# CellFlow Baseline Norman Holdout
echo "Starting: cellflow_baseline_norman_holdout"
nohup python "$CELLFLOW_DIR/train_cellflow_norman_scdfm_holdout.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name norman_baseline_holdout \
    --output-dir "results/outputs/outputs_norman_baseline_holdout_${RUN_ID}" \
    > "$LOG_DIR/cellflow_baseline_norman_holdout.log" 2>&1 &
echo "  PID: $!"

# CellFlow Baseline LOCO
echo "Starting: cellflow_baseline_loco"
nohup python "$CELLFLOW_DIR/train_cellflow_loco_new.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name loco_baseline \
    --output-dir "results/outputs/outputs_loco_baseline_${RUN_ID}" \
    > "$LOG_DIR/cellflow_baseline_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# Comparison Methods - GEARS
# =============================================================================
echo ""
echo "=== Starting GEARS ==="

echo "Starting: gears_norman_additive"
nohup python "$COMPARISON_DIR/scripts/gears_norman_additive.py" \
    > "$LOG_DIR/gears_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: gears_norman_holdout"
nohup python "$COMPARISON_DIR/scripts/gears_norman_holdout.py" \
    > "$LOG_DIR/gears_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: gears_loco"
nohup python "$COMPARISON_DIR/scripts/gears_loco.py" \
    > "$LOG_DIR/gears_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# Comparison Methods - scDFM
# =============================================================================
echo ""
echo "=== Starting scDFM ==="

echo "Starting: scdfm_norman_additive"
nohup python "$COMPARISON_DIR/scripts/scdfm_norman_additive.py" \
    > "$LOG_DIR/scdfm_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scdfm_norman_holdout"
nohup python "$COMPARISON_DIR/scripts/scdfm_norman_holdout.py" \
    > "$LOG_DIR/scdfm_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scdfm_loco"
nohup python "$COMPARISON_DIR/scripts/scdfm_loco.py" \
    > "$LOG_DIR/scdfm_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# Comparison Methods - PerturbDiff
# =============================================================================
echo ""
echo "=== Starting PerturbDiff ==="

echo "Starting: perturbdiff_norman_additive"
nohup python "$COMPARISON_DIR/scripts/perturbdiff_norman_additive.py" \
    > "$LOG_DIR/perturbdiff_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: perturbdiff_norman_holdout"
nohup python "$COMPARISON_DIR/scripts/perturbdiff_norman_holdout.py" \
    > "$LOG_DIR/perturbdiff_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: perturbdiff_loco"
nohup python "$COMPARISON_DIR/scripts/perturbdiff_loco.py" \
    > "$LOG_DIR/perturbdiff_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# Comparison Methods - Squidiff
# =============================================================================
echo ""
echo "=== Starting Squidiff ==="

echo "Starting: squidiff_norman_additive"
nohup python "$COMPARISON_DIR/scripts/squidiff_norman_additive.py" \
    > "$LOG_DIR/squidiff_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: squidiff_norman_holdout"
nohup python "$COMPARISON_DIR/scripts/squidiff_norman_holdout.py" \
    > "$LOG_DIR/squidiff_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: squidiff_loco"
nohup python "$COMPARISON_DIR/scripts/squidiff_loco.py" \
    > "$LOG_DIR/squidiff_loco.log" 2>&1 &
echo "  PID: $!"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "All 18 experiments launched!"
echo "=========================================="
echo ""
echo "Monitor all logs:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Check status:"
echo "  ./check_experiments.sh"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo ""
echo "Log directory: $LOG_DIR"
echo ""
echo "Experiments launched:"
echo "  - 3 CellFlow (with graph fusion)"
echo "  - 3 CellFlow baseline (without graph fusion)"
echo "  - 3 GEARS"
echo "  - 3 scDFM"
echo "  - 3 PerturbDiff"
echo "  - 3 Squidiff"
echo "=========================================="
