#!/bin/bash
# =============================================================================
# Run comparison method experiments (12 experiments)
# Conda environment: cmp_methods
# =============================================================================
# Methods:
#   1. GEARS
#   2. scDFM
#   3. PerturbDiff
#   4. Squidiff
#
# Data Splits:
#   1. Norman Additive (30% double perturbations as test)
#   2. Norman Holdout (12 genes holdout)
#   3. LOCO (Leave-One-Cell-Line-Out, hepg2)
# =============================================================================

set -e

CELLFLOW_DIR="/home/zhangshibo24s/cell_flow"
COMPARISON_DIR="$CELLFLOW_DIR/comparison_methods"
DATE=$(date +%Y%m%d_%H%M)
LOG_DIR="$CELLFLOW_DIR/logs_all_experiments_${DATE}"
mkdir -p "$LOG_DIR"

# Per-experiment GPU assignment (edit these to match your machine)
GPU_GEARS_ADDITIVE=0
GPU_GEARS_HOLDOUT=1
GPU_GEARS_LOCO=2
GPU_SCDFM_ADDITIVE=3
GPU_SCDFM_HOLDOUT=0
GPU_SCDFM_LOCO=1
GPU_PERTURBDIFF_ADDITIVE=2
GPU_PERTURBDIFF_HOLDOUT=3
GPU_PERTURBDIFF_LOCO=0
GPU_SQUIDIFF_ADDITIVE=1
GPU_SQUIDIFF_HOLDOUT=2
GPU_SQUIDIFF_LOCO=3

echo "=========================================="
echo "Starting comparison experiments (12 runs)"
echo "Conda env: cmp_methods"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# Activate conda environment
source activate cmp_methods || conda activate cmp_methods

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
# scDFM
# =============================================================================
echo ""
echo "=== Starting scDFM ==="

echo "Starting: scdfm_norman_additive (GPU $GPU_SCDFM_ADDITIVE)"
CUDA_VISIBLE_DEVICES=$GPU_SCDFM_ADDITIVE nohup python "$COMPARISON_DIR/scripts/scdfm_norman_additive.py" \
    > "$LOG_DIR/scdfm_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scdfm_norman_holdout (GPU $GPU_SCDFM_HOLDOUT)"
CUDA_VISIBLE_DEVICES=$GPU_SCDFM_HOLDOUT nohup python "$COMPARISON_DIR/scripts/scdfm_norman_holdout.py" \
    > "$LOG_DIR/scdfm_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scdfm_loco (GPU $GPU_SCDFM_LOCO)"
CUDA_VISIBLE_DEVICES=$GPU_SCDFM_LOCO nohup python "$COMPARISON_DIR/scripts/scdfm_loco.py" \
    > "$LOG_DIR/scdfm_loco.log" 2>&1 &
echo "  PID: $!"

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
echo "All 12 comparison experiments launched!"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/{gears,scdfm,perturbdiff,squidiff}_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
