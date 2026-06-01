#!/bin/bash
# =============================================================================
# Run all comparison methods: GEARS, scDFM, MyFlow
# 3 experiments each (additive, holdout, loco) = 9 total
# GPUs: 2, 3, 5, 7
# =============================================================================

set -e

MYFLOW_DIR="/home/zhangshibo24s/cell_flow"
COMPARISON_DIR="$MYFLOW_DIR/comparison_methods"
SCRIPTS_DIR="$MYFLOW_DIR/scripts"
LOG_DIR="$MYFLOW_DIR/results/logs/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting all comparison method experiments"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo "=========================================="

# ===== GEARS (cmp_methods env) =====
echo ""
echo "=== GEARS ==="

echo "Starting: GEARS norman_additive (GPU 2)"
CUDA_VISIBLE_DEVICES=2 nohup python "$COMPARISON_DIR/scripts/gears_norman_additive.py" \
    > "$LOG_DIR/gears_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: GEARS norman_holdout (GPU 3)"
CUDA_VISIBLE_DEVICES=3 nohup python "$COMPARISON_DIR/scripts/gears_norman_holdout.py" \
    > "$LOG_DIR/gears_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: GEARS loco (GPU 5)"
CUDA_VISIBLE_DEVICES=5 nohup python "$COMPARISON_DIR/scripts/gears_loco.py" \
    > "$LOG_DIR/gears_loco.log" 2>&1 &
echo "  PID: $!"

# ===== scDFM (cmp_methods env) =====
echo ""
echo "=== scDFM ==="

echo "Starting: scDFM norman_additive (GPU 2)"
CUDA_VISIBLE_DEVICES=2 nohup python "$COMPARISON_DIR/scripts/scdfm_norman_additive.py" \
    > "$LOG_DIR/scdfm_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scDFM norman_holdout (GPU 3)"
CUDA_VISIBLE_DEVICES=3 nohup python "$COMPARISON_DIR/scripts/scdfm_norman_holdout.py" \
    > "$LOG_DIR/scdfm_norman_holdout.log" 2>&1 &
echo "  PID: $!"

echo "Starting: scDFM loco (GPU 5)"
CUDA_VISIBLE_DEVICES=5 nohup python "$COMPARISON_DIR/scripts/scdfm_loco.py" \
    > "$LOG_DIR/scdfm_loco.log" 2>&1 &
echo "  PID: $!"

# ===== MyFlow (flow env, uses user's training scripts with --no-x-graph-fusion-enabled) =====
echo ""
echo "=== MyFlow ==="

echo "Starting: MyFlow norman_additive (GPU 7)"
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false nohup python "$SCRIPTS_DIR/train_myflow_norman_additive.py" \
    --no-x-graph-fusion-enabled \
    --condition-combined-loss-weight 0 \
    --run-name myflow_comparison_additive \
    --output-dir "$MYFLOW_DIR/results/outputs/outputs_myflow_comparison_additive" \
    > "$LOG_DIR/myflow_norman_additive.log" 2>&1 &
echo "  PID: $!"

echo "Starting: MyFlow norman_holdout (GPU 7, after additive finishes)"
# Holdout and LOCO will be launched after additive starts to avoid GPU OOM on JAX
# These will be launched separately

echo ""
echo "=========================================="
echo "Launched 7 experiments (GEARS x3, scDFM x3, MyFlow additive x1)"
echo "MyFlow holdout & loco will be launched after additive finishes"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep python"
echo "=========================================="
