#!/bin/bash
# Run Replogle LOCO experiments:
#   - MyFlow (ours)
#   - GEARS / CellFlow / scDFM / TxPert (comparison methods)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_SCRIPTS_DIR="$REPO_DIR/comparison_methods/scripts"
RUN_ID=${LOCO_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
export MYFLOW_RUN_ID=$RUN_ID

LOG_DIR="$REPO_DIR/results/logs/replogle_loco_4methods/$RUN_ID"
mkdir -p "$LOG_DIR"

FLOW_PY="${FLOW_PY:-$HOME/miniconda3/envs/flow/bin/python}"
CMP_PY="${CMP_PY:-$HOME/miniconda3/envs/cmp_methods/bin/python}"

GPU_MYFLOW=${GPU_MYFLOW:-0}
GPU_GEARS=${GPU_GEARS:-0}
GPU_CELLFLOW=${GPU_CELLFLOW:-1}
GPU_SCDFM=${GPU_SCDFM:-1}
GPU_TXPERT=${GPU_TXPERT:-2}

echo "=========================================="
echo "Starting Replogle LOCO runs"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo ""
echo "Methods:"
echo "  MyFlow   (ours)        -> scripts/train_myflow_loco_new.py"
echo "  GEARS    (comparison)  -> comparison_methods/scripts/gears_loco.py"
echo "  CellFlow (comparison)  -> comparison_methods/scripts/cellflow_baseline_loco.py"
echo "  scDFM   (comparison)   -> comparison_methods/scripts/scdfm_loco.py"
echo "  TxPert  (comparison)   -> comparison_methods/scripts/txpert_loco.py"
echo ""
echo "GPU assignment:"
echo "  MyFlow   -> GPU $GPU_MYFLOW (flow)"
echo "  GEARS    -> GPU $GPU_GEARS (cmp_methods)"
echo "  CellFlow -> GPU $GPU_CELLFLOW (flow; cmp_methods JAX is CPU-only)"
echo "  scDFM    -> GPU $GPU_SCDFM (cmp_methods)"
echo "  TxPert   -> GPU $GPU_TXPERT (cmp_methods)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_MYFLOW nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_loco_new.py" \
    --output-dir "$REPO_DIR/results/outputs/myflow_replogle_loco_$RUN_ID" \
    --run-name "myflow_replogle_loco_$RUN_ID" \
    --pert-gnn-enabled \
    --seed 20240508 \
    --n-train-perts 28 \
    --n-test-perts 40 \
    --train-cell-fraction 1.0 \
    --test-cell-fraction 1.0 \
    --num-iterations 30000 \
    --condition-combined-loss-weight 0.003 \
    --endpoint-mse-weight 0.5 \
    --cosine-loss-weight 0.3 \
    --condition-embedding-dim 256 \
    --cond-output-dropout 0.05 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-4 \
    --predict-n-cells 64 \
    > "$LOG_DIR/myflow_loco.log" 2>&1 &
echo "MyFlow PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_GEARS nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/gears_loco.py" \
#     > "$LOG_DIR/gears_loco.log" 2>&1 &
# echo "GEARS PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW nohup "$FLOW_PY" "$COMPARISON_SCRIPTS_DIR/cellflow_baseline_loco.py" \
#     > "$LOG_DIR/cellflow_loco.log" 2>&1 &
# echo "CellFlow PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_SCDFM nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/scdfm_loco.py" \
#     > "$LOG_DIR/scdfm_loco.log" 2>&1 &
# echo "scDFM PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_TXPERT nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/txpert_loco.py" \
#     > "$LOG_DIR/txpert_loco.log" 2>&1 &
# echo "TxPert PID: $!"

echo "=========================================="
echo "Launched all runs."
echo "Monitor:"
echo "  tail -f $LOG_DIR/myflow_loco.log"
echo "  tail -f $LOG_DIR/gears_loco.log"
echo "  tail -f $LOG_DIR/cellflow_loco.log"
echo "  tail -f $LOG_DIR/scdfm_loco.log"
echo "  tail -f $LOG_DIR/txpert_loco.log"
echo "=========================================="
