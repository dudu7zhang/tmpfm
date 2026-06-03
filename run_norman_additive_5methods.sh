#!/bin/bash
# Run Norman Additive experiments:
#   - MyFlow (ours)
#   - GEARS / CellFlow / scDFM / TxPert (comparison methods)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_SCRIPTS_DIR="$REPO_DIR/comparison_methods/scripts"
RUN_ID=${NORMAN_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
export MYFLOW_RUN_ID=$RUN_ID

LOG_DIR="$REPO_DIR/results/logs/norman_additive_5methods/$RUN_ID"
mkdir -p "$LOG_DIR"

FLOW_PY="${FLOW_PY:-$HOME/miniconda3/envs/flow/bin/python}"
CMP_PY="${CMP_PY:-$HOME/miniconda3/envs/cmp_methods/bin/python}"

GPU_MYFLOW=${GPU_MYFLOW:-0}
GPU_GEARS=${GPU_GEARS:-0}
GPU_CELLFLOW=${GPU_CELLFLOW:-1}
GPU_SCDFM=${GPU_SCDFM:-2}
GPU_TXPERT=${GPU_TXPERT:-1}

echo "=========================================="
echo "Starting Norman Additive runs"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo ""
echo "Methods:"
echo "  MyFlow   (ours)        -> scripts/train_myflow_norman_additive.py"
echo "  GEARS    (comparison)  -> comparison_methods/scripts/gears_norman_additive.py"
echo "  CellFlow (comparison)  -> comparison_methods/scripts/cellflow_baseline_norman_additive.py"
echo "  scDFM    (comparison)  -> comparison_methods/scripts/scdfm_norman_additive.py"
echo "  TxPert   (comparison)  -> comparison_methods/scripts/txpert_norman_additive.py"
echo ""
echo "GPU assignment:"
echo "  MyFlow   -> GPU $GPU_MYFLOW (flow)"
echo "  GEARS    -> GPU $GPU_GEARS (cmp_methods)"
echo "  CellFlow -> GPU $GPU_CELLFLOW (flow)"
echo "  scDFM    -> GPU $GPU_SCDFM (cmp_methods)"
echo "  TxPert   -> GPU $GPU_TXPERT (cmp_methods)"
echo "=========================================="

# CUDA_VISIBLE_DEVICES=$GPU_MYFLOW nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
#     --pert-gnn-enabled --run-name myflow_gnn \
#     > "$LOG_DIR/myflow_norman_additive.log" 2>&1 &
# echo "MyFlow PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_GEARS nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/gears_norman_additive.py" \
#     > "$LOG_DIR/gears_norman_additive.log" 2>&1 &
# echo "GEARS PID: $!"

CUDA_VISIBLE_DEVICES=$GPU_CELLFLOW nohup "$FLOW_PY" "$COMPARISON_SCRIPTS_DIR/cellflow_baseline_norman_additive.py" \
    > "$LOG_DIR/cellflow_norman_additive.log" 2>&1 &
echo "CellFlow PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_SCDFM nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/scdfm_norman_additive.py" \
#     > "$LOG_DIR/scdfm_norman_additive.log" 2>&1 &
# echo "scDFM PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_TXPERT nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/txpert_norman_additive.py" \
#     > "$LOG_DIR/txpert_norman_additive.log" 2>&1 &
# echo "TxPert PID: $!"

echo "=========================================="
echo "Launched all five runs."
echo "Monitor:"
echo "  tail -f $LOG_DIR/myflow_norman_additive.log"
echo "  tail -f $LOG_DIR/gears_norman_additive.log"
echo "  tail -f $LOG_DIR/cellflow_norman_additive.log"
echo "  tail -f $LOG_DIR/scdfm_norman_additive.log"
echo "  tail -f $LOG_DIR/txpert_norman_additive.log"
echo "=========================================="
