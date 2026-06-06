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
GPU_TXPERT=${GPU_TXPERT:-3}

MYFLOW_ENDPOINT_MSE_WEIGHT=${MYFLOW_ENDPOINT_MSE_WEIGHT:-1.0}
MYFLOW_CONDITION_MEAN_DELTA_WEIGHT=${MYFLOW_CONDITION_MEAN_DELTA_WEIGHT:-0.1}
MYFLOW_TOP_DELTA_LOSS_WEIGHT=${MYFLOW_TOP_DELTA_LOSS_WEIGHT:-0.0}
MYFLOW_TOP_DELTA_ENDPOINT_WEIGHT=${MYFLOW_TOP_DELTA_ENDPOINT_WEIGHT:-0.0}
MYFLOW_TOP_DELTA_FRACTION=${MYFLOW_TOP_DELTA_FRACTION:-0.05}
MYFLOW_TOP_DELTA_MIN_GENES=${MYFLOW_TOP_DELTA_MIN_GENES:-50}
MYFLOW_SNR_ENDPOINT_WEIGHT=${MYFLOW_SNR_ENDPOINT_WEIGHT:-0.5}
MYFLOW_COSINE_LOSS_WEIGHT=${MYFLOW_COSINE_LOSS_WEIGHT:-0.05}
MYFLOW_FLOW_NOISE=${MYFLOW_FLOW_NOISE:-0.1}
MYFLOW_DELTA_HEAD_ENABLED=${MYFLOW_DELTA_HEAD_ENABLED:-}
MYFLOW_DELTA_HEAD_WEIGHT=${MYFLOW_DELTA_HEAD_WEIGHT:-0.0}

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
    --enhanced-pert-gnn \
    --pert-gnn-hidden-dim 128 \
    --pert-gnn-num-layers 4 \
    --pert-gnn-num-heads 4 \
    --seed 20240508 \
    --n-train-perts 28 \
    --n-test-perts 40 \
    --train-cell-fraction 1.0 \
    --test-cell-fraction 1.0 \
    --num-iterations 20000 \
    --endpoint-mse-weight "$MYFLOW_ENDPOINT_MSE_WEIGHT" \
    --condition-mean-delta-weight "$MYFLOW_CONDITION_MEAN_DELTA_WEIGHT" \
    --top-delta-loss-weight "$MYFLOW_TOP_DELTA_LOSS_WEIGHT" \
    --top-delta-endpoint-weight "$MYFLOW_TOP_DELTA_ENDPOINT_WEIGHT" \
    --top-delta-fraction "$MYFLOW_TOP_DELTA_FRACTION" \
    --top-delta-min-genes "$MYFLOW_TOP_DELTA_MIN_GENES" \
    --snr-endpoint-weight "$MYFLOW_SNR_ENDPOINT_WEIGHT" \
    --condition-embedding-dim 512 \
    --condition-combined-loss-weight 0.0 \
    --cosine-loss-weight "$MYFLOW_COSINE_LOSS_WEIGHT" \
    --flow-noise "$MYFLOW_FLOW_NOISE" \
    $MYFLOW_DELTA_HEAD_ENABLED \
    --delta-head-weight "$MYFLOW_DELTA_HEAD_WEIGHT" \
    --batch-size 256 \
    --learning-rate 5e-4 \
    --gradient-accumulation-steps 1 \
    --match-every-n 20 \
    --cond-output-dropout 0.0 \
    --cross-attn-layers 1 \
    --gene-attn-dim 64 \
    --gene-self-attn-layers 0 \
    --cross-attn-heads 4 \
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

# CUDA_VISIBLE_DEVICES=$GPU_TXPERT \
#     TXPERT_EPOCHS=80 \
#     TXPERT_LR=1e-5 \
#     TXPERT_BATCH_SIZE=64 \
#     nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/txpert_loco.py" \
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
