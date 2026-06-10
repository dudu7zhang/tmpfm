#!/bin/bash
# Run Norman Additive experiments:
#   - MyFlow (ours)
#   - GEARS / CellFlow / scDFM / TxPert (comparison methods)
# -trrust-mask-enabled 和 --trrust-attn-bias-enabled 
# MYFLOW_TRRUST_MASK_ENABLED="--trrust-mask-enabled" \                                                                                                            
#   MYFLOW_TRRUST_ATTN_BIAS_ENABLED="--trrust-attn-bias-enabled" \                                                                                                  
#   bash run_norman_additive_5methods.sh
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
CPA_PY="${CPA_PY:-$HOME/miniconda3/envs/cmp_methods/bin/python}"

GPU_MYFLOW=${GPU_MYFLOW:-4}
GPU_GEARS=${GPU_GEARS:-1}
GPU_CELLFLOW=${GPU_CELLFLOW:-1}
GPU_SCDFM=${GPU_SCDFM:-2}
GPU_TXPERT=${GPU_TXPERT:-1}
GPU_CPA=${GPU_CPA:-0}

MYFLOW_ENDPOINT_MSE_WEIGHT=${MYFLOW_ENDPOINT_MSE_WEIGHT:-0.0}
MYFLOW_CONDITION_MEAN_DELTA_WEIGHT=${MYFLOW_CONDITION_MEAN_DELTA_WEIGHT:-0.0}
MYFLOW_TOP_DELTA_LOSS_WEIGHT=${MYFLOW_TOP_DELTA_LOSS_WEIGHT:-0.0}
MYFLOW_TOP_DELTA_ENDPOINT_WEIGHT=${MYFLOW_TOP_DELTA_ENDPOINT_WEIGHT:-0.0}
MYFLOW_TOP_DELTA_FRACTION=${MYFLOW_TOP_DELTA_FRACTION:-0.05}
MYFLOW_TOP_DELTA_MIN_GENES=${MYFLOW_TOP_DELTA_MIN_GENES:-50}
MYFLOW_SNR_ENDPOINT_WEIGHT=${MYFLOW_SNR_ENDPOINT_WEIGHT:-1.0}
MYFLOW_FLOW_NOISE=${MYFLOW_FLOW_NOISE:-0.0}
MYFLOW_DELTA_HEAD_ENABLED=${MYFLOW_DELTA_HEAD_ENABLED:-}
MYFLOW_DELTA_HEAD_WEIGHT=${MYFLOW_DELTA_HEAD_WEIGHT:-0.0}
MYFLOW_TRRUST_MASK_ENABLED=${MYFLOW_TRRUST_MASK_ENABLED:-}
MYFLOW_TRRUST_ATTN_BIAS_ENABLED=${MYFLOW_TRRUST_ATTN_BIAS_ENABLED:-}
MYFLOW_ENHANCED_GNN=${MYFLOW_ENHANCED_GNN:---enhanced-pert-gnn}
MYFLOW_GNN_HIDDEN_DIM=${MYFLOW_GNN_HIDDEN_DIM:-128}
MYFLOW_GNN_NUM_LAYERS=${MYFLOW_GNN_NUM_LAYERS:-4}
MYFLOW_GNN_NUM_HEADS=${MYFLOW_GNN_NUM_HEADS:-4}

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
echo "  CPA      (comparison)  -> comparison_methods/scripts/cpa_norman_additive.py"
echo ""
echo "GPU assignment:"
echo "  MyFlow   -> GPU $GPU_MYFLOW (flow)"
echo "  GEARS    -> GPU $GPU_GEARS (cmp_methods)"
echo "  CellFlow -> GPU $GPU_CELLFLOW (flow)"
echo "  scDFM    -> GPU $GPU_SCDFM (cmp_methods)"
echo "  TxPert   -> GPU $GPU_TXPERT (cmp_methods)"
echo "  CPA      -> GPU $GPU_CPA (cmp_methods)"
echo "=========================================="

# CUDA_VISIBLE_DEVICES=$GPU_MYFLOW nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
#     --pert-gnn-enabled --run-name myflow_gnn \
#     --condition-embedding-dim 256 \
#     $MYFLOW_ENHANCED_GNN \
#     --pert-gnn-hidden-dim "$MYFLOW_GNN_HIDDEN_DIM" \
#     --pert-gnn-num-layers "$MYFLOW_GNN_NUM_LAYERS" \
#     --pert-gnn-num-heads "$MYFLOW_GNN_NUM_HEADS" \
#     --endpoint-mse-weight "$MYFLOW_ENDPOINT_MSE_WEIGHT" \
#     --condition-mean-delta-weight "$MYFLOW_CONDITION_MEAN_DELTA_WEIGHT" \
#     --top-delta-loss-weight "$MYFLOW_TOP_DELTA_LOSS_WEIGHT" \
#     --top-delta-endpoint-weight "$MYFLOW_TOP_DELTA_ENDPOINT_WEIGHT" \
#     --top-delta-fraction "$MYFLOW_TOP_DELTA_FRACTION" \
#     --top-delta-min-genes "$MYFLOW_TOP_DELTA_MIN_GENES" \
#     --snr-endpoint-weight "$MYFLOW_SNR_ENDPOINT_WEIGHT" \
#     --flow-noise "$MYFLOW_FLOW_NOISE" \
#     $MYFLOW_DELTA_HEAD_ENABLED \
#     --delta-head-weight "$MYFLOW_DELTA_HEAD_WEIGHT" \
#     $MYFLOW_TRRUST_MASK_ENABLED \
#     $MYFLOW_TRRUST_ATTN_BIAS_ENABLED \
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

# CUDA_VISIBLE_DEVICES=$GPU_CPA \
#     CPA_EPOCHS=500 \
#     CPA_LR=3e-4 \
#     CPA_BATCH_SIZE=256 \
#     nohup "$CPA_PY" "$COMPARISON_SCRIPTS_DIR/cpa_norman_additive.py" \
#     > "$LOG_DIR/cpa_norman_additive.log" 2>&1 &
# echo "CPA PID: $!"

# CUDA_VISIBLE_DEVICES=$GPU_TXPERT \
#     TXPERT_EPOCHS=70 \
#     TXPERT_LR=4e-5 \
#     TXPERT_BATCH_SIZE=256 \
#     nohup "$CMP_PY" "$COMPARISON_SCRIPTS_DIR/txpert_norman_additive.py" \
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
echo "  tail -f $LOG_DIR/cpa_norman_additive.log"
echo "=========================================="
