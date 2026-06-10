#!/bin/bash
# Ablation study: PriorFlow 消融实验
# 4 variants:
#   1. Full PriorFlow (完整模型)
#   2. w/o GO+STRING GNN (移除图神经网络)
#   3. w/o TRRUST Gene Mask (移除TRRUST基因掩码+注意力偏置)
#   4. w/o SNR Weighted Loss (移除SNR加权损失)
#
# Usage:
#   bash run_norman_ablation.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID=${NORMAN_ABLATION_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export MYFLOW_RUN_ID=$RUN_ID

LOG_DIR="$REPO_DIR/results/logs/norman_ablation/$RUN_ID"
mkdir -p "$LOG_DIR"

FLOW_PY="${FLOW_PY:-$HOME/miniconda3/envs/flow/bin/python}"

# GPU assignment
GPU_FULL=${GPU_ABLATION_FULL:-2}
GPU_NO_GNN=${GPU_ABLATION_NO_GNN:-3}
GPU_NO_MASK=${GPU_ABLATION_NO_MASK:-4}
GPU_NO_SNR=${GPU_ABLATION_NO_SNR:-5}

# Shared hyperparams (consistent across all variants)
CONDITION_EMBEDDING_DIM=256
GNN_HIDDEN_DIM=128
GNN_NUM_LAYERS=4
GNN_NUM_HEADS=4
ENDPOINT_MSE_WEIGHT=0.0
CONDITION_MEAN_DELTA_WEIGHT=0.0
TOP_DELTA_LOSS_WEIGHT=0.0
TOP_DELTA_ENDPOINT_WEIGHT=0.0
TOP_DELTA_FRACTION=0.05
TOP_DELTA_MIN_GENES=50
FLOW_NOISE=0.0
DELTA_HEAD_WEIGHT=0.0

echo "=========================================="
echo "PriorFlow Ablation Study"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo ""
echo "Variants:"
echo "  1. Full PriorFlow          -> GPU $GPU_FULL"
echo "  2. w/o GO+STRING GNN       -> GPU $GPU_NO_GNN"
echo "  3. w/o TRRUST Gene Mask    -> GPU $GPU_NO_MASK"
echo "  4. w/o SNR Weighted Loss   -> GPU $GPU_NO_SNR"
echo "=========================================="

# ============================================================
# Variant 1: Full PriorFlow (完整模型)
#   GNN + TRRUST mask&attn-bias + SNR loss
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_FULL nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
    --pert-gnn-enabled --run-name ablation_full \
    --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
    --enhanced-pert-gnn \
    --pert-gnn-hidden-dim "$GNN_HIDDEN_DIM" \
    --pert-gnn-num-layers "$GNN_NUM_LAYERS" \
    --pert-gnn-num-heads "$GNN_NUM_HEADS" \
    --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
    --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
    --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
    --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
    --top-delta-fraction "$TOP_DELTA_FRACTION" \
    --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
    --snr-endpoint-weight 1.0 \
    --flow-noise "$FLOW_NOISE" \
    --delta-head-weight "$DELTA_HEAD_WEIGHT" \
    --trrust-mask-enabled \
    --trrust-attn-bias-enabled \
    > "$LOG_DIR/ablation_full.log" 2>&1 &
echo "Full PriorFlow PID: $!"

# ============================================================
# Variant 2: w/o GO+STRING GNN (移除图神经网络)
#   No --pert-gnn-enabled, no --enhanced-pert-gnn
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_NO_GNN nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
    --run-name ablation_no_gnn \
    --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
    --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
    --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
    --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
    --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
    --top-delta-fraction "$TOP_DELTA_FRACTION" \
    --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
    --snr-endpoint-weight 1.0 \
    --flow-noise "$FLOW_NOISE" \
    --delta-head-weight "$DELTA_HEAD_WEIGHT" \
    --trrust-mask-enabled \
    --trrust-attn-bias-enabled \
    > "$LOG_DIR/ablation_no_gnn.log" 2>&1 &
echo "w/o GNN PID: $!"

# ============================================================
# Variant 3: w/o TRRUST Gene Mask (移除TRRUST基因掩码+注意力偏置)
#   No --trrust-mask-enabled, no --trrust-attn-bias-enabled
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_NO_MASK nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
    --pert-gnn-enabled --run-name ablation_no_genemask \
    --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
    --enhanced-pert-gnn \
    --pert-gnn-hidden-dim "$GNN_HIDDEN_DIM" \
    --pert-gnn-num-layers "$GNN_NUM_LAYERS" \
    --pert-gnn-num-heads "$GNN_NUM_HEADS" \
    --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
    --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
    --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
    --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
    --top-delta-fraction "$TOP_DELTA_FRACTION" \
    --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
    --snr-endpoint-weight 1.0 \
    --flow-noise "$FLOW_NOISE" \
    --delta-head-weight "$DELTA_HEAD_WEIGHT" \
    > "$LOG_DIR/ablation_no_genemask.log" 2>&1 &
echo "w/o Gene Mask PID: $!"

# ============================================================
# Variant 4: w/o SNR Weighted Loss (移除SNR加权损失)
#   --snr-endpoint-weight 0.0
# ============================================================
CUDA_VISIBLE_DEVICES=$GPU_NO_SNR nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
    --pert-gnn-enabled --run-name ablation_no_snr \
    --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
    --enhanced-pert-gnn \
    --pert-gnn-hidden-dim "$GNN_HIDDEN_DIM" \
    --pert-gnn-num-layers "$GNN_NUM_LAYERS" \
    --pert-gnn-num-heads "$GNN_NUM_HEADS" \
    --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
    --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
    --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
    --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
    --top-delta-fraction "$TOP_DELTA_FRACTION" \
    --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
    --snr-endpoint-weight 0.0 \
    --flow-noise "$FLOW_NOISE" \
    --delta-head-weight "$DELTA_HEAD_WEIGHT" \
    --trrust-mask-enabled \
    --trrust-attn-bias-enabled \
    > "$LOG_DIR/ablation_no_snr.log" 2>&1 &
echo "w/o SNR PID: $!"

echo "=========================================="
echo "Launched all 4 ablation runs."
echo "Monitor:"
echo "  tail -f $LOG_DIR/ablation_full.log"
echo "  tail -f $LOG_DIR/ablation_no_gnn.log"
echo "  tail -f $LOG_DIR/ablation_no_genemask.log"
echo "  tail -f $LOG_DIR/ablation_no_snr.log"
echo "=========================================="
