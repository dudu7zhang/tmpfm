#!/bin/bash
# Ablation study: MyFlow 消融实验
# 消融两组组件:
#   - Prior (先验): GNN + TRRUST Gene Mask + TRRUST Attention Bias
#   - SNR Loss (SNR加权损失)
# 4 variants: 都没有, 只有Prior, 只有SNR, 全有(完整模型)
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
GPU_NEITHER=${GPU_ABLATION_NEITHER:-2}  # 都没有
GPU_PRIOR=${GPU_ABLATION_PRIOR:-3}      # 只有Prior
GPU_SNR=${GPU_ABLATION_SNR:-4}          # 只有SNR
GPU_FULL=${GPU_ABLATION_FULL:-5}        # 全有(完整模型)

# Shared hyperparams (matching run_norman_additive_5methods.sh)
CONDITION_EMBEDDING_DIM=256
GNN_HIDDEN_DIM=128
GNN_NUM_LAYERS=4
GNN_NUM_HEADS=4
ENDPOINT_MSE_WEIGHT=0.0
CONDITION_MEAN_DELTA_WEIGHT=0.0
TOP_DELTA_LOSS_WEIGHT=0.0
TOP_DELTA_ENDPOINT_WEIGHT=0.0
TOP_DELTA_FRACTION=0.0
TOP_DELTA_MIN_GENES=50
SNR_ENDPOINT_WEIGHT=1.0
FLOW_NOISE=0.0
DELTA_HEAD_WEIGHT=0.0

echo "=========================================="
echo "MyFlow Ablation Study"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Time: $(date)"
echo ""
echo "Variants:"
echo "  1. Neither (gene2vec only,           无SNR) -> GPU $GPU_NEITHER"
echo "  2. Prior   (gene2vec+GNN+TRRUST,     无SNR) -> GPU $GPU_PRIOR"
echo "  3. SNR     (gene2vec only,           SNR=1) -> GPU $GPU_SNR"
echo "  4. Full    (gene2vec+GNN+TRRUST,     SNR=1) -> GPU $GPU_FULL"
echo ""
echo "Prior = --pert-gnn-enabled --enhanced-pert-gnn --trrust-mask-enabled --trrust-attn-bias-enabled"
echo "SNR   = --snr-endpoint-weight 1.0"
echo "No GNN/TRRUST/SNR means those flags are absent or set to 0"
echo "=========================================="

# ============================================================
# Variant 1: Neither — 无Prior 无SNR
# ============================================================
# CUDA_VISIBLE_DEVICES=$GPU_NEITHER nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
#     --run-name ablation_neither \
#     --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
#     --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
#     --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
#     --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
#     --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
#     --top-delta-fraction "$TOP_DELTA_FRACTION" \
#     --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
#     --snr-endpoint-weight 0.0 \
#     --flow-noise "$FLOW_NOISE" \
#     --delta-head-weight "$DELTA_HEAD_WEIGHT" \
#     > "$LOG_DIR/ablation_neither.log" 2>&1 &
# echo "Neither PID: $!"

# ============================================================
# Variant 2: Prior only — 有Prior 无SNR
#   GNN + TRRUST Mask + TRRUST Attn Bias
# ============================================================
# CUDA_VISIBLE_DEVICES=$GPU_PRIOR nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
#     --pert-gnn-enabled --run-name ablation_prior \
#     --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
#     --enhanced-pert-gnn \
#     --pert-gnn-hidden-dim "$GNN_HIDDEN_DIM" \
#     --pert-gnn-num-layers "$GNN_NUM_LAYERS" \
#     --pert-gnn-num-heads "$GNN_NUM_HEADS" \
#     --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
#     --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
#     --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
#     --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
#     --top-delta-fraction "$TOP_DELTA_FRACTION" \
#     --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
#     --snr-endpoint-weight 0.0 \
#     --flow-noise "$FLOW_NOISE" \
#     --delta-head-weight "$DELTA_HEAD_WEIGHT" \
#     --trrust-mask-enabled \
#     --trrust-attn-bias-enabled \
#     > "$LOG_DIR/ablation_prior.log" 2>&1 &
# echo "Prior PID: $!"

# ============================================================
# Variant 3: SNR only — 无Prior 有SNR
# ============================================================
# CUDA_VISIBLE_DEVICES=$GPU_SNR nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
#     --no-gene2vec \
#     --run-name ablation_snr \
#     --condition-embedding-dim "$CONDITION_EMBEDDING_DIM" \
#     --endpoint-mse-weight "$ENDPOINT_MSE_WEIGHT" \
#     --condition-mean-delta-weight "$CONDITION_MEAN_DELTA_WEIGHT" \
#     --top-delta-loss-weight "$TOP_DELTA_LOSS_WEIGHT" \
#     --top-delta-endpoint-weight "$TOP_DELTA_ENDPOINT_WEIGHT" \
#     --top-delta-fraction "$TOP_DELTA_FRACTION" \
#     --top-delta-min-genes "$TOP_DELTA_MIN_GENES" \
#     --snr-endpoint-weight "$SNR_ENDPOINT_WEIGHT" \
#     --flow-noise "$FLOW_NOISE" \
#     --delta-head-weight "$DELTA_HEAD_WEIGHT" \
#     > "$LOG_DIR/ablation_snr.log" 2>&1 &
# echo "SNR PID: $!"

# ============================================================
# Variant 4: Full (完整模型) — 有Prior 有SNR
#   参数与 run_norman_additive_5methods.sh 中 MyFlow 完全一致
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
    --snr-endpoint-weight "$SNR_ENDPOINT_WEIGHT" \
    --flow-noise "$FLOW_NOISE" \
    --delta-head-weight "$DELTA_HEAD_WEIGHT" \
    --trrust-mask-enabled \
    --trrust-attn-bias-enabled \
    > "$LOG_DIR/ablation_full.log" 2>&1 &
echo "Full PID: $!"

echo "=========================================="
echo "Launched all 4 ablation runs."
echo "Monitor:"
echo "  tail -f $LOG_DIR/ablation_neither.log"
echo "  tail -f $LOG_DIR/ablation_prior.log"
echo "  tail -f $LOG_DIR/ablation_snr.log"
echo "  tail -f $LOG_DIR/ablation_full.log"
echo "=========================================="
