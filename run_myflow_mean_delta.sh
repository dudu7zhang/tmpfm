#!/bin/bash
# Launch MyFlow with endpoint, condition-mean-delta, high-delta, and delta-cosine losses.
# Override GPUs or iteration counts with environment variables, e.g.:
#   GPU_NORMAN=1 GPU_REPLOGLE=2 NORMAN_ITERS=30000 REPLOGLE_ITERS=20000 bash run_myflow_mean_delta.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_PY="${FLOW_PY:-$HOME/miniconda3/envs/flow/bin/python}"

RUN_ID="${MYFLOW_MEAN_DELTA_RUN_ID:-mean_delta_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$REPO_DIR/results/logs/myflow_mean_delta/$RUN_ID"
mkdir -p "$LOG_DIR"

GPU_NORMAN="${GPU_NORMAN:-2}"
GPU_REPLOGLE="${GPU_REPLOGLE:-1}"
NORMAN_ITERS="${NORMAN_ITERS:-30000}"
REPLOGLE_ITERS="${REPLOGLE_ITERS:-30000}"

echo "=========================================="
echo "Launching MyFlow mean-delta runs"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Python: $FLOW_PY"
echo "Norman GPU: $GPU_NORMAN, iterations: $NORMAN_ITERS"
echo "Replogle GPU: $GPU_REPLOGLE, iterations: $REPLOGLE_ITERS"
echo "=========================================="

# PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="$GPU_NORMAN" nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_norman_additive.py" \
#   --gpu-id "$GPU_NORMAN" \
#   --output-dir "$REPO_DIR/results/outputs/myflow_norman_mean_delta_$RUN_ID" \
#   --run-name "myflow_norman_mean_delta_$RUN_ID" \
#   --num-iterations "$NORMAN_ITERS" \
#   --endpoint-mse-weight 1.0 \
#   --condition-mean-delta-weight 0.0 \
#   --high-delta-endpoint-weight 0.0 \
#   --terminal-loss-time-power 2.0 \
#   --cosine-loss-weight 0.0 \
#   --overwrite \
#   > "$LOG_DIR/norman.log" 2>&1 &
# NORMAN_PID=$!

PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="$GPU_REPLOGLE" nohup "$FLOW_PY" "$REPO_DIR/scripts/train_myflow_loco_new.py" \
  --gpu-id "$GPU_REPLOGLE" \
  --output-dir "$REPO_DIR/results/outputs/myflow_replogle_mean_delta_$RUN_ID" \
  --run-name "myflow_replogle_mean_delta_$RUN_ID" \
  --seed 20240508 \
  --n-train-perts 28 \
  --n-test-perts 40 \
  --train-cell-fraction 1.0 \
  --test-cell-fraction 1.0 \
  --num-iterations "$REPLOGLE_ITERS" \
  --include-perturbation-in-base-condition \
  --use-cross-cell-delta-condition \
  --cross-cell-delta-prior-weight 0.0 \
  --condition-combined-loss-weight 0.003 \
  --condition-combined-sinkhorn-weight 0.0 \
  --condition-combined-energy-weight 1.0 \
  --endpoint-mse-weight 0.5 \
  --condition-mean-delta-weight 0.25 \
  --high-delta-endpoint-weight 0.25 \
  --high-delta-max-weight 4.0 \
  --terminal-loss-time-power 2.0 \
  --cosine-loss-weight 0.5 \
  --condition-embedding-dim 512 \
  --gradient-accumulation-steps 20 \
  --learning-rate 3e-4 \
  --go-response-num-layers 2 \
  --go-response-top-k 30 \
  --pert-graph-enabled \
  --pert-graph-ppi-file "$REPO_DIR/comparison_methods/TxPert-main/data/graphs/string/v11.5.parquet" \
  --pert-graph-num-layers 2 \
  --pert-graph-rho-dim 128 \
  --predict-n-cells 64 \
  --overwrite \
  > "$LOG_DIR/replogle.log" 2>&1 &
REPLOGLE_PID=$!

{
  echo "RUN_ID=$RUN_ID"
  echo "LOG_DIR=$LOG_DIR"
  if [[ -n "${NORMAN_PID:-}" ]]; then
    echo "NORMAN_PID=$NORMAN_PID"
  fi
  echo "REPLOGLE_PID=$REPLOGLE_PID"
  if [[ -n "${NORMAN_PID:-}" ]]; then
    echo "NORMAN_OUTPUT=$REPO_DIR/results/outputs/myflow_norman_mean_delta_$RUN_ID"
  fi
  echo "REPLOGLE_OUTPUT=$REPO_DIR/results/outputs/myflow_replogle_mean_delta_$RUN_ID"
} | tee "$LOG_DIR/pids.txt"

echo "=========================================="
echo "Launched."
echo "Monitor:"
echo "  tail -f $LOG_DIR/replogle.log"
echo "Check processes:"
echo "  ps -p $REPLOGLE_PID -o pid,stat,etime,cmd"
echo "=========================================="
