#!/bin/bash
set -e

RESULTS_DIR="/home/zhangshibo24s/cell_flow/results/outputs"
LOG_DIR="/home/zhangshibo24s/cell_flow/results/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "========================================" | tee "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "Starting experiments at $(date)" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "========================================" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"

# Experiment 1: Our method - Norman holdout
echo "" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "[1/4] Running our method on Norman holdout..." | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
conda run -n flow python /home/zhangshibo24s/cell_flow/scripts/train_myflow_norman_holdout.py \
    --output-dir "$RESULTS_DIR/our_method_holdout_${TIMESTAMP}" \
    --neighborhood-only --neighborhood-hops 2 --max-neighbors 128 \
    --change-loss-weight 0.1 \
    --conditioning film \
    > "$LOG_DIR/our_method_holdout_${TIMESTAMP}.log" 2>&1
echo "[1/4] Completed at $(date), exit code: $?" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"

# Experiment 2: Our method - Norman additive
echo "" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "[2/4] Running our method on Norman additive..." | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
conda run -n flow python /home/zhangshibo24s/cell_flow/scripts/train_myflow_norman_additive.py \
    --output-dir "$RESULTS_DIR/our_method_additive_${TIMESTAMP}" \
    --neighborhood-only --neighborhood-hops 2 --max-neighbors 128 \
    --change-loss-weight 0.1 \
    --conditioning film \
    > "$LOG_DIR/our_method_additive_${TIMESTAMP}.log" 2>&1
echo "[2/4] Completed at $(date), exit code: $?" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"

# Experiment 3: MyFlow baseline - Norman holdout
echo "" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "[3/4] Running MyFlow baseline on Norman holdout..." | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
conda run -n cmp_methods python /home/zhangshibo24s/cell_flow/comparison_methods/scripts/myflow_baseline_norman_holdout.py \
    --output-dir "$RESULTS_DIR/myflow_baseline_holdout_${TIMESTAMP}" \
    > "$LOG_DIR/myflow_baseline_holdout_${TIMESTAMP}.log" 2>&1
echo "[3/4] Completed at $(date), exit code: $?" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"

# Experiment 4: MyFlow baseline - Norman additive
echo "" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "[4/4] Running MyFlow baseline on Norman additive..." | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
conda run -n cmp_methods python /home/zhangshibo24s/cell_flow/comparison_methods/scripts/myflow_baseline_norman_additive.py \
    --output-dir "$RESULTS_DIR/myflow_baseline_additive_${TIMESTAMP}" \
    > "$LOG_DIR/myflow_baseline_additive_${TIMESTAMP}.log" 2>&1
echo "[4/4] Completed at $(date), exit code: $?" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"

echo "" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "========================================" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "All experiments completed at $(date)" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
echo "========================================" | tee -a "$LOG_DIR/experiment_summary_${TIMESTAMP}.log"
