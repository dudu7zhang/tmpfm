#!/bin/bash
# Run scDFM on Norman 2019 with additive split
# Split: 30% double perturbations as test, all singles in train
# Seed: 20240508, Fold: 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")/scripts"
RUN_ID=${CELLFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Starting scDFM Norman Additive at $(date)"
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_DIR/scdfm_norman_additive.log"

nohup python "$SCRIPTS_DIR/scdfm_norman_additive.py" \
    > "$LOG_DIR/scdfm_norman_additive.log" 2>&1 &

echo "PID: $!"
echo "Monitor with: tail -f $LOG_DIR/scdfm_norman_additive.log"
