#!/bin/bash
# Run Squidiff on Norman 2019 with holdout split
# Split: Hold out 12 genes, test on held-out singles and all doubles involving them
# Seed: 20240508, Split seed base: 42, Fold: 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")/scripts"
RUN_ID=${CELLFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Starting Squidiff Norman Holdout at $(date)"
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_DIR/squidiff_norman_holdout.log"

nohup python "$SCRIPTS_DIR/squidiff_norman_holdout.py" \
    > "$LOG_DIR/squidiff_norman_holdout.log" 2>&1 &

echo "PID: $!"
echo "Monitor with: tail -f $LOG_DIR/squidiff_norman_holdout.log"
