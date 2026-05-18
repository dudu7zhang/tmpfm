#!/bin/bash
# Run GEARS on Replogle with LOCO split
# Split: Leave-One-Cell-Line-Out (hepg2), 30% train/test perturbations
# Seed: 20240508

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")/scripts"
RUN_ID=${CELLFLOW_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export CELLFLOW_RUN_ID=$RUN_ID
LOG_DIR="$(dirname "$SCRIPT_DIR")/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Starting GEARS LOCO at $(date)"
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_DIR/gears_loco.log"

nohup python "$SCRIPTS_DIR/gears_loco.py" \
    > "$LOG_DIR/gears_loco.log" 2>&1 &

echo "PID: $!"
echo "Monitor with: tail -f $LOG_DIR/gears_loco.log"
