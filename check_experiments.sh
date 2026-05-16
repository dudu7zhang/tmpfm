#!/bin/bash
# =============================================================================
# Check status of all 18 experiments
# =============================================================================

LOG_DIR="/home/zhangshibo24s/cell_flow/logs_all_experiments"

echo "=========================================="
echo "Experiment Status Check"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory not found: $LOG_DIR"
    echo "Run ./run_all_experiments.sh first"
    exit 1
fi

# Count running python processes
RUNNING=$(ps aux | grep python | grep -v grep | wc -l)
echo "Running Python processes: $RUNNING"
echo ""

# Check each experiment
echo "Experiment Status:"
echo "------------------------------------------"

EXPERIMENTS=(
    "cellflow_norman_additive"
    "cellflow_norman_holdout"
    "cellflow_loco"
    "cellflow_baseline_norman_additive"
    "cellflow_baseline_norman_holdout"
    "cellflow_baseline_loco"
    "gears_norman_additive"
    "gears_norman_holdout"
    "gears_loco"
    "scdfm_norman_additive"
    "scdfm_norman_holdout"
    "scdfm_loco"
    "perturbdiff_norman_additive"
    "perturbdiff_norman_holdout"
    "perturbdiff_loco"
    "squidiff_norman_additive"
    "squidiff_norman_holdout"
    "squidiff_loco"
)

for exp in "${EXPERIMENTS[@]}"; do
    log_file="$LOG_DIR/${exp}.log"
    if [ -f "$log_file" ]; then
        # Check if process is still running
        if ps aux | grep -q "[p]ython.*${exp}"; then
            STATUS="RUNNING"
            LAST_LINE=$(tail -1 "$log_file" 2>/dev/null | head -c 80)
        elif grep -q "Done\.\|Evaluation failed\|Error\|Traceback" "$log_file" 2>/dev/null; then
            if grep -q "Done\." "$log_file" 2>/dev/null; then
                STATUS="COMPLETED"
            else
                STATUS="FAILED"
            fi
            LAST_LINE=$(tail -1 "$log_file" 2>/dev/null | head -c 80)
        else
            STATUS="UNKNOWN"
            LAST_LINE=$(tail -1 "$log_file" 2>/dev/null | head -c 80)
        fi
        printf "%-40s %-10s %s\n" "$exp" "$STATUS" "$LAST_LINE"
    else
        printf "%-40s %-10s\n" "$exp" "NO LOG"
    fi
done

echo ""
echo "=========================================="
echo "Quick Commands:"
echo "  View all logs: tail -f $LOG_DIR/*.log"
echo "  View specific: tail -f $LOG_DIR/<experiment>.log"
echo "  Kill all: pkill -f 'python.*train_cellflow\|python.*gears\|python.*scdfm\|python.*perturbdiff\|python.*squidiff'"
echo "=========================================="
