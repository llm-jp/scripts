#!/bin/bash

# This script is intended to be run as a cron job to evaluate new checkpoints.
# Usage:
#   bash cron.sh <checkpoint_glob_pattern> <log_file>

set -eux

# Get the path to the evaluation script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_PATH="${SCRIPT_DIR}/run_llm-jp-eval.sh"

# Arguments
CHECKPOINT_GLOB_PATTERN=$1
LOG_FILE=$2

# Create log file if it doesn't exist
touch "$LOG_FILE"

for checkpoint_path in $CHECKPOINT_GLOB_PATTERN; do
    # Skip if the checkpoint has already been evaluated
    if grep -qx $checkpoint_path $LOG_FILE; then
        continue
    fi
    
    # Log the checkpoint path
    # NOTE: This is done before evaluation to avoid re-evaluating the same checkpoint
    echo $checkpoint_path >> $LOG_FILE

    # Evaluate the checkpoint
    echo "Evaluating checkpoint $checkpoint_path"
    wandb_run_name=$(basename $checkpoint_path)
    sbatch $SCRIPT_PATH $checkpoint_path $wandb_run_name
done
