#!/bin/bash
#SBATCH --job-name=FIXME
#SBATCH --partition=FIXME
#SBATCH --nodes=0
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# LLM-jp v4 7.7B launcher script.
# Usage:
#   sbatch \
#     --job-name=9999_train \
#     --partition=gpu \
#     --nodes=64 \
#     --outpout=train.slurm.out \
#     sbatch_train.sh \
#       /path/to/env \
#       /path/to/model \
#       llm-jp \
#       9999_train

set -eu -o pipefail

# Arguments
if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 ENV_DIR MODEL_DIR WANDB_ENTITY WANDB_PROJECT"
    exit 1
fi
ENV_DIR=$(realpath -eP $1); shift
MODEL_DIR=$(realpath -m $1); shift
WANDB_ENTITY=$1; shift
WANDB_PROJECT=$1; shift
echo "ENV_DIR=${ENV_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"

# Find the script directory
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(
        scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/Command=/{print $2}' \
        | cut -d ' ' -f 1
    )
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
echo "SCRIPT_DIR=${SCRIPT_DIR}"

# Setup environment
source ${SCRIPT_DIR}/setup.sh

# Load ALL_PARAMS
source ${SCRIPT_DIR}/params.sh

# Load TRAIN_DATA_PATH
source ${SCRIPT_DIR}/../train_data/llama3_simulation_15_6t.sh
# NOTE(odashi): For testing on small corpus
#source ${SCRIPT_DIR}/../train_data/small.sh

# Run the trainer script
mpirun \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
        ${ALL_PARAMS[@]} \
        ${TRAIN_DATA_PATH[@]}