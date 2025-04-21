#!/bin/bash
#SBATCH --job-name=FIXME
#SBATCH --partition=FIXME
#SBATCH --nodes=0
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# LLM-jp v4 trainer launcher
# Usage:
#   sbatch \
#     --job-name=9999_train \
#     --partition=gpu \
#     --nodes=64 \
#     --outpout=train_%j.log \
#     sbatch_train.sh \
#       /path/to/env \            ... ENV_DIR: path to the trainer environment
#       /path/to/model \          ... MODEL_DIR: path to the model to save
#       7.7b-llama3-ecjk \        ... PARAM_NAME: model config; corresponding file in `params/` should exist
#       llama3-simulation_15_6t \ ... DATASET_NAME: training dataset config: corresponding file in `train_data` must exist
#       llm-jp \                  ... WANDB_ENTITY
#       9999_train                ... WANDB_PROJECT

set -eu -o pipefail

# Arguments
if [ $# -ne 6 ]; then
    >&2 echo "Usage: $0 ENV_DIR MODEL_DIR PARAM_NAME DATASET_NAME WANDB_ENTITY WANDB_PROJECT"
    exit 1
fi
ENV_DIR=$(realpath -eP $1); shift
MODEL_DIR=$(realpath -m $1); shift
PARAM_NAME=$1; shift
DATASET_NAME=$1; shift
WANDB_ENTITY=$1; shift
WANDB_PROJECT=$1; shift
echo "ENV_DIR=${ENV_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "PARAM_NAME=${PARAM_NAME}"
echo "DATASET_NAME=${DATASET_NAME}"

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
source ${SCRIPT_DIR}/common/setup.sh

# Load ALL_PARAMS
source ${SCRIPT_DIR}/params/${PARAM_NAME}.sh

# Load TRAIN_DATA_PATH
source ${SCRIPT_DIR}/train_data/${DATASET_NAME}.sh

# Add W&B params
ALL_PARAMS+=(
    --wandb-entity ${WANDB_ENTITY}
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name train_$(date '+%Y%m%d-%H%M%S')_${SLURM_JOB_ID}
)

# Add Checkpointing params
TASK_CHECKPOINT_DIR=${MODEL_DIR}/checkpoints
ALL_PARAMS+=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
    --save-interval 1000
)

# Run the trainer script
mpirun \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
        ${ALL_PARAMS[@]} \
        ${TRAIN_DATA_PATH[@]}

echo "Done"