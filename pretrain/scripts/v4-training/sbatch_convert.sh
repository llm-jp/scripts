#!/bin/bash
#SBATCH --job-name=FIXME
#SBATCH --partition=FIXME
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# LLM-jp v4 model converter
# Usage:
#   sbatch \
#     --job-name=9999_convert \
#     --partition=gpu \
#     --outpout=convert_%j.log \
#     sbatch_train.sh \
#       /path/to/env \     ... ENV_DIR: path to the trainer environment
#       /path/to/model \   ... MODEL_DIR: path to the model to save
#       7.7b-llama3-ecjk \ ... PARAM_NAME: model config; corresponding file in `params/` should exist
#       123000             ... Checkpoint step to convert

set -eu -o pipefail

# Arguments
if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 ENV_DIR MODEL_DIR PARAM_NAME ITER"
    exit 1
fi
ENV_DIR=$(realpath -eP $1); shift
MODEL_DIR=$(realpath -m $1); shift
PARAM_NAME=$1; shift
ITER=$1; shift
echo "ENV_DIR=${ENV_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "PARAM_NAME=${PARAM_NAME}"
echo "ITER=${ITER}"

ITER_NAME=iter_$(printf %07d ${ITER})  # iter_0123456

MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
TOKENIZER_MODEL_PATH=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2
OUTPUT_DIR=${MODEL_DIR}/checkpoints_hf/${ITER_NAME}

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

# Setup working directory
TEMP_DIR=$(mktemp -d /nvme56/converter_${SLURM_JOB_ID}_XXXXXX)
echo "TEMP_DIR=${TEMP_DIR}"
function rm_tempdir {
    if [ -e ${TEMP_DIR} ]; then
        echo "Removing remporary directory: ${TEMP_DIR}"
        rm -rf ${TEMP_DIR}
        echo "Done removing"
    fi
}
trap rm_tempdir EXIT
trap 'trap - EXIT; rm_tempdir; exit 1' INT PIPE TERM

########
# Step 1: Convert `torch_dist` format to `torch`
# This process requires to launch the trainer script with the same parallelism configs.
########
echo "Start converting: torch_dist --> torch"

# Prepare source model at specific iteration
mkdir ${TEMP_DIR}/torch_dist
echo ${ITER} > ${TEMP_DIR}/torch_dist/latest_checkpointed_iteration.txt
ln -s ${MODEL_DIR}/checkpoints/${ITER_NAME} ${TEMP_DIR}/torch_dist/${ITER_NAME}

# Load ALL_PARAMS
source ${SCRIPT_DIR}/params/${PARAM_NAME}.sh

# Load TRAIN_DATA_PATH
source ${SCRIPT_DIR}/train_data/llama3_simulation_15_6t.sh

# Add params specific to model conversion
ALL_PARAMS+=(
    --ckpt-convert-format torch
    --ckpt-convert-save ${TEMP_DIR}
)

# NOTE(odashi): We don't need to set W&B parmas

# Launch trainer script to convert the checkpoint
mpirun \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python ${MEGATRON_PATH}/pretrain_gpt.py \
        ${ALL_PARAMS[@]} \
        ${TRAIN_DATA_PATH[@]}

echo "Files created by the Step 1:"
find ${TEMP_DIR}/torch | sort

########
# Step 2: Convert `torch` to `Hugging Face Llama2`
########

echo "Start converting: torch --> hf"

python ${MEGATRON_PATH}/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mcore \
    --saver llmjp4_hf \
    --load-dir ${TEMP_DIR}/torch \
    --save-dir ${OUTPUT_DIR} \
    --hf-tokenizer-path ${TOKENIZER_MODEL_PATH} \
    --save-dtype bfloat16 \
    --loader-transformer-impl transformer_engine \
    --megatron-path ${MEGATRON_PATH}

echo "Files created by the Step 2:"
find ${OUTPUT_DIR} | sort

########
# Step 3: Replace tokenizer model
########

echo "Start replacing tokenizer"

cp ${TOKENIZER_MODEL_PATH}/* ${OUTPUT_DIR}

echo "Final model files:"
find ${OUTPUT_DIR} | sort

echo "Done processing"
