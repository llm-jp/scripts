#!/bin/bash

# Predefined variables:
# * ENV_DIR: Directory containing the Megatron-LM environment
# * TASK_ROOT_DIR: Directory for the task, containing train_data.sh and logs
# * TASK_NAME: Name of the task
# * ITER: Target iteration number

cd ${PBS_O_WORKDIR}

TASK_DIR=${TASK_ROOT_DIR}/${TASK_NAME}
JOB_ID=${PBS_JOBID%%.*}

mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/convert-${JOB_ID}.out
ERRFILE=${TASK_DIR}/logs/convert-${JOB_ID}.err
exec > ${LOGFILE} 2> ${ERRFILE}

set -eu -o pipefail

# This directory
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Load common environment variables
source ${ENV_DIR}/scripts/environment.sh

# Load modules
source /etc/profile.d/modules.sh
module load cuda/${PRETRAIN_CUDA_VERSION}/${PRETRAIN_CUDA_VERSION}.${PRETRAIN_CUDA_VERSION_PATCH}
module load cudnn/${PRETRAIN_CUDNN_VERSION}/${PRETRAIN_CUDNN_VERSION_WITH_PATCH}
module load hpcx/${PRETRAIN_HPCX_VERSION}
module load nccl/${PRETRAIN_NCCL_VERSION}/${PRETRAIN_NCCL_VERSION_WITH_PATCH}
# For logging
module list

# Load Python venv
source ${ENV_DIR}/venv/bin/activate

## Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE} | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))
echo "hostname: ${MASTER_ADDR}"

ITER_NAME=iter_$(printf %07d ${ITER})  # iter_0123456

MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
TOKENIZER_MODEL_PATH=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2
OUTPUT_DIR=${TASK_DIR}/checkpoints_hf/${ITER_NAME}

# Setup working directory
TEMP_DIR=$(mktemp -d "${TASK_DIR}/tmp_converter_${JOB_ID}_XXXXXX")
echo "TEMP_DIR=${TEMP_DIR}"
function rm_tempdir {
    if [ -e ${TEMP_DIR} ]; then
        echo "Removing temporary directory: ${TEMP_DIR}"
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
ln -s ${TASK_DIR}/checkpoints/${ITER_NAME} ${TEMP_DIR}/torch_dist/${ITER_NAME}

# Training steps
TRAIN_ITERS=$(cat ${TASK_DIR}/train_iters.txt)

# Training data: TRAIN_DATA_PATH
source ${TASK_DIR}/train_data.sh

# Synthesize all model params: ALL_PARAMS
# Requires TRAIN_ITERS and TRAIN_DATA_PATH
source ${TASK_DIR}/params.sh

# Add params for model conversion
ALL_PARAMS+=(
    --load ${TEMP_DIR}/torch_dist
    --ckpt-convert-format torch
    --ckpt-convert-save ${TEMP_DIR}
)

echo "ALL_PARAMS: ${ALL_PARAMS[@]}"

NUM_NODES=$(wc -l < ${PBS_NODEFILE})
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "nnodes: ${NUM_NODES}; ngpus: ${NUM_GPUS}"
echo NUM_NODES=${NUM_NODES}
echo NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}
echo NUM_GPUS=${NUM_GPUS}

# Launch trainer script to convert the checkpoint
mpirun \
    --display-allocation \
    --report-bindings \
    --oversubscribe \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python \
        ${MEGATRON_PATH}/pretrain_gpt.py \
        ${ALL_PARAMS[@]}

echo "Files created by the Step 1:"
find ${TEMP_DIR}/torch | sort

########
# Step 2: Convert `torch` to `Hugging Face`
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
