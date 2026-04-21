#!/bin/bash
#SBATCH --job-name=FIXME
#SBATCH --partition=llmjp-pj
#SBATCH --account=llmjp-pj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/convert-%j.out
#SBATCH --error=logs/convert-%j.err

set -eu -o pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage: sbatch [sbatch options] $0 <EXPERIMENT_DIR> <TASK_NAME> <ITER> <TOKENIZER_DIR>"
    exit 1
fi

EXPERIMENT_DIR=$1; shift
TASK_NAME=$1; shift
ITER=$1; shift
TOKENIZER_DIR=$1; shift

cd "${SLURM_SUBMIT_DIR}"

TASK_DIR=${EXPERIMENT_DIR}/tasks/${TASK_NAME}
JOB_ID=${SLURM_JOB_ID}

ENV_DIR=${EXPERIMENT_DIR}/env
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts

# Load common environment variables & Python venv
source "${ENV_DIR}/scripts/environment.sh"
source "${ENV_DIR}/venv/bin/activate"

## Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

export MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

ITER_NAME=iter_$(printf %07d "${ITER}")

MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
OUTPUT_DIR=${TASK_DIR}/checkpoints_hf/${ITER_NAME}

TEMP_DIR=$(mktemp -d "${TASK_DIR}/tmp_converter_${JOB_ID}_XXXXXX")
echo "TEMP_DIR=${TEMP_DIR}"
rm_tempdir() {
    if [ -e "${TEMP_DIR}" ]; then
        echo "Removing temporary directory: ${TEMP_DIR}"
        rm -rf "${TEMP_DIR}"
        echo "Done removing"
    fi
}
trap rm_tempdir EXIT
trap 'trap - EXIT; rm_tempdir; exit 1' INT PIPE TERM

########
# Step 1: Convert `torch_dist` format to `torch`
# This process requires launching the trainer script with the same parallelism configs.
########
echo "Start converting: torch_dist --> torch"

mkdir -p "${TEMP_DIR}/torch_dist"
echo "${ITER}" > "${TEMP_DIR}/torch_dist/latest_checkpointed_iteration.txt"
ln -s "${TASK_DIR}/checkpoints/${ITER_NAME}" "${TEMP_DIR}/torch_dist/${ITER_NAME}"

# Load training data: TRAIN_DATA_PATH
source "${TASK_DIR}/train_data.sh"

# Load model params: ALL_PARAMS
# Requires TRAIN_ITERS and TRAIN_DATA_PATH
source "${TASK_DIR}/params.sh"

ALL_PARAMS+=(
    --load "${TEMP_DIR}/torch_dist"
    --ckpt-convert-format torch
    --ckpt-convert-save "${TEMP_DIR}"
)

echo "ALL_PARAMS: ${ALL_PARAMS[*]}"

NUM_NODES=${SLURM_JOB_NUM_NODES}
NUM_GPUS_PER_NODE=$(echo "${SLURM_TASKS_PER_NODE:-8}" | cut -d '(' -f 1)
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
scontrol show hostname "${SLURM_JOB_NODELIST}"

mpirun \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    -x EXPERIMENT_DIR=${EXPERIMENT_DIR} \
    -x MASTER_ADDR=${MASTER_ADDR} \
    -x MASTER_PORT=${MASTER_PORT} \
    -x NUM_NODES=${NUM_NODES} \
    -x NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} \
    python \
        "${MEGATRON_PATH}/pretrain_gpt.py" \
        "${ALL_PARAMS[@]}"

echo "Files created by Step 1:"
find "${TEMP_DIR}/torch" | sort

########
# Step 2: Convert `torch` to `Hugging Face`
########
echo "Start converting: torch --> hf"

mkdir -p "${OUTPUT_DIR}"

python "${MEGATRON_PATH}/tools/checkpoint/convert.py" \
    --model-type GPT \
    --loader mcore \
    --saver llmjp4_hf \
    --load-dir "${TEMP_DIR}/torch" \
    --save-dir "${OUTPUT_DIR}" \
    --hf-tokenizer-path "${TOKENIZER_DIR}" \
    --save-dtype bfloat16 \
    --loader-transformer-impl transformer_engine \
    --megatron-path "${MEGATRON_PATH}"

echo "Files created by Step 2:"
find "${OUTPUT_DIR}" | sort

########
# Step 3: Replace tokenizer model
########
echo "Start replacing tokenizer"

cp "${TOKENIZER_DIR}"/* "${OUTPUT_DIR}"

echo "Final model files:"
find "${OUTPUT_DIR}" | sort

echo "Done processing"
