#!/bin/bash
#SBATCH --job-name=FIXME
#SBATCH --partition=llmjp-pj
#SBATCH --account=llmjp-pj
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: sbatch [sbatch options] $0 <EXPERIMENT_DIR> <TASK_NAME> <WANDB_PROJECT>"
    exit 1
fi

EXPERIMENT_DIR=$1; shift
TASK_NAME=$1; shift
WANDB_PROJECT=$1; shift

cd "${SLURM_SUBMIT_DIR}"

TASK_DIR=${EXPERIMENT_DIR}/tasks/${TASK_NAME}
JOB_ID=${SLURM_JOB_ID}

ENV_DIR=${EXPERIMENT_DIR}/env
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts

EXPERIMENT_NAME=pretrain_${TASK_NAME}_${JOB_ID}

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

# Set up environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

NUM_NODES=${SLURM_JOB_NUM_NODES}
NUM_GPUS_PER_NODE=$(echo "${SLURM_TASKS_PER_NODE:-8}" | cut -d '(' -f 1)
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"

# For logging
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
scontrol show hostname "${SLURM_JOB_NODELIST}"

# Load training data: TRAIN_DATA_PATH
source "${TASK_DIR}/train_data.sh"

# Load model params: ALL_PARAMS
# Requires TRAIN_DATA_PATH
source "${TASK_DIR}/params.sh"

# Add logging params
ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --moe-per-layer-logging
    --wandb-entity llm-jp
    --wandb-project "${WANDB_PROJECT}"
    --wandb-exp-name "${EXPERIMENT_NAME}"
)

# Add Checkpointing params
BASE_CHECKPOINT_DIR=${TASK_DIR}/base_checkpoints
TASK_CHECKPOINT_DIR=${TASK_DIR}/checkpoints

if [ -e "${TASK_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Resume from the last checkpoint in this task"
    LOAD_DIR=${TASK_CHECKPOINT_DIR}
elif [ -e "${BASE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Start from the base checkpoint"
    LOAD_DIR=${BASE_CHECKPOINT_DIR}
else
    echo "Start from scratch"
    LOAD_DIR=${TASK_CHECKPOINT_DIR}
fi

ALL_PARAMS+=(
    --load "${LOAD_DIR}"
    --save "${TASK_CHECKPOINT_DIR}"
    --save-interval 1000
    --async-save
    --ckpt-format torch_dist
)

# For logging
echo "ALL_PARAMS: ${ALL_PARAMS[*]}"

echo "Start training..."
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
        "${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py" \
        "${ALL_PARAMS[@]}"

echo "Training completed successfully."
