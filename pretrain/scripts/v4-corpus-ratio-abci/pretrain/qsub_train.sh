#!/bin/bash

# Predefined variables:
# * ENV_DIR: Directory containing the Megatron-LM environment
# * TASK_DIR: Directory for the task, containing train_data.sh and logs
# * WANDB_PROJECT: W&B project name

cd $PBS_O_WORKDIR

job_id=${PBS_JOBID%%.*}
mkdir -p ${TASK_DIR}/logs
logfile=${TASK_DIR}/logs/pretrain-${job_id}.out
errfile=${TASK_DIR}/logs/pretrain-${job_id}.err
exec > $logfile 2> $errfile

set -eu -o pipefail

# This directory
script_root=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Load common environment variables
source ${ENV_DIR}/scripts/environment.sh

# Load modules
source /etc/profile.d/modules.sh
module load cuda/${PRETRAIN_CUDA_VERSION}/${PRETRAIN_CUDA_VERSION}.${PRETRAIN_CUDA_VERSION_PATCH}
module load cudnn/${PRETRAIN_CUDNN_VERSION}/${PRETRAIN_CUDNN_VERSION_WITH_PATCH}
module load hpcx/${PRETRAIN_HPCX_VERSION}
module load nccl/${PRETRAIN_NCCL_VERSION}/${PRETRAIN_NCCL_VERSION_WITH_PATCH}
echo $(module list)

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

# Set up environment variables for distributed training
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))
echo "hostname: ${MASTER_ADDR}"

num_nodes=$(wc -l < $PBS_NODEFILE)
num_gpus_per_node=8
num_gpus=$((${num_nodes} * ${num_gpus_per_node}))
echo "nnodes: ${num_nodes}; ngpus: ${num_gpus}"
echo NUM_NODES=$num_nodes
echo NUM_GPUS_PER_NODE=$num_gpus_per_node
echo NUM_GPUS=$num_gpus

cat $PBS_NODEFILE

# Training steps
TRAIN_ITERS=$(cat ${TASK_DIR}/train_iters.txt)

# Training data: TRAIN_DATA_PATH
source ${TASK_DIR}/train_data.sh

# Synthesize all model params: ALL_PARAMS
# Requires TRAIN_ITERS and TRAIN_DATA_PATH
source ${TASK_DIR}/params.sh

# Add W&B params
ALL_PARAMS+=(
    --wandb-entity llm-jp
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name pretrain_${JOBID}
)

# Add Checkpointing params
task_checkpoint_dir=${TASK_DIR}/checkpoints
ALL_PARAMS+=(
    --load ${task_checkpoint_dir}
    --save ${task_checkpoint_dir}
    --save-interval 1000
)

echo "ALL_PARAMS: ${ALL_PARAMS[@]}"

mpirun \
  --display-allocation \
  --report-bindings \
  --oversubscribe \
  -np $num_gpus \
  --npernode $num_gpus_per_node \
  -bind-to none \
  -map-by slot \
  python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
    ${ALL_PARAMS[@]}
