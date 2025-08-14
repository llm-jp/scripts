#!/bin/bash

# Predefined variables:
# * EXPERIMENT_DIR: Directory containing the Megatron-LM environment
# * TASK_NAME: Name of the task

cd ${PBS_O_WORKDIR}

TASK_DIR=${EXPERIMENT_DIR}/tasks/${TASK_NAME}
JOB_ID=${PBS_JOBID%%.*}

mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/merge-${JOB_ID}.out
ERRFILE=${TASK_DIR}/logs/merge-${JOB_ID}.err
exec > ${LOGFILE} 2> ${ERRFILE}

set -eu -o pipefail

ENV_DIR=${EXPERIMENT_DIR}/env
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts

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

models=$(
    for i in $(cat ${TASK_DIR}/merge_iters.txt); do
        echo ${TASK_DIR}/checkpoints_hf/iter_$(printf '%07d' ${i})
    done
)
echo "Models to merge:"
for model in ${models}; do
    echo "  ${model}"
done

python ${SCRIPT_DIR}/pretrain/scripts/v4-corpus-ratio-abci/merge/merge.py \
    --source-models ${models[@]} \
    --output-model ${TASK_DIR}/checkpoints_hf_merged

echo "Done processing"
