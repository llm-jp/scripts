#!/bin/bash

set -eu -o pipefail

if [ $# -ne 7 ]; then
    >&2 echo "Usage: $0 <RESERVATION_ID> <EXPERIMENT_ID> <ENV_DIR> <TASK_ROOT_DIR> <TASK_NAME> <WANDB_PROJECT> <NUM_NODES>"
    >&2 echo "Example: $0 R0123456789 0123 /path/to/installed/env /path/to/tasks TASKNAME 0174_CORPUS_RATIO 32"
    exit 1
fi

# NOTE(odashi):
# Some variables are not used, but maintained for compatibility with training script.
RESERVATION_ID=$1; shift
EXPERIMENT_ID=$1; shift
ENV_DIR=$1; shift
TASK_ROOT_DIR=$1; shift
TASK_NAME=$1; shift
WANDB_PROJECT=$1; shift
NUM_NODES=$1; shift

# This directory
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ITER=$(cat ${TASK_ROOT_DIR}/${TASK_NAME}/checkpoints/latest_checkpointed_iteration.txt)

qsub \
    -P gcg51557 \
    -q ${RESERVATION_ID} \
    -N ${EXPERIMENT_ID}_convert \
    -l select=${NUM_NODES},walltime=1000:00:00 \
    -v RTYPE=rt_HF,ENV_DIR=${ENV_DIR},TASK_ROOT_DIR=${TASK_ROOT_DIR},TASK_NAME=${TASK_NAME},ITER=${ITER} \
    -o /dev/null \
    -e /dev/null \
    ${SCRIPT_ROOT}/qsub_convert.sh

