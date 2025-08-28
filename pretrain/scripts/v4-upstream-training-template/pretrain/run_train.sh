#!/bin/bash

set -eu -o pipefail

if [ $# -ne 8 ]; then
    >&2 echo "Usage: $0 <GROUP_ID> <RESERVATION_ID> <JOB_NAME> <EXPERIMENT_DIR> <TASK_NAME> <WANDB_PROJECT> <NUM_NODES> <WALLTIME>"
    >&2 echo "Example: $0 gcg51557 R0123456789 0123 /path/to/0123_experiment task_name 0123_experiment 32 720:00:00"
    exit 1
fi

GROUP_ID=$1; shift
RESERVATION_ID=$1; shift
JOB_NAME=$1; shift
EXPERIMENT_DIR=$1; shift
TASK_NAME=$1; shift
WANDB_PROJECT=$1; shift
NUM_NODES=$1; shift
WALLTIME=$1; shift

# This directory
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

qsub \
    -P ${GROUP_ID} \
    -q ${RESERVATION_ID} \
    -N ${JOB_NAME} \
    -l select=${NUM_NODES},walltime=${WALLTIME} \
    -v RTYPE=rt_HF,USE_SSH=1,EXPERIMENT_DIR=${EXPERIMENT_DIR},TASK_NAME=${TASK_NAME},WANDB_PROJECT=${WANDB_PROJECT} \
    -o /dev/null \
    -e /dev/null \
    -m n \
    ${SCRIPT_ROOT}/qsub_train.sh

