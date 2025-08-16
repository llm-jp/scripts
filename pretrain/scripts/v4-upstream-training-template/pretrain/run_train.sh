#!/bin/bash

set -eu -o pipefail

if [ $# -ne 6 ]; then
    >&2 echo "Usage: $0 <RESERVATION_ID> <EXPERIMENT_ID> <EXPERIMENT_DIR> <TASK_NAME> <WANDB_PROJECT> <NUM_NODES>"
    >&2 echo "Example: $0 R0123456789 0123 /path/to/0123_experiment task_name 0123_experiment 32"
    exit 1
fi

RESERVATION_ID=$1; shift
EXPERIMENT_ID=$1; shift
EXPERIMENT_DIR=$1; shift
TASK_NAME=$1; shift
WANDB_PROJECT=$1; shift
NUM_NODES=$1; shift

# This directory
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WALLTIME=720:00:00 # 30 days
# WALLTIME=01:00:00 # 1 hour

qsub \
    -P gcg51557 \
    -q ${RESERVATION_ID} \
    -N ${EXPERIMENT_ID}_pretrain \
    -l select=${NUM_NODES},walltime=${WALLTIME} \
    -v RTYPE=rt_HF,EXPERIMENT_DIR=${EXPERIMENT_DIR},TASK_NAME=${TASK_NAME},WANDB_PROJECT=${WANDB_PROJECT} \
    -o /dev/null \
    -e /dev/null \
    -m n \
    ${SCRIPT_ROOT}/qsub_train.sh

