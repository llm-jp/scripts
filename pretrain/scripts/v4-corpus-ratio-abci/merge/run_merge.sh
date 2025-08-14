#!/bin/bash

set -eu -o pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 <RESERVATION_ID> <EXPERIMENT_ID> <EXPERIMENT_DIR> <TASK_NAME>"
    >&2 echo "Example: $0 R0123456789 0123 /path/to/0123_experiment task_name"
    exit 1
fi

# NOTE(odashi):
# Some variables are not used, but maintained for compatibility with training script.
RESERVATION_ID=$1; shift
EXPERIMENT_ID=$1; shift
EXPERIMENT_DIR=$1; shift
TASK_NAME=$1; shift

# This directory
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

qsub \
    -P gcg51557 \
    -q ${RESERVATION_ID} \
    -N ${EXPERIMENT_ID}_merge \
    -l select=1,walltime=6:00:00 \
    -v RTYPE=rt_HG,EXPERIMENT_DIR=${EXPERIMENT_DIR},TASK_NAME=${TASK_NAME} \
    -o /dev/null \
    -e /dev/null \
    -m n \
    ${SCRIPT_ROOT}/qsub_merge.sh
