#!/bin/bash

set -eu -o pipefail

if [ $# -ne 6 ]; then
    >&2 echo "Usage: $0 <RESERVATION_ID> <EXPERIMENT_ID> <EXPERIMENT_DIR> <TASK_NAME> <TOKENIZER_DIR> <NUM_NODES>"
    >&2 echo "Example: $0 R0123456789 0123 /path/to/0123_experiment task_name /path/to/tokenizer 1"
    exit 1
fi

# NOTE(odashi):
# Some variables are not used, but maintained for compatibility with training script.
RESERVATION_ID=$1; shift
EXPERIMENT_ID=$1; shift
EXPERIMENT_DIR=$1; shift
TASK_NAME=$1; shift
TOKENIZER_DIR=$1; shift
NUM_NODES=$1; shift

# This directory
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TASK_DIR=${EXPERIMENT_DIR}/tasks/${TASK_NAME}
LAST_ITER=$(cat ${TASK_DIR}/checkpoints/latest_checkpointed_iteration.txt)

dependency=()

for iter in $(seq 1000 1000 ${LAST_ITER}); do
    if [ ! -e ${TASK_DIR}/checkpoints/iter_$(printf '%07d' ${iter}) ]; then
        #echo "Skip iter=${iter}: Source model does not exist."
        continue
    fi
    if [ -e ${TASK_DIR}/checkpoints_hf/iter_$(printf '%07d' ${iter})/tokenizer.json ]; then
        #echo "Skip iter=${iter}: Converted model already exists."
        continue
    fi

    # NOTE(odashi): RTYPE=rt_HG doesn't work for 8B models.
    job_id=$(qsub \
        ${dependency[@]} \
        -P gcg51557 \
        -q ${RESERVATION_ID} \
        -N ${EXPERIMENT_ID}_convert \
        -l select=${NUM_NODES},walltime=6:00:00 \
        -v RTYPE=rt_HF,EXPERIMENT_DIR=${EXPERIMENT_DIR},TASK_NAME=${TASK_NAME},ITER=${iter},TOKENIZER_DIR=${TOKENIZER_DIR} \
        -o /dev/null \
        -e /dev/null \
        -m n \
        ${SCRIPT_ROOT}/qsub_convert.sh
    )
    echo "Submitted iter=${iter}: job_id=${job_id}"
    #dependency=(-W depend=afterany:${job_id})
done
