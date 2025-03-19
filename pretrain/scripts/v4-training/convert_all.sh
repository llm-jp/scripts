#!/bin/bash

set -eu -o pipefail

# Arguments
if [ $# -ne 6 ]; then
    >&2 echo "Usage: $0 JOB_NAME PARTITION LOG_DIR ENV_DIR MODEL_DIR PARAM_NAME"
    exit 1
fi

JOB_NAME=$1; shift
PARTITION=$1; shift
LOG_DIR=$(realpath -eP $1); shift
ENV_DIR=$(realpath -eP $1); shift
MODEL_DIR=$(realpath -eP $1); shift
PARAM_NAME=$1; shift

# Find the script directory
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(
        scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/Command=/{print $2}' \
        | cut -d ' ' -f 1
    )
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
echo "SCRIPT_DIR=${SCRIPT_DIR}"

for iter_name in $(ls ${MODEL_DIR}/checkpoints | grep iter_.......); do
    if [ ${iter_name} == iter_0000000 ]; then
        iter=0
    else
        iter=$(echo ${iter_name} | sed 's/iter_0*//g')
    fi

    if [ -e ${MODEL_DIR}/checkpoints_hf/${iter_name}/tokenizer.json ]; then
        continue
    fi

    echo ${iter}

    sbatch \
        --job-name=${JOB_NAME} \
        --partition=${PARTITION} \
        --output=${LOG_DIR}/convert_%j.log \
        ${SCRIPT_DIR}/sbatch_convert.sh \
        ${ENV_DIR} \
        ${MODEL_DIR} \
        ${PARAM_NAME} \
        ${iter}
done
