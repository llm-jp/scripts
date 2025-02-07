#!/bin/bash
#
# Usage:
#   sbatch tokenize_sbatch.sh PRETRAIN_ENV_DIR INPUT_FILE OUTPUT_PREFIX

set -eu -o pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage $0 PRETRAIN_ENV_DIR INPUT_FILE OUTPUT_PREFIX WORKERS"
fi

PRETRAIN_ENV_DIR=$(realpath -eP $1); shift
INPUT_FILE=$(realpath -eP $1); shift
OUTPUT_PREFIX=$(realpath -mP $1); shift
WORKERS=$1; shift

echo "PRETRAIN_ENV_DIR=${PRETRAIN_ENV_DIR}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
echo "WORKERS=${WORKERS}"

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

mkdir -p $(dirname ${OUTPUT_PREFIX})

# Make temporary file for decompressed inputs
mkdir -p /nvme12/tmp_0111
DECOMPRESSED_INPUT_FILE=$(mktemp /nvme12/tmp_0111/decompressed.XXXXXX)
echo "Input is decompressed to ${DECOMPRESSED_INPUT_FILE}"

function rm_tempfile {
    if [ -f ${DECOMPRESSED_INPUT_FILE} ]; then
        rm -f ${DECOMPRESSED_INPUT_FILE}
    fi
}
trap rm_tempfile EXIT
trap 'trap - EXIT; rm_tempfile; exit 1' INT PIPE TERM

# Decompress input
echo "Start decompressing input"
gunzip -c ${INPUT_FILE} > ${DECOMPRESSED_INPUT_FILE}

source ${PRETRAIN_ENV_DIR}/scripts/environment.sh
source ${PRETRAIN_ENV_DIR}/venv/bin/activate

# Perform tokenization
echo "Start tokenization"
python ${PRETRAIN_ENV_DIR}/src/Megatron-LM/tools/preprocess_data.py \
    --input ${DECOMPRESSED_INPUT_FILE} \
    --output-prefix ${OUTPUT_PREFIX} \
    --tokenizer-model ${PRETRAIN_ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model \
    --tokenizer-type Llama2Tokenizer \
    --workers $WORKERS \
    --append-eod

# Count number of tokens
echo "Counting tokens"
python ${SCRIPT_DIR}/count_tokens.py \
    --megatron-path ${PRETRAIN_ENV_DIR}/src/Megatron-LM \
    --prefix ${OUTPUT_PREFIX}

deactivate
echo "Done"
