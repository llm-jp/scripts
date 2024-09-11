#!/bin/bash
# Model conversion script for FP8 experiment.
# Usage:
#   sbatch /path/to/convert.sh SRC_DIR DEST_DIR
#
#SBATCH --job-name=0031_convert
#SBATCH --partition=<FIX_ME>
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 SRC_DIR DEST_DIR"
    exit 1
fi

SRC_DIR=$1; shift
DEST_DIR=$1; shift

if [ -e ${DEST_DIR} ]; then
    >&2 echo "DEST_DIR already exists: ${DEST_DIR}"
    exit 1
fi

ENV_DIR=environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

TOKENIZER_MODEL_DIR=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2

python ${ENV_DIR}/src/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama2_hf \
  --load-dir ${SRC_DIR} \
  --save-dir ${DEST_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path ${ENV_DIR}/src/Megatron-LM

cp ${TOKENIZER_MODEL_DIR}/* ${DEST_DIR}

echo "Done"
