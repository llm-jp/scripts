#!/bin/bash
#SBATCH --job-name=0309
#SBATCH --partition=llmjp-pj
#SBATCH --account=llmjp-pj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/convert-%j.out
#SBATCH --error=logs/convert-%j.err

cd "${SLURM_SUBMIT_DIR}"
set -eu -o pipefail

ENV_DIR=$1
HF_FORMAT_DIR=$2
MCORE_FORMAT_DIR=$3
TARGET_TP_SIZE=$4
TARGET_PP_SIZE=$5

# Setup environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver4.0/alpha_1.0
MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

mkdir -p "${MCORE_FORMAT_DIR}"

python "${MEGATRON_PATH}"/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --saver mcore \
    --target-tensor-parallel-size "${TARGET_TP_SIZE}" \
    --target-pipeline-parallel-size "${TARGET_PP_SIZE}" \
    --load-dir "${HF_FORMAT_DIR}" \
    --save-dir "${MCORE_FORMAT_DIR}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --bf16 \
    --model-size llama2-7B \
    --checkpoint-type hf \
    --make-vocab-size-divisible-by 512 \
    --saver-transformer-impl "transformer_engine"
