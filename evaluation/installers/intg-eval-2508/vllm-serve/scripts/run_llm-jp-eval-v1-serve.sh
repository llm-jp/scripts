#!/bin/bash
#
# Endpoint-based counterpart of llm-jp-eval v1.4.x's run_llm-jp-eval.sh:
# the dump (dump_prompts.py) and eval (evaluate_llm.py) phases are identical,
# but the inference phase sends prompts to a pre-launched vLLM server via
# inference_openai_v1.py instead of loading the model in-process.
#
# Usage:
#   run_llm-jp-eval-v1-serve.sh MODEL OUTPUT_DIR BASE_URL VERSION_ENV_DIR \
#       [--max_num_samples N] [--client-concurrency N]
#
#   MODEL           Served model name (must equal the server's model id)
#   OUTPUT_DIR      Output directory
#   BASE_URL        e.g. http://127.0.0.1:8000/v1
#   VERSION_ENV_DIR Installed v1.4.x env (e.g. .../environment/llm-jp-eval-v1.4.1)

set -eux -o pipefail

usage() {
    >&2 echo "Usage: $0 MODEL OUTPUT_DIR BASE_URL VERSION_ENV_DIR [--max_num_samples N] [--client-concurrency N]"
    exit 1
}

if [ $# -lt 4 ]; then usage; fi
MODEL_PATH=$1; shift
OUTPUT_DIR=$(realpath $1); shift
BASE_URL=$1; shift
VERSION_ENV_DIR=$(realpath $1); shift

MAX_NUM_SAMPLES=100
CLIENT_CONCURRENCY=256
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_num_samples) MAX_NUM_SAMPLES=$2; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY=$2; shift 2 ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

mkdir -p ${OUTPUT_DIR}

ENV_DIR=${VERSION_ENV_DIR}/environment
CONFIG_DIR=${VERSION_ENV_DIR}/resources
PROMPT_OUTPUT_DIR=${OUTPUT_DIR}/prompts
OFFLINE_OUTPUT_DIR=${OUTPUT_DIR}/offline
RESULT_DIR=${OUTPUT_DIR}/results
LLM_JP_EVAL_DIR=${ENV_DIR}/src/llm-jp-eval

# Provides LLM_JP_EVAL_TAG (used to locate the preprocessed datasets).
source ${ENV_DIR}/scripts/environment.sh
DATASET_DIR=${ENV_DIR}/data/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev

LLM_JP_EVAL_OVERRIDES=(
    model.pretrained_model_name_or_path=${MODEL_PATH}
    tokenizer.pretrained_model_name_or_path=${MODEL_PATH}
    dataset_dir=${DATASET_DIR}
    prompt_dump_dir=${PROMPT_OUTPUT_DIR}
    offline_dir=${OFFLINE_OUTPUT_DIR}
    log_dir=${RESULT_DIR}
    metainfo.max_num_samples=${MAX_NUM_SAMPLES}
)
if [ -n "${HF_HOME:-}" ]; then
    LLM_JP_EVAL_OVERRIDES+=(
        resource_dir=${HF_HOME}
    )
fi

source ${ENV_DIR}/venv-eval/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/scripts/dump_prompts.py \
    -cp ${CONFIG_DIR} \
    -cn config_base \
    hydra.run.dir=${PROMPT_OUTPUT_DIR}/dump_prompts \
    ${LLM_JP_EVAL_OVERRIDES[@]}
deactivate

# Inference via the shared vLLM server (no model load in this process).
# venv-vllm provides the openai and transformers packages.
source ${ENV_DIR}/venv-vllm/bin/activate
python \
    ${SCRIPT_DIR}/inference_openai_v1.py \
    --base-url ${BASE_URL} \
    --model ${MODEL_PATH} \
    --prompt-json-path "${PROMPT_OUTPUT_DIR}/*.eval-prompt.json" \
    --output-dir ${OFFLINE_OUTPUT_DIR} \
    --num-concurrent ${CLIENT_CONCURRENCY}
deactivate

source ${ENV_DIR}/venv-eval/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    -cp ${CONFIG_DIR} \
    -cn config_base \
    hydra.run.dir=${RESULT_DIR}/evaluate_llm \
    ${LLM_JP_EVAL_OVERRIDES[@]}
deactivate

echo "Done"
