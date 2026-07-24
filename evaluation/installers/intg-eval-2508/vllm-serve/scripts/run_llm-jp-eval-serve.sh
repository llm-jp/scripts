#!/bin/bash
#
# Endpoint-based counterpart of the llm-jp-eval v2.x run_llm-jp-eval.sh:
# the dump and eval phases are identical, but the inference phase sends
# prompts to a pre-launched vLLM server (via inference_openai.py) instead of
# loading the model in-process.
#
# Usage:
#   run_llm-jp-eval-serve.sh MODEL OUTPUT_DIR BASE_URL VERSION_ENV_DIR \
#       [--max_num_samples N] [--apply_chat_template] [--tokenize_kwargs JSON] \
#       [--client-concurrency N] [--basemodel]
#
#   --client-concurrency N  Prompts kept in flight against the server
#       (default: 256); becomes server.num_concurrent of inference_openai.py
#       (single-prompt requests, so the count maps 1:1).
#   --basemodel  Base-model (pretrained checkpoint) evaluation: fixed prompt
#       template (config_basemodel.yaml), add_special_tokens=False,
#       temperature=0.0, and the 4-shot datasets only (only_4shots.yaml).
#       Requires the basemodel resources of the llm-jp-eval v2.1.5 installer.
#
#   MODEL           Served model name (must equal the server's model id)
#   OUTPUT_DIR      Output directory
#   BASE_URL        e.g. http://127.0.0.1:8000/v1
#   VERSION_ENV_DIR Installed llm-jp-eval env (e.g. .../environment/llm-jp-eval-v2.1.5)
#
# NOTE: --reasoning_parser is intentionally unsupported here; use the offline
# run_llm-jp-eval.sh for models that need one.

set -eux -o pipefail

usage() {
    >&2 echo "Usage: $0 MODEL OUTPUT_DIR BASE_URL VERSION_ENV_DIR [--max_num_samples N] [--apply_chat_template] [--tokenize_kwargs JSON] [--basemodel]"
    exit 1
}

if [ $# -lt 4 ]; then usage; fi
MODEL_PATH=$1; shift
OUTPUT_DIR=$(realpath $1); shift
BASE_URL=$1; shift
VERSION_ENV_DIR=$(realpath $1); shift

MAX_NUM_SAMPLES=100
APPLY_CHAT_TEMPLATE=false
TOKENIZE_KWARGS=""
CLIENT_CONCURRENCY=256
BASEMODEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_num_samples) MAX_NUM_SAMPLES=$2; shift 2 ;;
        --apply_chat_template) APPLY_CHAT_TEMPLATE=true; shift ;;
        --tokenize_kwargs) TOKENIZE_KWARGS=$2; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY=$2; shift 2 ;;
        --basemodel) BASEMODEL=true; shift ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

if [ "${BASEMODEL}" = true ] && { [ "${APPLY_CHAT_TEMPLATE}" = true ] || [ -n "${TOKENIZE_KWARGS}" ]; }; then
    >&2 echo "Error: --basemodel cannot be combined with --apply_chat_template / --tokenize_kwargs."
    exit 1
fi

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

mkdir -p ${OUTPUT_DIR}

ENV_DIR=${VERSION_ENV_DIR}/environment
CONFIG_DIR=${VERSION_ENV_DIR}/resources
PROMPT_OUTPUT_DIR=${OUTPUT_DIR}/prompts
OFFLINE_OUTPUT_DIR=${OUTPUT_DIR}/offline
RESULT_DIR=${OUTPUT_DIR}/results
LLM_JP_EVAL_DIR=${ENV_DIR}/src/llm-jp-eval
DATASET_DIR=${ENV_DIR}/data/llm-jp-eval
VLLM_VENV=${LLM_JP_EVAL_DIR}/llm-jp-eval-inference/inference-modules/vllm/.venv

CONFIG_FILE=config_base.yaml
if [ "${BASEMODEL}" = true ]; then
    CONFIG_FILE=config_basemodel.yaml
    if [ ! -f "${CONFIG_DIR}/${CONFIG_FILE}" ]; then
        >&2 echo "ERROR: ${CONFIG_DIR}/${CONFIG_FILE} not found; --basemodel requires an llm-jp-eval installation that ships the basemodel resources (v2.1.5 or later)."
        exit 1
    fi
fi

# Code-execution datasets (mbpp, jhumaneval) require the dify-sandbox container.
# Skip them when no container runtime is available or DISABLE_CODE_EXEC=1 is set.
if [ "${BASEMODEL}" = true ]; then
    # only_4shots.yaml contains no code-execution datasets, so no sandbox is needed.
    EVAL_DATASET_CONFIG_PATH=${LLM_JP_EVAL_DIR}/eval_configs/only_4shots.yaml
    ENABLE_CODE_EXEC=false
elif command -v singularity >/dev/null 2>&1 && [ "${DISABLE_CODE_EXEC:-0}" != "1" ]; then
    EVAL_DATASET_CONFIG_PATH=${LLM_JP_EVAL_DIR}/eval_configs/all_datasets.yaml
    ENABLE_CODE_EXEC=true
else
    ENABLE_CODE_EXEC=false
    >&2 echo "WARNING: singularity is unavailable (or DISABLE_CODE_EXEC=1); skipping code-execution datasets (mbpp, jhumaneval) and the CG category."
    EVAL_DATASET_CONFIG_PATH=${OUTPUT_DIR}/all_datasets_no_code_exec.yaml
fi

source ${LLM_JP_EVAL_DIR}/.venv/bin/activate

if [ "${ENABLE_CODE_EXEC}" = false ] && [ "${BASEMODEL}" = false ]; then
    python -c "
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg['datasets'] = [d for d in cfg['datasets'] if d not in ('mbpp', 'jhumaneval')]
cfg.get('categories', {}).pop('CG', None)
with open(dst, 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
" ${LLM_JP_EVAL_DIR}/eval_configs/all_datasets.yaml ${EVAL_DATASET_CONFIG_PATH}
fi

DUMP_OPTS=(
    --config=${CONFIG_DIR}/${CONFIG_FILE}
    --output_dir=${DATASET_DIR}
    --eval_dataset_config_path=${EVAL_DATASET_CONFIG_PATH}
    --inference_input_dir=${PROMPT_OUTPUT_DIR}
    --max_num_samples=${MAX_NUM_SAMPLES}
)

python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    dump \
    ${DUMP_OPTS[@]}
deactivate

# Inference via the shared vLLM server (no model load in this process).
SERVE_CONFIG=${OUTPUT_DIR}/inference_openai_config.yaml
cat > ${SERVE_CONFIG} <<EOF
server:
  base_url: ${BASE_URL}
  model: ${MODEL_PATH}
  num_concurrent: ${CLIENT_CONCURRENCY}
tokenizer:
  pretrained_model_name_or_path: ${MODEL_PATH}
EOF
if [ "${BASEMODEL}" = true ]; then
    # Offline parity with inference_config_basemodel.yaml of the v2.1.5
    # installer: fix add_special_tokens and temperature explicitly.
    cat >> ${SERVE_CONFIG} <<EOF
tokenize_kwargs:
  add_special_tokens: false
generation_config:
  temperature: 0.0
EOF
fi

INFERENCE_OPTS=(
    --config=${SERVE_CONFIG}
    --output_base_dir=${OFFLINE_OUTPUT_DIR}
    # TODO: Specify the exact prompt_json_path for safety
    --prompt_json_path=${PROMPT_OUTPUT_DIR}_*/*.eval-prompt.json
)
if [ "${APPLY_CHAT_TEMPLATE}" = true ]; then
    INFERENCE_OPTS+=(--apply_chat_template)
fi
if [ -n "${TOKENIZE_KWARGS}" ]; then
    INFERENCE_OPTS+=(--tokenize_kwargs "${TOKENIZE_KWARGS}")
fi

source ${VLLM_VENV}/bin/activate
RUN_NAME=$(python \
    ${SCRIPT_DIR}/inference_openai.py \
    get_run_name \
    "${INFERENCE_OPTS[@]}" | tail -n1)
python \
    ${SCRIPT_DIR}/inference_openai.py \
    inference \
    "${INFERENCE_OPTS[@]}"
deactivate

if [ "${ENABLE_CODE_EXEC}" = true ]; then
    TEMP_DIR=$(mktemp -d)
    SANDBOX_DIR=$TEMP_DIR/dify-sandbox
    LOG_DIR=$TEMP_DIR/dify-sandbox-logs

    mkdir -p $SANDBOX_DIR $LOG_DIR

    # Set an open port. Dify-sandbox internally uses this environment variable.
    export SANDBOX_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    # Used by llm-jp-eval
    export CODE_EXECUTION_ENDPOINT="http://localhost:$SANDBOX_PORT"

    singularity run --bind $SANDBOX_DIR:/var/sandbox,$LOG_DIR:/logs --pwd / docker://langgenius/dify-sandbox@sha256:7ce01bc519069365f22dc0916155608aeff997eeeeda279b784120412c1e71aa &
    SINGULARITY_PID=$!

    cleanup_sandbox() {
        if [ -n "${SINGULARITY_PID:-}" ] && ps -p $SINGULARITY_PID > /dev/null; then
            kill -9 $SINGULARITY_PID 2>/dev/null
        fi
        rm -rf "${TEMP_DIR}"
    }

    trap cleanup_sandbox EXIT

    until curl -s http://localhost:$SANDBOX_PORT/health | grep -q "ok"; do
        echo "Waiting for dify-sandbox to be ready..."
        sleep 3
    done
fi

# TODO: Specify the exact inference_result_dir for safety
INFERENCE_RESULT_DIR=$(find "${OFFLINE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1)
EVAL_OPTS=(
    --config=${CONFIG_DIR}/${CONFIG_FILE}
    # NOTE: OUTPUT_DIRに出力したいが、一部のデータセットはなぜかeval時にdumpを実行する。
    #  その時、データセットの **読み込み先** として `output_dir` が参照されるため、
    #  `output_dir` にはデータセットの保存先 (DATASET_DIR) を指定しなければならない。
    #  それによりevalの結果 (`result.json`) もDATASET_DIRに出力されるので、
    #  eval結果を最後にOUTPUT_DIRに移動する必要がある。
    --output_dir=${DATASET_DIR}
    --eval_dataset_config_path=${EVAL_DATASET_CONFIG_PATH}
    --inference_result_dir=${INFERENCE_RESULT_DIR}
)

source ${LLM_JP_EVAL_DIR}/.venv/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    eval \
    ${EVAL_OPTS[@]}
deactivate

# Move results to OUTPUT_DIR
mkdir -p ${OUTPUT_DIR}/results
cp ${DATASET_DIR}/results/result_${RUN_NAME}.json ${OUTPUT_DIR}/results/result.json
rm ${DATASET_DIR}/results/result_${RUN_NAME}.json

# Update result JSON structure
python3 ${ENV_DIR}/scripts/update_result_json.py ${RESULT_DIR}/result.json

echo "Done"
