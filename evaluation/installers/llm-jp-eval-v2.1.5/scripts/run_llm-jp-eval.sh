#!/bin/bash
#SBATCH --job-name=0060_eval
#SBATCH --partition=<FIX_ME>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=200G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

usage() {
    >&2 echo "Usage: $0 MODEL_PATH OUTPUT_DIR [--max_num_samples N] [--apply_chat_template] [--reasoning_parser PARSER] [--tokenize_kwargs JSON] [--basemodel] [--max_tokens N] [--reasoning_content_length N] [--max_model_len N]"
    exit 1
}

# Positional arguments
if [ $# -lt 2 ]; then usage; fi
MODEL_PATH=$1; shift
OUTPUT_DIR=$(realpath $1); shift

# Optional arguments
MAX_NUM_SAMPLES=100
APPLY_CHAT_TEMPLATE=false
REASONING_PARSER=""
TOKENIZE_KWARGS=""
BASEMODEL=false
MAX_TOKENS=""
REASONING_CONTENT_LENGTH=""
MAX_MODEL_LEN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_num_samples) MAX_NUM_SAMPLES=$2; shift 2 ;;
        --apply_chat_template) APPLY_CHAT_TEMPLATE=true; shift ;;
        --reasoning_parser) REASONING_PARSER=$2; shift 2 ;;
        --tokenize_kwargs) TOKENIZE_KWARGS=$2; shift 2 ;;
        --basemodel) BASEMODEL=true; shift ;;
        # Global cap on generated tokens. Default (unset) keeps the upstream
        # behavior: the per-dataset output_length decides max_tokens.
        --max_tokens) MAX_TOKENS=$2; shift 2 ;;
        # Extra token budget added to each dataset's output_length for the
        # reasoning content of thinking models. Only meaningful together with
        # --reasoning_parser (upstream ignores it otherwise).
        --reasoning_content_length) REASONING_CONTENT_LENGTH=$2; shift 2 ;;
        # Context length cap for inference. Default (unset) keeps the value
        # in resources/inference_config*.yaml (4096).
        --max_model_len) MAX_MODEL_LEN=$2; shift 2 ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

if [ -n "${REASONING_CONTENT_LENGTH}" ] && [ -z "${REASONING_PARSER}" ]; then
    >&2 echo "Error: --reasoning_content_length requires --reasoning_parser."
    exit 1
fi

# --basemodel fixes the prompt template (config_basemodel.yaml) and the
# tokenization/sampling parameters (inference_config_basemodel.yaml:
# add_special_tokens=False, temperature=0.0), and evaluates on the 4-shot
# datasets only (eval_configs/only_4shots.yaml).
if [ "${BASEMODEL}" = true ] && { [ "${APPLY_CHAT_TEMPLATE}" = true ] || [ -n "${TOKENIZE_KWARGS}" ]; }; then
    >&2 echo "Error: --basemodel cannot be combined with --apply_chat_template / --tokenize_kwargs."
    exit 1
fi
TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE-1)))

# TODO: Must specify an empty directory
mkdir -p ${OUTPUT_DIR}

ENV_DIR=$(pwd)/environment
source ${ENV_DIR}/scripts/environment.sh

CONFIG_DIR=$(pwd)/resources
PROMPT_OUTPUT_DIR=${OUTPUT_DIR}/prompts
OFFLINE_OUTPUT_DIR=${OUTPUT_DIR}/offline
RESULT_DIR=${OUTPUT_DIR}/results
LLM_JP_EVAL_DIR=${ENV_DIR}/src/llm-jp-eval
DATASET_DIR=${ENV_DIR}/data/llm-jp-eval

# evaluate_llm.py derives three paths from a single --output_dir: the dataset
# READ root (output_dir/datasets/<ver>/...), the metric cache (output_dir/cache,
# where the COMET checkpoint lives), and the results WRITE root
# (output_dir/results). We want results under the user's OUTPUT_DIR while still
# reading the datasets and COMET checkpoint that were installed (once) into the
# shared DATASET_DIR. So point --output_dir at OUTPUT_DIR and symlink the two
# read-only inputs back to the shared install. This removes any write into the
# shared installation directory (previously a cross-user permission hazard).
ln -sfnT ${DATASET_DIR}/datasets ${OUTPUT_DIR}/datasets
ln -sfnT ${DATASET_DIR}/cache    ${OUTPUT_DIR}/cache
# Shared caches prefetched by install.sh; read-only at eval time.
EVAL_ENV=(
    HF_HOME=${DATASET_DIR}/hf
    NLTK_DATA=${DATASET_DIR}/nltk
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
)

CONFIG_FILE=config_base.yaml
INFERENCE_CONFIG_FILE=inference_config.yaml
if [ "${BASEMODEL}" = true ]; then
    CONFIG_FILE=config_basemodel.yaml
    INFERENCE_CONFIG_FILE=inference_config_basemodel.yaml
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
    --output_dir=${OUTPUT_DIR}
    --eval_dataset_config_path=${EVAL_DATASET_CONFIG_PATH}
    --inference_input_dir=${PROMPT_OUTPUT_DIR}
    --max_num_samples=${MAX_NUM_SAMPLES}
)

python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    dump \
    ${DUMP_OPTS[@]}
deactivate

INFERENCE_OPTS=(
    --config=${CONFIG_DIR}/${INFERENCE_CONFIG_FILE}
    --output_base_dir=${OFFLINE_OUTPUT_DIR}
    --model.model=${MODEL_PATH}
    --model.tensor_parallel_size=${TP_SIZE}
    --tokenizer.pretrained_model_name_or_path=${MODEL_PATH}
    # TODO: Specify the exact prompt_json_path for safety
    --prompt_json_path=${PROMPT_OUTPUT_DIR}_*/*.eval-prompt.json
)
if [ "${APPLY_CHAT_TEMPLATE}" = true ]; then
    INFERENCE_OPTS+=(--apply_chat_template)
fi
if [ -n "${REASONING_PARSER}" ]; then
    INFERENCE_OPTS+=(--model.reasoning_parser ${REASONING_PARSER})
fi
if [ -n "${TOKENIZE_KWARGS}" ]; then
    INFERENCE_OPTS+=(--tokenize_kwargs "${TOKENIZE_KWARGS}")
fi
if [ -n "${MAX_TOKENS}" ]; then
    INFERENCE_OPTS+=(--generation_config.max_tokens ${MAX_TOKENS})
fi
if [ -n "${REASONING_CONTENT_LENGTH}" ]; then
    INFERENCE_OPTS+=(--reasoning_content_length ${REASONING_CONTENT_LENGTH})
fi
if [ -n "${MAX_MODEL_LEN}" ]; then
    INFERENCE_OPTS+=(--model.max_model_len ${MAX_MODEL_LEN})
fi

source ${LLM_JP_EVAL_DIR}/llm-jp-eval-inference/inference-modules/vllm/.venv/bin/activate
RUN_NAME=$(python \
    ${LLM_JP_EVAL_DIR}/llm-jp-eval-inference/inference-modules/vllm/inference.py \
    get_run_name \
    "${INFERENCE_OPTS[@]}" | tail -n1)
python \
    ${LLM_JP_EVAL_DIR}/llm-jp-eval-inference/inference-modules/vllm/inference.py \
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
    # output_dir/datasets and output_dir/cache are symlinks to the shared
    # install (see above), so eval reads the datasets and COMET checkpoint from
    # there while writing result_${RUN_NAME}.json into ${OUTPUT_DIR}/results.
    --output_dir=${OUTPUT_DIR}
    --eval_dataset_config_path=${EVAL_DATASET_CONFIG_PATH}
    --inference_result_dir=${INFERENCE_RESULT_DIR}
)

source ${LLM_JP_EVAL_DIR}/.venv/bin/activate
env "${EVAL_ENV[@]}" python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    eval \
    ${EVAL_OPTS[@]}
deactivate

# Normalize the result filename (result_${RUN_NAME}.json -> result.json).
mv ${RESULT_DIR}/result_${RUN_NAME}.json ${RESULT_DIR}/result.json

# Update result JSON structure
python3 ${ENV_DIR}/scripts/update_result_json.py ${RESULT_DIR}/result.json

echo "Done"
