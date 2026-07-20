#!/bin/bash
#
# Run swallow and/or llm-jp-eval evaluations against a single shared vLLM
# server, so that the model is loaded exactly once per job instead of once
# per lm_eval invocation (swallow launches lm_eval 6 times) and once per
# llm-jp-eval version. This matters most for large models where each load
# takes tens of minutes.
#
# Usage:
#   bash run_eval_serve.sh MODEL OUTPUT_DIR [options]
#
# Options:
#   --experiment-dir DIR      Parent of environment/ (default: $INTG_EVAL_EXPERIMENT_DIR or the directory two levels up from this script)
#   --serve-venv DIR          venv that provides `vllm serve` (default: auto-detect, see below)
#   --tensor-parallel-size N  (default: number of visible GPUs)
#   --gpu-memory-utilization F (default: 0.9)
#   --max-model-len N         --max-model-len for the server (default: model config)
#   --port N                  (default: random open port)
#   --swallow                 Run the swallow English evaluation
#   --swallow-env NAME        swallow environment dir name (default: swallow_v202411-tf5)
#   --swallow-max-length N    max_length for the harness client (default: 4096)
#   --llm-jp-eval-versions V... llm-jp-eval versions to run (e.g. v2.1.5; default: none)
#   --max-num-samples N       llm-jp-eval max_num_samples (default: 100)
#   --apply-chat-template     llm-jp-eval: apply chat template
#   --tokenize-kwargs JSON    llm-jp-eval: tokenize_kwargs JSON
#
# The server venv, swallow environment and llm-jp-eval environments must be
# installed beforehand (see ../README.md). Existing environments are only
# read, never modified.

set -eu -o pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source "${SCRIPT_DIR}/serve_common.sh"

usage() { sed -n '3,30p' "${BASH_SOURCE[0]}" >&2; exit 1; }

if [ $# -lt 2 ]; then usage; fi
MODEL=$1; shift
OUTPUT_DIR=$(realpath "$1"); shift

EXPERIMENT_DIR="${INTG_EVAL_EXPERIMENT_DIR:-$(realpath "${SCRIPT_DIR}/../../..")}"
SERVE_VENV=""
TP=""
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=""
PORT=""
RUN_SWALLOW=false
SWALLOW_ENV=swallow_v202411-tf5
SWALLOW_MAX_LENGTH=4096
LLM_JP_EVAL_VERSIONS=()
MAX_NUM_SAMPLES=100
APPLY_CHAT_TEMPLATE=false
TOKENIZE_KWARGS=""

while [ $# -gt 0 ]; do
    case $1 in
        --experiment-dir) EXPERIMENT_DIR=$2; shift 2 ;;
        --serve-venv) SERVE_VENV=$2; shift 2 ;;
        --tensor-parallel-size) TP=$2; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL=$2; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --swallow) RUN_SWALLOW=true; shift ;;
        --swallow-env) SWALLOW_ENV=$2; shift 2 ;;
        --swallow-max-length) SWALLOW_MAX_LENGTH=$2; shift 2 ;;
        --llm-jp-eval-versions) shift; while [ $# -gt 0 ] && [[ $1 != --* ]]; do LLM_JP_EVAL_VERSIONS+=("$1"); shift; done ;;
        --max-num-samples) MAX_NUM_SAMPLES=$2; shift 2 ;;
        --apply-chat-template) APPLY_CHAT_TEMPLATE=true; shift ;;
        --tokenize-kwargs) TOKENIZE_KWARGS=$2; shift 2 ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

ENV_DIR=${EXPERIMENT_DIR}/environment
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p "$LOG_DIR"

# Auto-detect a venv that can serve the model. Preference order: the newest
# llm-jp-eval inference venv (vllm 0.19.x), then the transformers-v5 swallow
# variant, then the original swallow harness venv (vllm 0.10.x).
if [ -z "$SERVE_VENV" ]; then
    for cand in \
        "${ENV_DIR}/llm-jp-eval-v2.1.5/environment/src/llm-jp-eval/llm-jp-eval-inference/inference-modules/vllm/.venv" \
        "${ENV_DIR}/swallow_v202411-tf5/environment/venv-harness" \
        "${ENV_DIR}/swallow_v202411/environment/venv-harness"; do
        if [ -x "${cand}/bin/vllm" ]; then SERVE_VENV=$cand; break; fi
    done
fi
if [ -z "$SERVE_VENV" ] || [ ! -x "${SERVE_VENV}/bin/vllm" ]; then
    >&2 echo "ERROR: no venv providing 'vllm serve' found; pass --serve-venv."
    exit 1
fi

if [ -z "$TP" ]; then
    TP=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
if [ -z "$PORT" ]; then
    PORT=$(find_open_port)
fi
BASE_URL="http://127.0.0.1:${PORT}/v1"

SERVER_ARGS=()
if [ -n "$MAX_MODEL_LEN" ]; then
    SERVER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

>&2 echo "== serve venv: $SERVE_VENV"
>&2 echo "== model: $MODEL (tp=$TP)"
>&2 echo "== endpoint: $BASE_URL"

start_vllm_server "$SERVE_VENV" "$MODEL" "$PORT" "$TP" "$GPU_MEM_UTIL" \
    "${LOG_DIR}/vllm_serve.log" ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}
trap stop_vllm_server EXIT
wait_vllm_server "$PORT" 3600

if [ "$RUN_SWALLOW" = true ]; then
    >&2 echo "== running swallow (${SWALLOW_ENV}) against ${BASE_URL}"
    bash "${SCRIPT_DIR}/run-swallow-serve.sh" \
        "$MODEL" \
        "${OUTPUT_DIR}/swallow" \
        "$BASE_URL" \
        "${ENV_DIR}/${SWALLOW_ENV}" \
        "$SWALLOW_MAX_LENGTH" \
        > "${LOG_DIR}/swallow_eval.log" 2> "${LOG_DIR}/swallow_eval.err"
fi

for version in ${LLM_JP_EVAL_VERSIONS[@]+"${LLM_JP_EVAL_VERSIONS[@]}"}; do
    >&2 echo "== running llm-jp-eval ${version} against ${BASE_URL}"
    LLM_JP_EVAL_OPTS=(--max_num_samples "$MAX_NUM_SAMPLES")
    if [ "$APPLY_CHAT_TEMPLATE" = true ]; then
        LLM_JP_EVAL_OPTS+=(--apply_chat_template)
    fi
    if [ -n "$TOKENIZE_KWARGS" ]; then
        LLM_JP_EVAL_OPTS+=(--tokenize_kwargs "$TOKENIZE_KWARGS")
    fi
    mkdir -p "${OUTPUT_DIR}/llm-jp-eval/${version}"
    bash "${SCRIPT_DIR}/run_llm-jp-eval-serve.sh" \
        "$MODEL" \
        "${OUTPUT_DIR}/llm-jp-eval/${version}" \
        "$BASE_URL" \
        "${ENV_DIR}/llm-jp-eval-${version}" \
        "${LLM_JP_EVAL_OPTS[@]}" \
        > "${LOG_DIR}/llm-jp-eval-${version}.log" 2> "${LOG_DIR}/llm-jp-eval-${version}.err"
done

stop_vllm_server
trap - EXIT
echo "Done"
