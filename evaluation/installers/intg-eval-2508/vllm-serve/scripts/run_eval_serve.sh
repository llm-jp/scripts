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
#   --tensor-parallel-size N  (default: number of visible GPUs / data-parallel-size)
#   --data-parallel-size N    --data-parallel-size for the server (default: 1)
#   --gpu-memory-utilization F (default: 0.9)
#   --max-model-len N         --max-model-len for the server (default: model config)
#   --port N                  (default: random open port)
#   --swallow                 Run the swallow English evaluation
#   --swallow-env NAME        swallow environment dir name (default: swallow_v202411-tf5)
#   --swallow-max-length N    max_length for the harness client (default: the
#                             server's max_model_len, matching the offline
#                             harness which follows the engine's context size)
#   --client-concurrency N    prompts each evaluation client keeps in flight
#                             against the server (default: 256, on the order
#                             of vLLM's default max_num_seqs). Each client
#                             translates this into its own request shape
#                             (swallow: N / batch_size concurrent batches;
#                             llm-jp-eval: N concurrent single-prompt
#                             requests), so the server-side saturation target
#                             is framework-independent.
#   --llm-jp-eval-versions V... llm-jp-eval versions to run (e.g. v2.1.5; default: none)
#   --max-num-samples N       llm-jp-eval max_num_samples (default: 100)
#   --apply-chat-template     llm-jp-eval: apply chat template
#   --tokenize-kwargs JSON    llm-jp-eval: tokenize_kwargs JSON
#   --legacy-output           llm-jp-eval results under llm-jp-eval_<version>/
#                             instead of llm-jp-eval/<version>/
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

# Installed layout is <experiment_dir>/environment/vllm-serve/run_eval_serve.sh,
# so the experiment dir is two levels up from this script.
EXPERIMENT_DIR="${INTG_EVAL_EXPERIMENT_DIR:-$(realpath "${SCRIPT_DIR}/../..")}"
SERVE_VENV=""
TP=""
DP=1
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=""
PORT=""
RUN_SWALLOW=false
SWALLOW_ENV=swallow_v202411-tf5
SWALLOW_MAX_LENGTH=""
CLIENT_CONCURRENCY=256
LLM_JP_EVAL_VERSIONS=()
MAX_NUM_SAMPLES=100
APPLY_CHAT_TEMPLATE=false
TOKENIZE_KWARGS=""
LEGACY_OUTPUT=false

while [ $# -gt 0 ]; do
    case $1 in
        --experiment-dir) EXPERIMENT_DIR=$2; shift 2 ;;
        --serve-venv) SERVE_VENV=$2; shift 2 ;;
        --tensor-parallel-size) TP=$2; shift 2 ;;
        --data-parallel-size) DP=$2; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL=$2; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --swallow) RUN_SWALLOW=true; shift ;;
        --swallow-env) SWALLOW_ENV=$2; shift 2 ;;
        --swallow-max-length) SWALLOW_MAX_LENGTH=$2; shift 2 ;;
        --client-concurrency) CLIENT_CONCURRENCY=$2; shift 2 ;;
        --llm-jp-eval-versions) shift; while [ $# -gt 0 ] && [[ $1 != --* ]]; do LLM_JP_EVAL_VERSIONS+=("$1"); shift; done ;;
        --max-num-samples) MAX_NUM_SAMPLES=$2; shift 2 ;;
        --apply-chat-template) APPLY_CHAT_TEMPLATE=true; shift ;;
        --tokenize-kwargs) TOKENIZE_KWARGS=$2; shift 2 ;;
        --legacy-output) LEGACY_OUTPUT=true; shift ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

ENV_DIR=${EXPERIMENT_DIR}/environment
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p "$LOG_DIR"

# Auto-detect a venv that can serve the model. Preference order: a dedicated
# serve venv (see README: needed e.g. for vllm 0.11.2, whose server breaks
# with openai>=1.99.2), then the newest llm-jp-eval inference venv
# (vllm 0.19.x), then the transformers-v5 swallow variant, then the original
# swallow harness venv (vllm 0.10.x).
if [ -z "$SERVE_VENV" ]; then
    for cand in \
        "${ENV_DIR}"/vllm-serve/serve-venv-* \
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
    TP=$(( $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) / DP ))
fi
if [ -z "$PORT" ]; then
    PORT=$(find_open_port)
fi
BASE_URL="http://127.0.0.1:${PORT}/v1"

SERVER_ARGS=()
if [ -n "$MAX_MODEL_LEN" ]; then
    SERVER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ "$DP" -gt 1 ]; then
    SERVER_ARGS+=(--data-parallel-size "$DP")
fi

>&2 echo "== serve venv: $SERVE_VENV"
>&2 echo "== model: $MODEL (tp=$TP)"
>&2 echo "== endpoint: $BASE_URL"

start_vllm_server "$SERVE_VENV" "$MODEL" "$PORT" "$TP" "$GPU_MEM_UTIL" \
    "${LOG_DIR}/vllm_serve.log" ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}
trap stop_vllm_server EXIT
wait_vllm_server "$PORT" 3600

# Default the harness client's max_length to the server's actual context
# size, matching the offline harness (which follows the engine's
# max_model_len derived from the model config).
if [ -z "$SWALLOW_MAX_LENGTH" ]; then
    SWALLOW_MAX_LENGTH=$(curl -sf "http://127.0.0.1:${PORT}/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])')
    >&2 echo "== swallow max_length defaulted to server max_model_len: ${SWALLOW_MAX_LENGTH}"
fi

if [ "$RUN_SWALLOW" = true ]; then
    >&2 echo "== running swallow (${SWALLOW_ENV}) against ${BASE_URL}"
    bash "${SCRIPT_DIR}/run-swallow-serve.sh" \
        "$MODEL" \
        "${OUTPUT_DIR}/swallow" \
        "$BASE_URL" \
        "${ENV_DIR}/${SWALLOW_ENV}" \
        "$SWALLOW_MAX_LENGTH" \
        "$CLIENT_CONCURRENCY" \
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
    if [ "$LEGACY_OUTPUT" = true ]; then
        version_output_dir=${OUTPUT_DIR}/llm-jp-eval_${version}
    else
        version_output_dir=${OUTPUT_DIR}/llm-jp-eval/${version}
    fi
    LLM_JP_EVAL_OPTS+=(--client-concurrency "$CLIENT_CONCURRENCY")
    mkdir -p "$version_output_dir"
    bash "${SCRIPT_DIR}/run_llm-jp-eval-serve.sh" \
        "$MODEL" \
        "$version_output_dir" \
        "$BASE_URL" \
        "${ENV_DIR}/llm-jp-eval-${version}" \
        "${LLM_JP_EVAL_OPTS[@]}" \
        > "${LOG_DIR}/llm-jp-eval-${version}.log" 2> "${LOG_DIR}/llm-jp-eval-${version}.err"
done

stop_vllm_server
trap - EXIT
echo "Done"
