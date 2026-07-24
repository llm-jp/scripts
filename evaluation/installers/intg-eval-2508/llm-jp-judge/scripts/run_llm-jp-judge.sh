#!/bin/bash
#
# Run llm-jp-judge: generate responses with the target model on a local vLLM
# server, then score them with a judge model (LLM-as-a-Judge).
#
# Usage:
#   run_llm-jp-judge.sh MODEL_PATH OUTPUT_DIR [options]
#
# Options:
#   --judge-client {openai,azure,bedrock,vllm}
#                             Judge client (default: openai). 'vllm' serves
#                             --judge-model locally with vLLM after generation
#                             and scores through the OpenAI-compatible
#                             endpoint (no external API required).
#   --judge-model NAME        Judge model name (default: gpt-4o-2024-08-06)
#   --judge-base-url URL      Base URL for --judge-client openai (default:
#                             OPENAI_BASE_URL env or https://api.openai.com/v1)
#   --judge-request-interval S  Async request interval in seconds for the
#                             judge (default: 0.5; use a smaller value for a
#                             local judge)
#   --gen-request-interval S  Async request interval in seconds for generation
#                             against the local server (default: 0.1)
#   --tensor-parallel-size N  TP size for local vLLM servers (default: number
#                             of visible GPUs)
#   --gpu-memory-utilization F (default: 0.9)
#   --max-model-len N         --max-model-len for local vLLM servers
#                             (default: model config)
#   --benchmark-size N        Use only the first N samples of each benchmark
#                             (default: all samples)
#   --disable-mt-bench        Skip mt_bench_en / mt_bench_ja
#   --gen-base-url URL        Generate against an already-running
#                             OpenAI-compatible server (e.g. the shared server
#                             of vllm-serve) instead of launching one here.
#                             The server must serve MODEL_PATH as its model id.
#   --generation-only         Stop after the generation phase
#   --judge-only              Skip the generation phase and only run the judge
#                             (OUTPUT_DIR/generation must already exist)
#
# Benchmarks whose dataset was not downloaded at install time (e.g. the gated
# AnswerCarefully datasets) are skipped with a warning.
#
# Judge API credentials are read from environment variables (OPENAI_API_KEY,
# AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY/OPENAI_API_VERSION, AWS_*) or a
# .env file placed in the llm-jp-judge checkout.
#
# Outputs:
#   OUTPUT_DIR/generation/           generated responses (one jsonl per benchmark)
#   OUTPUT_DIR/evaluation/score_table.json  judge scores

set -eux -o pipefail

usage() {
    >&2 echo "Usage: $0 MODEL_PATH OUTPUT_DIR [--judge-client {openai,azure,bedrock,vllm}] [--judge-model NAME] [--judge-base-url URL] [--judge-request-interval S] [--gen-request-interval S] [--tensor-parallel-size N] [--gpu-memory-utilization F] [--max-model-len N] [--benchmark-size N] [--disable-mt-bench] [--gen-base-url URL] [--generation-only] [--judge-only]"
    exit 1
}

# Positional arguments
if [ $# -lt 2 ]; then usage; fi
MODEL_PATH=$1; shift
OUTPUT_DIR=$(realpath $1); shift

# Optional arguments
JUDGE_CLIENT=openai
JUDGE_MODEL=gpt-4o-2024-08-06
JUDGE_BASE_URL=""
JUDGE_REQUEST_INTERVAL=0.5
GEN_REQUEST_INTERVAL=0.1
TP_SIZE=""
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=""
BENCHMARK_SIZE=""
DISABLE_MT_BENCH=false
GEN_BASE_URL=""
GENERATION_ONLY=false
JUDGE_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --judge-client) JUDGE_CLIENT=$2; shift 2 ;;
        --judge-model) JUDGE_MODEL=$2; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL=$2; shift 2 ;;
        --judge-request-interval) JUDGE_REQUEST_INTERVAL=$2; shift 2 ;;
        --gen-request-interval) GEN_REQUEST_INTERVAL=$2; shift 2 ;;
        --tensor-parallel-size) TP_SIZE=$2; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL=$2; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
        --benchmark-size) BENCHMARK_SIZE=$2; shift 2 ;;
        --disable-mt-bench) DISABLE_MT_BENCH=true; shift ;;
        --gen-base-url) GEN_BASE_URL=$2; shift 2 ;;
        --generation-only) GENERATION_ONLY=true; shift ;;
        --judge-only) JUDGE_ONLY=true; shift ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

case ${JUDGE_CLIENT} in
    openai|azure|bedrock|vllm) ;;
    *) >&2 echo "Unknown --judge-client: ${JUDGE_CLIENT}"; usage ;;
esac

if [ "${GENERATION_ONLY}" = true ] && [ "${JUDGE_ONLY}" = true ]; then
    >&2 echo "Error: --generation-only and --judge-only are mutually exclusive."
    exit 1
fi

mkdir -p ${OUTPUT_DIR}/logs

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ENV_DIR=${SCRIPT_DIR}/environment
JUDGE_DIR=${ENV_DIR}/src/llm-jp-judge
VENV=${JUDGE_DIR}/.venv

if [ -z "${TP_SIZE}" ]; then
    TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Relative paths (./data/cache, ./src/llm_jp_judge/data) resolve from the
# repository root.
cd ${JUDGE_DIR}

find_open_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

SERVER_PID=""

# start_vllm_server MODEL PORT LOG_FILE
start_vllm_server() {
    local server_args=(--port $2 --tensor-parallel-size ${TP_SIZE} --gpu-memory-utilization ${GPU_MEM_UTIL})
    if [ -n "${MAX_MODEL_LEN}" ]; then
        server_args+=(--max-model-len ${MAX_MODEL_LEN})
    fi
    ${VENV}/bin/vllm serve "$1" "${server_args[@]}" > "$3" 2>&1 &
    SERVER_PID=$!
}

# wait_vllm_server PORT
wait_vllm_server() {
    local waited=0
    until curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null; do
        if ! ps -p ${SERVER_PID} > /dev/null; then
            >&2 echo "ERROR: vLLM server exited during startup; see the server log."
            exit 1
        fi
        if [ ${waited} -ge 3600 ]; then
            >&2 echo "ERROR: vLLM server did not become ready within 3600s."
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
}

stop_vllm_server() {
    if [ -n "${SERVER_PID}" ] && ps -p ${SERVER_PID} > /dev/null; then
        kill ${SERVER_PID} 2> /dev/null || true
        wait ${SERVER_PID} 2> /dev/null || true
    fi
    SERVER_PID=""
}

trap stop_vllm_server EXIT

# --- Generation: generate responses with the target model ---
# The server is either launched here or, with --gen-base-url, an external
# already-running one (e.g. the shared server of vllm-serve).
if [ "${JUDGE_ONLY}" = false ]; then

if [ -z "${GEN_BASE_URL}" ]; then
    GEN_PORT=$(find_open_port)
    start_vllm_server "${MODEL_PATH}" ${GEN_PORT} ${OUTPUT_DIR}/logs/vllm_serve_target.log
    wait_vllm_server ${GEN_PORT}
    GEN_BASE_URL=http://127.0.0.1:${GEN_PORT}/v1
fi

GEN_ARGS=(
    output.dir=${OUTPUT_DIR}/generation
    client=openai
    client.model_name=${MODEL_PATH}
    client.api_key=local-vllm
    client.base_url=${GEN_BASE_URL}
    client.async_request_interval=${GEN_REQUEST_INTERVAL}
)

# Benchmarks whose dataset.path stays null are skipped by llm-jp-judge.
BENCHMARK_NAMES=(quality_ja safety_ja culture_ja safety_borderline_ja safety_boundary_ja)
BENCHMARK_PATHS=(
    ./data/cache/llm-jp/llm-jp-instructions/v1.0/test.json
    ./data/cache/llm-jp/AnswerCarefully/v2.0/test.json
    ./data/cache/llm-jp/llm-jp-instructions-jculture/v1.0/test.json
    ./data/cache/llm-jp/AnswerCarefully/borderline_v1.0/test.json
    ./data/cache/safety-boundary-test/data/test.csv
)
for i in "${!BENCHMARK_NAMES[@]}"; do
    if [ -f "${BENCHMARK_PATHS[$i]}" ]; then
        GEN_ARGS+=(benchmark.${BENCHMARK_NAMES[$i]}.dataset.path=${BENCHMARK_PATHS[$i]})
    else
        >&2 echo "WARNING: dataset for ${BENCHMARK_NAMES[$i]} not found (${BENCHMARK_PATHS[$i]}); skipping this benchmark."
    fi
done

# mt_bench_en / mt_bench_ja use datasets bundled in the llm-jp-judge repository
# (enabled by default).
if [ "${DISABLE_MT_BENCH}" = true ]; then
    GEN_ARGS+=(benchmark.mt_bench_en.dataset.path=null benchmark.mt_bench_ja.dataset.path=null)
fi

if [ -n "${BENCHMARK_SIZE}" ]; then
    for name in "${BENCHMARK_NAMES[@]}" mt_bench_en mt_bench_ja; do
        GEN_ARGS+=(benchmark.${name}.dataset.size=${BENCHMARK_SIZE})
    done
fi

${VENV}/bin/python -m src.llm_jp_judge.generate "${GEN_ARGS[@]}" \
    > ${OUTPUT_DIR}/logs/generate.log 2> ${OUTPUT_DIR}/logs/generate.err

stop_vllm_server

fi  # JUDGE_ONLY

if [ "${GENERATION_ONLY}" = true ]; then
    trap - EXIT
    echo "Done (generation)"
    exit 0
fi

if [ ! -d "${OUTPUT_DIR}/generation" ]; then
    >&2 echo "ERROR: ${OUTPUT_DIR}/generation does not exist; run the generation phase first."
    exit 1
fi

# --- Judge: score the generated responses ---
EVAL_ARGS=(
    input.dir=${OUTPUT_DIR}/generation
    output.dir=${OUTPUT_DIR}/evaluation
    client.model_name=${JUDGE_MODEL}
    client.async_request_interval=${JUDGE_REQUEST_INTERVAL}
)
case ${JUDGE_CLIENT} in
    vllm)
        JUDGE_PORT=$(find_open_port)
        start_vllm_server "${JUDGE_MODEL}" ${JUDGE_PORT} ${OUTPUT_DIR}/logs/vllm_serve_judge.log
        wait_vllm_server ${JUDGE_PORT}
        EVAL_ARGS+=(
            client=openai
            client.api_key=local-vllm
            client.base_url=http://127.0.0.1:${JUDGE_PORT}/v1
        )
        ;;
    openai)
        EVAL_ARGS+=(client=openai)
        if [ -n "${JUDGE_BASE_URL}" ]; then
            EVAL_ARGS+=(client.base_url=${JUDGE_BASE_URL})
        fi
        ;;
    azure|bedrock)
        EVAL_ARGS+=(client=${JUDGE_CLIENT})
        ;;
esac

${VENV}/bin/python -m src.llm_jp_judge.evaluate "${EVAL_ARGS[@]}" \
    > ${OUTPUT_DIR}/logs/evaluate.log 2> ${OUTPUT_DIR}/logs/evaluate.err

stop_vllm_server
trap - EXIT

echo "Done"
