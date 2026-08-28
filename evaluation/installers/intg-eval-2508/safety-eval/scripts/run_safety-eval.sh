#!/bin/bash
#
# Run the safety evaluation (LLM_Safety_Eva): generate responses with the
# target model on local vLLM (offline inference), score them with local
# evaluators (JBBQ, JTruthfulQA) and an Azure OpenAI judge (AnswerCarefully,
# JSocialFact, safety_boundary), then aggregate the scores.
#
# Usage:
#   run_safety-eval.sh MODEL_PATH OUTPUT_DIR [options]
#
# Options:
#   --benchmarks LIST         Comma-separated benchmarks to run (default: all).
#                             Available: jbbq_age, jtruthfulqa,
#                             answer_carefully_test, JSocialFact-01-test,
#                             safety_boundary
#   --tensor-parallel-size N  TP size for vLLM generation (default: number of
#                             visible GPUs)
#   --ask-times N             Generations per prompt (default: 3)
#   --batch-size N            Prompts per llm.generate() call (default: 64)
#   --max-tokens N            Max generated tokens (default: 4096)
#   --temperature F           Sampling temperature (default: 1.0)
#   --top-p F                 Nucleus sampling top_p (default: 0.95)
#   --judge-model NAME        Judge model for the judge-scored benchmarks
#                             (Azure: deployment name, default
#                             gpt-4o-2024-11-20; OpenAI-compatible: model
#                             name served by the endpoint)
#   --judge-max-tokens N      max_tokens for each judge request (default:
#                             512). Thinking judge models spend the budget on
#                             reasoning first, so they need more (e.g. 2048)
#                             to reach the final "評価：[[N]]" verdict.
#   --benchmark-size N        Use only the first N samples of each benchmark
#                             (default: all samples). Mainly for smoke tests;
#                             note that existing generations under
#                             model_output/ are reused as-is.
#   --generation-only         Stop after the generation phase
#   --eval-only               Skip generation and only run evaluation +
#                             aggregation (OUTPUT_DIR/model_output must exist)
#
# The judge-scored benchmarks (answer_carefully_test, JSocialFact-01-test,
# safety_boundary) call an external judge API. Either of:
#   Azure OpenAI:      AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
#                      (+ AZURE_OPENAI_DEPLOYMENT_NAME, default gpt-4o-2024-11-20)
#   OpenAI-compatible: OPENAI_API_KEY + OPENAI_BASE_URL (+ --judge-model)
# Azure takes precedence when both are set. jbbq_age and jtruthfulqa are
# scored locally without an external API.
#
# The vendored evaluation code (LLM_Safety_Eva, received from the safety WG)
# is used as-is except for the OpenAI-compatible judge endpoint support in
# evaluators/llm_as_a_judge_chatgpt.py. Note its hardcoded generation
# settings: gpu_memory_utilization=0.85 and max_model_len=4096
# (LLM_Safety_Eva/run_one_vllm_model.py).
#
# Outputs (under OUTPUT_DIR):
#   model_output/<model>_output/<benchmark>.json           raw generations
#   evaluator_output/<model>_evaluate/<benchmark>_base_evaluated_<date>.json
#                                                          per-sample scores
#   evaluate_count/<model>_evaluate/...                    aggregated scores
#   work/                                                  working copy of the
#                                                          code + config.yaml
#                                                          used for each phase

set -eux -o pipefail

usage() {
    >&2 echo "Usage: $0 MODEL_PATH OUTPUT_DIR [--benchmarks LIST] [--tensor-parallel-size N] [--ask-times N] [--batch-size N] [--max-tokens N] [--temperature F] [--top-p F] [--judge-model NAME] [--judge-max-tokens N] [--benchmark-size N] [--generation-only] [--eval-only]"
    exit 1
}

ALL_BENCHMARKS="jbbq_age,jtruthfulqa,answer_carefully_test,JSocialFact-01-test,safety_boundary"
# Benchmark -> evaluator/prompt combinations recommended by the safety WG
# (LLM_Safety_Eva/README.md). The judge benchmarks use different prompt
# templates, so evaluate.py runs once per combination.
JUDGE_V1_BENCHMARKS="answer_carefully_test JSocialFact-01-test"
JUDGE_SB_BENCHMARKS="safety_boundary"

# Positional arguments
if [ $# -lt 2 ]; then usage; fi
MODEL_PATH=$1; shift
OUTPUT_DIR=$(realpath $1); shift

# Optional arguments
BENCHMARKS=${ALL_BENCHMARKS}
TP_SIZE=""
ASK_TIMES=3
BATCH_SIZE=64
MAX_TOKENS=4096
TEMPERATURE=1.0
TOP_P=0.95
JUDGE_MODEL=""
BENCHMARK_SIZE=""
GENERATION_ONLY=false
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmarks) BENCHMARKS=$2; shift 2 ;;
        --tensor-parallel-size) TP_SIZE=$2; shift 2 ;;
        --ask-times) ASK_TIMES=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        --max-tokens) MAX_TOKENS=$2; shift 2 ;;
        --temperature) TEMPERATURE=$2; shift 2 ;;
        --top-p) TOP_P=$2; shift 2 ;;
        --judge-model) JUDGE_MODEL=$2; shift 2 ;;
        --judge-max-tokens) export SAFETY_EVAL_JUDGE_MAX_TOKENS=$2; shift 2 ;;
        --benchmark-size) BENCHMARK_SIZE=$2; shift 2 ;;
        --generation-only) GENERATION_ONLY=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        *) >&2 echo "Unknown option: $1"; usage ;;
    esac
done

if [ "${GENERATION_ONLY}" = true ] && [ "${EVAL_ONLY}" = true ]; then
    >&2 echo "Error: --generation-only and --eval-only are mutually exclusive."
    exit 1
fi

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
CODE_DIR=${SCRIPT_DIR}/LLM_Safety_Eva
ENV_DIR=${SCRIPT_DIR}/environment
VENV=${ENV_DIR}/venv

if [ -z "${TP_SIZE}" ]; then
    TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Validate the benchmark selection and drop benchmarks whose data file was
# not staged at install time (with a warning), same policy as llm-jp-judge.
SELECTED=()
for b in ${BENCHMARKS//,/ }; do
    case ",${ALL_BENCHMARKS}," in
        *",${b},"*) ;;
        *) >&2 echo "Error: unknown benchmark '${b}' (available: ${ALL_BENCHMARKS})"; exit 1 ;;
    esac
    if [ ! -f "${CODE_DIR}/benchmark_data/${b}.json" ]; then
        >&2 echo "WARNING: benchmark data ${CODE_DIR}/benchmark_data/${b}.json not found; skipping ${b}."
        continue
    fi
    SELECTED+=("${b}")
done
if [ ${#SELECTED[@]} -eq 0 ]; then
    >&2 echo "Error: no runnable benchmark (check the benchmark data staged at install time)."
    exit 1
fi

# intersect_benchmarks GROUP -> space-separated selected benchmarks in GROUP
intersect_benchmarks() {
    local out=()
    for b in $1; do
        for s in "${SELECTED[@]}"; do
            if [ "${b}" = "${s}" ]; then out+=("${b}"); fi
        done
    done
    echo "${out[@]:-}"
}

JUDGE_V1_SELECTED=$(intersect_benchmarks "${JUDGE_V1_BENCHMARKS}")
JUDGE_SB_SELECTED=$(intersect_benchmarks "${JUDGE_SB_BENCHMARKS}")
JBBQ_SELECTED=$(intersect_benchmarks "jbbq_age")
JTQA_SELECTED=$(intersect_benchmarks "jtruthfulqa")

# Judge benchmarks need judge API credentials (Azure OpenAI, or an
# OpenAI-compatible endpoint; Azure takes precedence); fail before loading
# the model rather than after hours of generation.
if [ "${GENERATION_ONLY}" = false ] && [ -n "${JUDGE_V1_SELECTED}${JUDGE_SB_SELECTED}" ]; then
    if [ -n "${AZURE_OPENAI_API_KEY:-}" ] && [ -n "${AZURE_OPENAI_ENDPOINT:-}" ]; then
        # The judge client concatenates the endpoint without a separator;
        # make sure it ends with a slash.
        export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT%/}/"
        if [ -n "${JUDGE_MODEL}" ]; then
            export AZURE_OPENAI_DEPLOYMENT_NAME=${JUDGE_MODEL}
        fi
    elif [ -n "${OPENAI_API_KEY:-}" ] && [ -n "${OPENAI_BASE_URL:-}" ]; then
        if [ -n "${JUDGE_MODEL}" ]; then
            export SAFETY_EVAL_JUDGE_MODEL=${JUDGE_MODEL}
        fi
    else
        >&2 echo "Error: judge API credentials are required for the judge-scored benchmarks (${JUDGE_V1_SELECTED} ${JUDGE_SB_SELECTED})."
        >&2 echo "Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (Azure) or OPENAI_API_KEY + OPENAI_BASE_URL (OpenAI-compatible),"
        >&2 echo "or restrict --benchmarks to jbbq_age/jtruthfulqa."
        exit 1
    fi
fi

mkdir -p ${OUTPUT_DIR}/logs
LOG_DIR=${OUTPUT_DIR}/logs

# --- Working directory -------------------------------------------------
# The vendored code resolves config.yaml, benchmark_data/ and its output
# directories relative to the CWD. Run it from a working copy under
# OUTPUT_DIR so nothing is written into the shared install directory;
# outputs are redirected into OUTPUT_DIR via symlinks.
WORK_DIR=${OUTPUT_DIR}/work
mkdir -p ${WORK_DIR}
cp ${CODE_DIR}/*.py ${WORK_DIR}/
for d in eva_prompt evaluators model utils; do
    mkdir -p ${WORK_DIR}/$d
    cp ${CODE_DIR}/$d/* ${WORK_DIR}/$d/
done
# Benchmark data: the full staged files, or head-truncated copies with
# --benchmark-size (rm first: an earlier run may have left either variant).
rm -rf ${WORK_DIR}/benchmark_data
if [ -n "${BENCHMARK_SIZE}" ]; then
    mkdir -p ${WORK_DIR}/benchmark_data
    for b in "${SELECTED[@]}"; do
        ${VENV}/bin/python - "${CODE_DIR}/benchmark_data/${b}.json" "${WORK_DIR}/benchmark_data/${b}.json" "${BENCHMARK_SIZE}" <<'PYEOF'
import json
import sys

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, encoding="utf-8") as f:
    items = json.load(f)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(items[:n], f, ensure_ascii=False, indent=2)
PYEOF
    done
else
    ln -sfn ${CODE_DIR}/benchmark_data ${WORK_DIR}/benchmark_data
fi
for d in model_output evaluator_output evaluate_count; do
    mkdir -p ${OUTPUT_DIR}/$d
    ln -sfn ${OUTPUT_DIR}/$d ${WORK_DIR}/$d
done

# write_config "BENCH1 BENCH2" "EVALUATOR" "PROMPT" NAME
# Writes WORK_DIR/config.yaml for one phase and keeps a copy as
# config_NAME.yaml for provenance.
write_config() {
    local benchmarks=$1 evaluator=$2 prompt=$3 name=$4
    {
        echo "models:"
        echo "  - name: \"${MODEL_PATH}\""
        echo "benchmark_data:"
        for b in ${benchmarks}; do
            echo "  - ${b}"
        done
        echo "evaluation_items:"
        if [ -n "${evaluator}" ]; then
            echo "  - ${evaluator}"
        else
            echo "  []"
        fi
        echo "eva_prompt:"
        echo "  - ${prompt}"
        echo "ask_times: ${ASK_TIMES}"
        echo "tensor_parallel_size: ${TP_SIZE}"
        echo "batch_size: ${BATCH_SIZE}"
        echo "max_tokens: ${MAX_TOKENS}"
        echo "temperature: ${TEMPERATURE}"
        echo "top_p: ${TOP_P}"
    } > ${WORK_DIR}/config.yaml
    cp ${WORK_DIR}/config.yaml ${WORK_DIR}/config_${name}.yaml
}

cd ${WORK_DIR}

# --- Generation: answer every selected benchmark with the target model ---
# run_one_vllm_model.py is called directly (one model per job) instead of
# through run_vllm_v2.py, whose pkill-based cleanup could kill unrelated
# processes of the same user on shared nodes. Existing outputs under
# model_output/ are skipped, so interrupted runs can be resumed.
if [ "${EVAL_ONLY}" = false ]; then
    write_config "${SELECTED[*]}" "" "V1" generation
    ${VENV}/bin/python run_one_vllm_model.py --model "${MODEL_PATH}" \
        > ${LOG_DIR}/generation.log 2> ${LOG_DIR}/generation.err
fi

if [ "${GENERATION_ONLY}" = true ]; then
    echo "Done (generation)"
    exit 0
fi

# --- Evaluation: one evaluate.py pass per evaluator/prompt combination ---
if [ -n "${JUDGE_V1_SELECTED}" ]; then
    write_config "${JUDGE_V1_SELECTED}" llm_as_a_judge_chatgpt V1 judge_v1
    ${VENV}/bin/python evaluate.py \
        > ${LOG_DIR}/evaluate_judge_v1.log 2> ${LOG_DIR}/evaluate_judge_v1.err
fi

if [ -n "${JUDGE_SB_SELECTED}" ]; then
    write_config "${JUDGE_SB_SELECTED}" llm_as_a_judge_chatgpt safety_boundary_1_0_1 judge_safety_boundary
    ${VENV}/bin/python evaluate.py \
        > ${LOG_DIR}/evaluate_judge_safety_boundary.log 2> ${LOG_DIR}/evaluate_judge_safety_boundary.err
fi

if [ -n "${JBBQ_SELECTED}" ]; then
    write_config "${JBBQ_SELECTED}" jbbq V1 jbbq
    ${VENV}/bin/python evaluate.py \
        > ${LOG_DIR}/evaluate_jbbq.log 2> ${LOG_DIR}/evaluate_jbbq.err
fi

if [ -n "${JTQA_SELECTED}" ]; then
    write_config "${JTQA_SELECTED}" jtruthfulqa V1 jtruthfulqa
    # The classifier (nlp-waseda/roberta_jtruthfulqa) was prefetched into the
    # environment-local HF cache at install time; read it offline so the
    # evaluation phase never downloads on compute nodes. Its tokenizer shells
    # out to the jumanpp binary built into the environment at install time.
    JTQA_ENV=()
    if [ -d "${ENV_DIR}/data/hf" ]; then
        JTQA_ENV+=(HF_HOME=${ENV_DIR}/data/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
    fi
    if [ -x "${ENV_DIR}/jumanpp/bin/jumanpp" ]; then
        JTQA_ENV+=("PATH=${ENV_DIR}/jumanpp/bin:${PATH}")
    fi
    env ${JTQA_ENV[@]+"${JTQA_ENV[@]}"} ${VENV}/bin/python evaluate.py \
        > ${LOG_DIR}/evaluate_jtruthfulqa.log 2> ${LOG_DIR}/evaluate_jtruthfulqa.err
fi

# --- Aggregation: walk evaluator_output/ and write final scores ---
${VENV}/bin/python evaluate_count.py \
    > ${LOG_DIR}/evaluate_count.log 2> ${LOG_DIR}/evaluate_count.err

echo "Done. Aggregated scores: ${OUTPUT_DIR}/evaluate_count/"
