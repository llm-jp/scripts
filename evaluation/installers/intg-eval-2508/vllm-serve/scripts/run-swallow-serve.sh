#!/bin/bash
#
# Endpoint-based counterpart of swallow's run-eval.sh: runs the same English
# evaluation task groups as evaluate_english-vllm.sh, but through
# `lm_eval --model local-completions` against a pre-launched vLLM server.
# lm_eval starts 6 times as before, but never loads the model.
#
# Usage: run-swallow-serve.sh MODEL OUTPUT_DIR BASE_URL SWALLOW_ENV_DIR [MAX_LENGTH] [CLIENT_CONCURRENCY]
#   MODEL           Model name/path; must equal the --served-model-name of the server
#   OUTPUT_DIR      Output directory (same layout as run-eval.sh)
#   BASE_URL        e.g. http://127.0.0.1:8000/v1
#   SWALLOW_ENV_DIR Installed swallow environment (e.g. .../environment/swallow_v202411-tf5)
#   MAX_LENGTH      Harness client max context length (default: 4096)
#   CLIENT_CONCURRENCY  Prompts kept in flight against the server (default:
#                   256). Translated per task group into the harness's
#                   num_concurrent = ceil(CLIENT_CONCURRENCY / batch_size),
#                   so the server's continuous batching stays saturated.
#
# Score-parity notes vs the offline vllm backend:
# - Prompts are sent as token IDs produced by the same HF tokenizer, and
#   loglikelihoods come from echo=True + logprobs, so the math is unchanged.
# - Generation uses temperature=0 (greedy), as in the offline backend.
# - The engine version is that of the *server* venv; scores are comparable
#   only against runs using the same vLLM version.

set -eux -o pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    >&2 echo "Usage: $0 MODEL OUTPUT_DIR BASE_URL SWALLOW_ENV_DIR [MAX_LENGTH] [CLIENT_CONCURRENCY]"
    exit 1
fi

MODEL=$1
OUTPUT_DIR=$2
BASE_URL=$3
SWALLOW_ENV_DIR=$4
MAX_LENGTH=${5:-4096}
CLIENT_CONCURRENCY=${6:-256}

mkdir -p "$OUTPUT_DIR/results"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

EVAL_DIR=${SWALLOW_ENV_DIR}/environment/src/swallow-evaluation
VENV_DIR=${SWALLOW_ENV_DIR}/environment/venv-harness

# The openai client requires this to be set even for keyless local servers.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL_ARGS="model=${MODEL},base_url=${BASE_URL},tokenizer=${MODEL},tokenizer_backend=huggingface,max_length=${MAX_LENGTH}"

# Task groups below mirror scripts/evaluate_english-vllm.sh in the swallow
# repository; keep them in sync.
GENERAL_LABEL="General"
GENERAL_TASK_NAME="triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en,squadv2"
GENERAL_NUM_FEWSHOT=4
GENERAL_NUM_TESTCASE="all"
GENERAL_BATCH_SIZE=16
GENERAL_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${GENERAL_NUM_FEWSHOT}shot_${GENERAL_NUM_TESTCASE}cases/general"

MMLU_LABEL="MMLU"
MMLU_TASK_NAME="mmlu"
MMLU_NUM_FEWSHOT=5
MMLU_NUM_TESTCASE="all"
MMLU_BATCH_SIZE=16
MMLU_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${MMLU_NUM_FEWSHOT}shot_${MMLU_NUM_TESTCASE}cases/mmlu"

BBH_LABEL="BBH"
BBH_TASK_NAME="bbh_cot_fewshot"
BBH_NUM_FEWSHOT=3
BBH_NUM_TESTCASE="all"
BBH_BATCH_SIZE=16
BBH_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${BBH_NUM_FEWSHOT}shot_${BBH_NUM_TESTCASE}cases/bbh_cot"

GPQA_LABEL="GPQA"
GPQA_NUM_FEWSHOT=0
GPQA_NUM_TESTCASE="all"
GPQA_BATCH_SIZE=16
GPQA_TASK_NAME="gpqa_main_cot_zeroshot_meta_llama3_wo_chat"
GPQA_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${GPQA_NUM_FEWSHOT}shot_${GPQA_NUM_TESTCASE}cases/gpqa_main_cot_zeroshot_meta_llama3_wo_chat"

MATH_LABEL="MATH"
MATH_NUM_FEWSHOT=4
MATH_NUM_TESTCASE="all"
MATH_BATCH_SIZE=16
MATH_TASK_NAME="math_500"
MATH_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${MATH_NUM_FEWSHOT}shot_${MATH_NUM_TESTCASE}cases/${MATH_TASK_NAME}"

mkdir -p $GENERAL_OUTDIR $MMLU_OUTDIR $BBH_OUTDIR $GPQA_OUTDIR $MATH_OUTDIR

LABELS=($GENERAL_LABEL $MMLU_LABEL $BBH_LABEL $GPQA_LABEL $MATH_LABEL)
TASK_NAME=($GENERAL_TASK_NAME $MMLU_TASK_NAME $BBH_TASK_NAME $GPQA_TASK_NAME $MATH_TASK_NAME)
NUM_FEWSHOT=($GENERAL_NUM_FEWSHOT $MMLU_NUM_FEWSHOT $BBH_NUM_FEWSHOT $GPQA_NUM_FEWSHOT $MATH_NUM_FEWSHOT)
BATCH_SIZE=($GENERAL_BATCH_SIZE $MMLU_BATCH_SIZE $BBH_BATCH_SIZE $GPQA_BATCH_SIZE $MATH_BATCH_SIZE)
OUTDIRS=($GENERAL_OUTDIR $MMLU_OUTDIR $BBH_OUTDIR $GPQA_OUTDIR $MATH_OUTDIR)

source "${VENV_DIR}/bin/activate"
pushd "${EVAL_DIR}/lm-evaluation-harness-en"

for i in "${!TASK_NAME[@]}"; do
    echo "Starting evaluation for: ${LABELS[$i]} (via ${BASE_URL})"
    # Translate the framework-independent in-flight prompt target into this
    # group's number of concurrent batches.
    NUM_CONCURRENT=$(( (CLIENT_CONCURRENCY + BATCH_SIZE[i] - 1) / BATCH_SIZE[i] ))
    lm_eval --model local-completions \
        --model_args "${MODEL_ARGS},num_concurrent=${NUM_CONCURRENT}" \
        --tasks ${TASK_NAME[$i]} \
        --num_fewshot ${NUM_FEWSHOT[$i]} \
        --batch_size ${BATCH_SIZE[$i]} \
        --write_out \
        --output_path "${OUTDIRS[$i]}" \
        --use_cache "${OUTDIRS[$i]}" \
        --log_samples \
        --seed 42
done
popd

pushd "${EVAL_DIR}"
python scripts/aggregate_result.py --model "$MODEL" --result-dir "$OUTPUT_DIR"
popd
deactivate

mv "${OUTPUT_DIR}/result.json" "${OUTPUT_DIR}/results/result.json"

echo "All evaluations are done."
