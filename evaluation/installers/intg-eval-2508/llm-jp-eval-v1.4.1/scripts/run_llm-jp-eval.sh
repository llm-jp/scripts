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

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    >&2 echo "Usage: $0 MODEL_PATH OUTPUT_DIR [MAX_NUM_SAMPLES]"
    exit 1
fi

# Arguments
MODEL_PATH=$1; shift
OUTPUT_DIR=$1; shift
MAX_NUM_SAMPLES=${1:-100}
TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE-1)))

mkdir -p ${OUTPUT_DIR}

ENV_DIR=$(pwd)/environment
source ${ENV_DIR}/scripts/environment.sh

CONFIG_DIR=$(pwd)/resources
PROMPT_OUTPUT_DIR=${OUTPUT_DIR}/prompts
OFFLINE_OUTPUT_DIR=${OUTPUT_DIR}/offline
RESULT_DIR=${OUTPUT_DIR}/results
LLM_JP_EVAL_DIR=${ENV_DIR}/src/llm-jp-eval
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

if [ -n "${HF_HOME}" ]; then
    LLM_JP_EVAL_OVERRIDES+=(
        resource_dir=${HF_HOME}
    )
fi

echo "LLM_JP_EVAL_OVERRIDES:"
for x in ${LLM_JP_EVAL_OVERRIDES[@]}; do
    echo "  $x"
done

OFFLINE_INFERENCE_VLLM_OVERRIDES=(
    "offline_inference.prompt_json_path=[\"${PROMPT_OUTPUT_DIR}/*.eval-prompt.json\"]"
    offline_inference.exact_output_dir=${OFFLINE_OUTPUT_DIR}
    model.model=${MODEL_PATH}
    tokenizer.pretrained_model_name_or_path=${MODEL_PATH}
    model.tensor_parallel_size=${TP_SIZE}
)
echo "OFFLINE_INFERENCE_VLLM_OVERRIDES:"
for x in ${OFFLINE_INFERENCE_VLLM_OVERRIDES[@]}; do
    echo "  $x"
done

source ${ENV_DIR}/venv-eval/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/scripts/dump_prompts.py \
    -cp ${CONFIG_DIR} \
    -cn config_base \
    hydra.run.dir=${PROMPT_OUTPUT_DIR}/dump_prompts \
    ${LLM_JP_EVAL_OVERRIDES[@]}
deactivate

source ${ENV_DIR}/venv-vllm/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/offline_inference/vllm/offline_inference_vllm.py \
    -cp ${CONFIG_DIR} \
    -cn config_offline_inference_vllm \
    hydra.run.dir=${OFFLINE_OUTPUT_DIR}/offline_inference_vllm \
    ${OFFLINE_INFERENCE_VLLM_OVERRIDES[@]}
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
