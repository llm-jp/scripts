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

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 MODEL_PATH OUTPUT_DIR"
    exit 1
fi

# Arguments
MODEL_PATH=$1; shift
OUTPUT_DIR=$(realpath $1); shift
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

DUMP_OPTS=(
    --config=${CONFIG_DIR}/config_base.yaml
    --output_dir=${DATASET_DIR}
    --eval_dataset_config_path=${LLM_JP_EVAL_DIR}/eval_configs/all_datasets.yaml
    --inference_input_dir=${PROMPT_OUTPUT_DIR}
)

source ${LLM_JP_EVAL_DIR}/.venv/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    dump \
    ${DUMP_OPTS[@]}
deactivate

INFERENCE_OPTS=(
    --config=${CONFIG_DIR}/inference_config.yaml
    --output_base_dir=${OFFLINE_OUTPUT_DIR}
    --model.model=${MODEL_PATH}
    --model.tensor_parallel_size=${TP_SIZE}
    --tokenizer.pretrained_model_name_or_path=${MODEL_PATH}
    # TODO: Specify the exact prompt_json_path for safety
    --prompt_json_path=${PROMPT_OUTPUT_DIR}_*/*.eval-prompt.json
)

source ${LLM_JP_EVAL_DIR}/llm-jp-eval-inference/inference-modules/vllm/.venv/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/llm-jp-eval-inference/inference-modules/vllm/inference.py \
    inference \
    ${INFERENCE_OPTS[@]}
deactivate

SANDBOX_DIR=$(mktemp -d)
trap 'rm -rf "${SANDBOX_DIR}"' EXIT

singularity run --bind ${SANDBOX_DIR}:/var/sandbox,./conf:/conf docker://langgenius/dify-sandbox &
SINGULARITY_PID=$!

cleanup_singularity() {
    if ps -p $SINGULARITY_PID > /dev/null; then
        kill -9 $SINGULARITY_PID 2>/dev/null
    fi
}

trap cleanup_singularity EXIT

until curl -s http://localhost:8194/health | grep -q "ok"; do
    echo "Waiting for dify-sandbox to be ready..."
    sleep 3
done

# TODO: Specify the exact inference_result_dir for safety
INFERENCE_RESULT_DIR=$(find "${OFFLINE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1)
EVAL_OPTS=(
    --config=${CONFIG_DIR}/config_base.yaml
    # NOTE: OUTPUT_DIRに出力したいが、一部のデータセットはなぜかeval時にdumpを実行する。
    #  その時、データセットの **読み込み先** として `output_dir` が参照されるため、
    #  `output_dir` にはデータセットの保存先 (DATASET_DIR) を指定しなければならない。
    #  それによりevalの結果 (`result.json`) もDATASET_DIRに出力されるので、
    #  eval結果を最後にOUTPUT_DIRに移動する必要がある。
    --output_dir=${DATASET_DIR}
    --eval_dataset_config_path=${LLM_JP_EVAL_DIR}/eval_configs/all_datasets.yaml
    --inference_result_dir=${INFERENCE_RESULT_DIR}
)

source ${LLM_JP_EVAL_DIR}/.venv/bin/activate
python \
    ${LLM_JP_EVAL_DIR}/scripts/evaluate_llm.py \
    eval \
    ${EVAL_OPTS[@]}
deactivate

# Move results to OUTPUT_DIR
cp -r ${DATASET_DIR}/results ${OUTPUT_DIR}/
rm -r ${DATASET_DIR}/results

# Update result JSON structure
python3 ${ENV_DIR}/scripts/update_result_json.py ${RESULT_DIR}/result.json

echo "Done"
