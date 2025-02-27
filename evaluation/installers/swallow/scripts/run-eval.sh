#!/bin/bash
#SBATCH --job-name=llm-jp-eval
#SBATCH --partition=<FIX_ME>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux

# Open file limit
ulimit -n 65536 1048576

#wandb configs
WANDB_ENTITY=swallow-eval # FIX_ME
WANDB_PROJECT=test # FIX_ME
WANDB_RUN_NAME=$2


ENV_DIR=environment
source ${ENV_DIR}/scripts/environment.sh

# Arguments
MODEL=$1
NUM_PARAMETERS_IN_BILLION=$3
OUTPUT_DIR=${4:-results/${MODEL_NAME_PATH}}

# Convert OUTPUT_DIR to an absolute path
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

echo "OUTPUT_DIR" $OUTPUT_DIR

# Estimate the proportion of GPU MEMORY NEEDED
. scripts/estimate_vllm_memory_usage.sh
GPU_MEM_PROPORTION=$(estimate_vllm_memory $NUM_PARAMETERS_IN_BILLION)
echo "MEMORY PROPORTION" $GPU_MEM_PROPORTION

# fixed vars
EVAL_DIR=${ENV_DIR}/src/swallow-evaluation

HARNESS_SCRIPT_PATH=${EVAL_DIR}/scripts/evaluate_english.sh

source ${ENV_DIR}/venv-harness/bin/activate
pushd ${EVAL_DIR}
bash scripts/evaluate_english-vllm.sh $MODEL $GPU_MEM_PROPORTION $OUTPUT_DIR
popd
deactivate   

source ${ENV_DIR}/venv-postprocessing/bin/activate     
python scripts/upload_to_wandb.py --entity $WANDB_ENTITY --project $WANDB_PROJECT --run $WANDB_RUN_NAME --aggregated_result ${OUTPUT_DIR}/aggregated_result.json
deactivate

echo "Done"
