#!/bin/bash
#SBATCH --job-name=swallow
#SBATCH --partition=<FIX_ME>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

ENV_DIR=environment
source ${ENV_DIR}/scripts/environment.sh

# Arguments
MODEL=$1
OUTPUT_DIR=${2:-results/${MODEL}}
NUM_PARAMETERS_IN_BILLION=${3:--1}

# Create OUTPUT_DIR if it does not exist
mkdir -p $OUTPUT_DIR/results

# Convert OUTPUT_DIR to an absolute path
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

# Set GPU_MEM_PROPORTION
if [ "$NUM_PARAMETERS_IN_BILLION" -eq -1 ]; then
    NUM_PARAMETERS_IN_BILLION=$(${ENV_DIR}/venv-harness/bin/python3 ./scripts/count_params_billion.py $MODEL)
fi

# Estimate the proportion of GPU MEMORY NEEDED
GPU_MEM_PROPORTION=""
TARGETED_DP_SIZE=""
TARGETED_TP_SIZE=""
. scripts/estimate_vllm_memory_usage.sh
estimate_vllm_memory_tpdp $NUM_PARAMETERS_IN_BILLION GPU_MEM_PROPORTION TARGETED_TP_SIZE TARGETED_DP_SIZE
echo "VLLM TPDP CONFIG" $NUM_PARAMETERS_IN_BILLION B $GPU_MEM_PROPORTION $TARGETED_TP_SIZE $TARGETED_DP_SIZE

# fixed vars
EVAL_DIR=${ENV_DIR}/src/swallow-evaluation

HARNESS_SCRIPT_PATH=${EVAL_DIR}/scripts/evaluate_english.sh

source ${ENV_DIR}/venv-harness/bin/activate
pushd ${EVAL_DIR}
bash scripts/evaluate_english-vllm.sh $MODEL $GPU_MEM_PROPORTION $OUTPUT_DIR $TARGETED_TP_SIZE $TARGETED_DP_SIZE
popd
deactivate   

mv ${OUTPUT_DIR}/result.json ${OUTPUT_DIR}/results/result.json

echo "Done"
