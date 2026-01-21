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

if [[ $# -lt 2 || $# -gt 5 ]]; then
    >&2 echo "Usage: $0 MODEL OUTPUT_DIR [GPU_MEMORY_UTILIZATION] [TENSOR_PARALLEL_SIZE] [DATA_PARALLEL_SIZE]"
    >&2 echo "Defaults: GPU_MEMORY_UTILIZATION=0.9, TENSOR_PARALLEL_SIZE=1, DATA_PARALLEL_SIZE=1"
    exit 1
fi

# Arguments
MODEL=$1
OUTPUT_DIR=$2
GPU_MEMORY_UTILIZATION=${3:-0.9}
TENSOR_PARALLEL_SIZE=${4:-1}
DATA_PARALLEL_SIZE=${5:-1}


# Create OUTPUT_DIR if it does not exist
mkdir -p $OUTPUT_DIR/results

# Convert OUTPUT_DIR to an absolute path
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

# fixed vars
EVAL_DIR=${ENV_DIR}/src/swallow-evaluation

HARNESS_SCRIPT_PATH=${EVAL_DIR}/scripts/evaluate_english.sh

source ${ENV_DIR}/venv-harness/bin/activate
pushd ${EVAL_DIR}
bash scripts/evaluate_english-vllm.sh $MODEL $GPU_MEMORY_UTILIZATION $OUTPUT_DIR $TENSOR_PARALLEL_SIZE $DATA_PARALLEL_SIZE
popd
deactivate   

mv ${OUTPUT_DIR}/result.json ${OUTPUT_DIR}/results/result.json

echo "Done"
