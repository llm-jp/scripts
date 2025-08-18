#!/bin/bash
# Model conversion script for converting Hugging Face format into Megatron format checkpoints.
#
# This script requires one node with a GPU in the `gpu` partition of the cluster.
# The GPU is necessary to validate the CUDA environment, although no significant VRAM usage occurs.
#
# **Usage**:
# On a SLURM-managed cluster:
#   sbatch --partition {partition} hf2megatron.sh SOURCE_DIR TARGET_DIR
# On a system without SLURM:
#   bash hf2megatron.sh SOURCE_DIR TARGET_DIR > outputs/convert.out 2> outputs/convert.err
#
# **Parameters**:
# - SOURCE_DIR: Directory containing Hugging Face model checkpoints.
# - TARGET_DIR: Directory for the converted Megatron checkpoints. The directory name must include
#   tensor parallel and pipeline parallel sizes in the format `tp{N:int}` and `pp{M:int}`.
#
# **Example**:
# sbatch hf2megatron.sh /data/experiments/{exp-id}/hf_checkpoints /data/experiments/{exp-id}/checkpoints/tp2-pp2-cp1
#
# Example
# sbatch hf2megatron.sh /data/experiments/{exp-id}/hf_checkpoints /data/experiments/{exp-id}/checkpoints/tp2-pp2-cp1
#
#SBATCH --job-name=ckpt-convert
#SBATCH --partition=<FIX_ME>
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eux -o pipefail

if [ $# -ne 2 ]; then
  echo >&2 "Usage: sbatch --partition {partition} (or bash) convert.sh SOURCE_DIR TARGET_DIR"
  echo >&2 "Example: sbatch hf2megatron.sh /data/hf_checkpoints /data/checkpoints/tp2-pp2-cp1"
  exit 1
fi

HF_CHECKPOINT_DIR=${1%/}
MEGATRON_CHECKPOINT_DIR=${2%/}

ENV_DIR=environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model

# Extract parallel sizes from directory name
TARGET_DIR=$(basename $MEGATRON_CHECKPOINT_DIR) # e.g., tp2-pp2-cp1
TENSOR_PARALLEL_SIZE=$(echo $TARGET_DIR | sed -E 's/.*tp([0-9]+).*/\1/')
PIPELINE_PARALLEL_SIZE=$(echo $TARGET_DIR | sed -E 's/.*pp([0-9]+).*/\1/')

if [[ -z "$TENSOR_PARALLEL_SIZE" || -z "$PIPELINE_PARALLEL_SIZE" ]]; then
  echo "Error: Invalid directory name format. Expected format: tp{N:int} and pp{M:int}"
  echo "Example: sbatch hf2megatron.sh hf_checkpoints checkpoints/tp2-pp2-cp1"
  exit 1
fi

echo TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
echo PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

echo "Converting ${HF_CHECKPOINT_DIR} to ${MEGATRON_CHECKPOINT_DIR}"

python ${ENV_DIR}/src/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama2_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine" \
  --megatron-path ${ENV_DIR}/src/Megatron-LM

echo "Done"
