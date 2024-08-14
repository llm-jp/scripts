#!/bin/bash
# Model conversion script for converting Megatron format checkpoints into Hugging Face format
#
# This script needs one node on the `gpu` partition of the cluster.
# However, a GPU is necessary to verify CUDA functionality, even though no VRAM will be used.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --partition {partition} convert.sh SOURCE_DIR TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash convert.sh SOURCE_DIR TARGET_DIR TEMPORAL_DIR > outpus/convert.out 2> outputs/convert.err`
# - SOURCE_DIR: Megatron checkpoint directory including `iter_NNNNNNN`
# - TARGET_DIR: Output directory for the Hugging Face format
#
# Example:
# sbatch convert.sh /data/experiments/{exp-id}/checkpoints/iter_0001000 /data/experiments/{exp-id}/hf_checkpoints/iter_0001000 
#
#SBATCH --job-name=ckpt-convert
#SBATCH --partition=<FIX_ME>
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -e

MEGATRON_CHECKPOINT_DIR=${1%/}
HF_CHECKPOINT_DIR=$2

ENV_DIR=environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

TOKENIZER_MODEL_DIR=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2

TARGET_ITER_DIR=$(basename $MEGATRON_CHECKPOINT_DIR) # iter_NNNNNNN
ITER=$(echo $TARGET_ITER_DIR | sed 's/^iter_0*//') # NNNNNNN (no 0 padding)
if [[ -z "$ITER" || ! "$ITER" =~ ^[0-9]+$ ]]; then # check if directory is valid
  echo "Error: ITER is not a valid number. Exiting."
  exit 1
fi

# Create a unique temporal working directory to avoid affecting the original directory and 
# to allow multiple runs to execute simultaneously.
TMP_DIR=${HOME}/ckpt_convert_$(date +%Y%m%d%H%M%S)
mkdir -p "${TMP_DIR}"
ln -s $(readlink -f $MEGATRON_CHECKPOINT_DIR) ${TMP_DIR}/${TARGET_ITER_DIR}
echo $ITER > "${TMP_DIR}/latest_checkpointed_iteration.txt"

echo "Converting $MEGATRON_CHECKPOINT_DIR"

python ${ENV_DIR}/src/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama2_hf \
  --load-dir $TMP_DIR \
  --save-dir $HF_CHECKPOINT_DIR \
  --hf-tokenizer-path $TOKENIZER_MODEL_DIR \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path ${ENV_DIR}/src/Megatron-LM

cp ${TOKENIZER_MODEL_DIR}/* $HF_CHECKPOINT_DIR

rm -r $TMP_DIR
echo "Done"
