#!/bin/bash
# Model conversion script for converting Megatron format checkpoints into Huggingface format
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch convert.sh SOURCE_DIR TARGET_DIR TEMPORAL_DIR` 
# On a cluster without SLURM:
#   Run `bash convert.sh SOURCE_DIR TARGET_DIR TEMPORAL_DIR` 
# - SOURCE_DIR: Megatron checkpoint directory including `iter_NNNNNNN`
# - TARGET_DIR: Output directory for the Hugging Face format
# - TEMPORAL_DIR: Temporary directory for intermediate files (optional)
#
# This script requires 1 node on the `cpu` partition on the cluster.

#SBATCH --job-name=ckpt-convert
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=ologs/%x-%j.err

set -e

ENV_DIR=environment
TMP_DIR_DEFAULT=$HOME/tmp

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

MEGATRON_CHECKPOINT_DIR=${1%/}
HF_CHECKPOINT_DIR=$2
TMP_DIR={$3:-$TMP_DIR_DEFAULT}

TMP_DIR=${TMP_DIR}_$(date +%Y%m%d%H%M%S)

TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2

ITER=$(echo $MEGATRON_CHECKPOINT_DIR | grep -oP 'iter_\K[0-9]+' | sed 's/^0*//')
if [[ -z "$ITER" || ! "$ITER" =~ ^[0-9]+$ ]]; then
  echo "Error: ITER is not a valid number. Exiting."
  exit 1
fi
FORMATTED_ITERATION=$(printf "%07d" $ITER)

mkdir -p "${TMP_DIR}"
ln -s $MEGATRON_CHECKPOINT_DIR $TMP_DIR/iter_${FORMATTED_ITERATION}
echo $ITER > "${TMP_DIR}/latest_checkpointed_iteration.txt"

echo "Converting $MEGATRON_CHECKPOINT_DIR"

python ${ENV_DIR}/src/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama2_hf \
  --load-dir $TMP_DIR \
  --save-dir $HF_CHECKPOINT_DIR \
  --hf-tokenizer-path ${TOKENIZER_MODEL} \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path ${ENV_DIR}/src/Megatron-LM

cp $TOKENIZER_MODEL_DIR/* $HF_CHECKPOINT_DIR

rm -rf $TMP_DIR
echo "Done"
