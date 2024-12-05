#!/bin/bash
#SBATCH --job-name=0087_convert
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -e

MEGATRON_CHECKPOINT_DIR=${1%/}
HF_CHECKPOINT_DIR=$2

ENV_DIR=/home/shared/experiments/0087_llmjp-440m/environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

TOKENIZER_MODEL_DIR=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2

TARGET_ITER_DIR=$(basename $MEGATRON_CHECKPOINT_DIR) # iter_NNNNNNN
ITER=$(( 10#$(echo $TARGET_ITER_DIR | sed 's/^iter_//') )) # NNNNNNN (no 0 padding)
echo ITER=$ITER

if [[ -z "$ITER" || ! "$ITER" =~ ^[0-9]+$ ]]; then # check if directory is valid
  >&2 echo "Error: ITER=$ITER is not a valid number. Exiting."
  exit 1
fi

# Create a unique temporal working directory to avoid affecting the original directory and
# to allow multiple runs to execute simultaneously.
TMP_DIR=$(mktemp -d "${HOME}/ckpt_convert.XXXXXXXX")
>&2 echo TMP_DIR=$TMP_DIR 
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
