#!/bin/bash
#PBS -P gcg51557
#PBS -q R10415
#PBS -N 0130_convert
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -m n


cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
LOGFILE=./logs/convert-${JOBID}.out
ERRFILE=./logs/convert-${JOBID}.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

echo $MEGATRON_CHECKPOINT_DIR
echo $HF_CHECKPOINT_DIR

EXPERIMENT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts/pretrain/scripts/v3-instruct-pretrain-abci/pretrain
ENV_DIR=${EXPERIMENT_DIR}/environments/pretrain-test

## Setup environment
source ${SCRIPT_DIR}/common/setup.sh
echo $(which python)


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
  --saver llama3_hf \
  --load-dir $TMP_DIR \
  --save-dir $HF_CHECKPOINT_DIR \
  --hf-tokenizer-path $TOKENIZER_MODEL_DIR \
  --save-dtype bfloat16 \
  --loader-transformer-impl "transformer_engine" \
  --megatron-path ${ENV_DIR}/src/Megatron-LM

cp ${TOKENIZER_MODEL_DIR}/* $HF_CHECKPOINT_DIR

rm -r $TMP_DIR
echo "Done"
