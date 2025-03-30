#!/bin/bash
#PBS -P gcg51557
#PBS -q R10415
#PBS -N 0130_convert
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -v RTYPE=rt_HF
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/convert-${JOBID}.out
ERRFILE=./logs/convert-${JOBID}.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

EXPERIMENT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining
SCRIPT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining/scripts/pretrain/scripts/v3-instruct-pretrain-abci/pretrain
ENV_DIR=${EXPERIMENT_DIR}/environments/pretrain_torch_v2.6.0

# Setup environment
source ${SCRIPT_DIR}/common/setup.sh

TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/hf/ver3.0/llm-jp-tokenizer-100k.ver3.0b2
MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

TARGET_TP_SIZE=1
TARGET_EP_SIZE=8
TARGET_PP_SIZE=2

HF_FORMAT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining/pretrained_checkpoints/llmjp3-8x13b-hf/iter_0494120
MEGATRON_FORMAT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining/pretrained_checkpoints/llmjp3-8x13b-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}
mkdir -p $MEGATRON_FORMAT_DIR

python ${MEGATRON_PATH}/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader mixtral_hf \
    --saver mcore \
    --target-tensor-parallel-size ${TARGET_TP_SIZE} \
    --target-pipeline-parallel-size ${TARGET_PP_SIZE} \
    --target-expert-parallel-size ${TARGET_EP_SIZE} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --true-vocab-size 99574 \
    --tokenizer-model ${TOKENIZER_MODEL}