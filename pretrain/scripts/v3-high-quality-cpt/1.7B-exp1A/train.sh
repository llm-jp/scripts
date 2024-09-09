#!/bin/bash

set -eu -o pipefail

ENV_DIR=environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/scripts/mpi_variables.sh
source ${ENV_DIR}/venv/bin/activate

# open file limit
ulimit -n 65536 1048576

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

# model config
# llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=7168
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((NUM_GPUS / (TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE)))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=512

LR=3e-5
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# data config
# export $TRAIN_DATA_PATH in this script and $TOTAL_TOKEN_SIZE
source ./dataset_token_mapper.sh 

# validation set
VALID_DATA_PATH="" # Skip validation

# total number of iterations
# 210,033,012,552 (number of tokens) / 4096 (seq len) / 512 (batch size) = 100151.54 -> 100152
STEP_DETAIL=$((TOTAL_TOKEN_SIZE / SEQ_LENGTH / GLOBAL_BATCH_SIZE))
LR_DECAY_ITERS=$(awk "BEGIN {print int($STEP_DETAIL + 0.5)}")
LR_DECAY_STYLE=constant
LR_WARMUP_STEPS=0
TRAIN_STEPS=$((LR_WARMUP_STEPS + LR_DECAY_ITERS))

# model config
TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_LOAD_DIR=checkpoints/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-cp${CONTEXT_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=checkpoints/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-cp${CONTEXT_PARALLEL_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# job name
JOB_NAME="llama-2-1.7b-cpt1a"
PROJECT_NAME="high-quality-cpt"
exit 1
# run
export NVTE_FUSED_ATTN=0
python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --load ${CHECKPOINT_LOAD_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path "$TRAIN_DATA_PATH" \
  --split 1000,0,0 \
  --data-cache-path cache \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style ${LR_DECAY_STYLE} \
  --lr-decay-iters "${LR_DECAY_ITERS}" \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --log-interval 1 \
  --eval-interval ${TRAIN_STEPS} \
  --eval-iters 0 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --attention-softmax-in-fp32 \
  --transformer-impl "transformer_engine" \
  --use-mpi \
  --use-z-loss \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project ${PROJECT_NAME} \
  --wandb-entity "llm-jp" \
