#!/bin/bash

set -eu -o pipefail

source scripts/environment.sh
source scripts/mpi_variables.sh
source venv/bin/activate

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

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

# model config
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=1792
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2
EXPERT_PARALLEL_SIZE=8
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE} * ${EXPERT_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# total number of iterations
# 2072488058295 (number of tokens) / 4096 (seq len) / 1024 (batch size) = 494119.65806365 -> 494120
LR_WARMUP_STEPS=2000
LR_DECAY_ITERS=20000
TRAIN_STEPS=20000

CACHE_DIR=cache_v4_exp3

# model config
TOKENIZER_MODEL=src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_LOAD_DIR=checkpoints/megatron/16expert_4granularity_0shared_top8_17b_active2.9b_z_loss_v4_exp3
CHECKPOINT_SAVE_DIR=checkpoints/megatron/16expert_4granularity_0shared_top8_17b_active2.9b_z_loss_v4_exp3

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

TRAIN_DATA_PATH=""


# en
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5494262694 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolma-books_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3896965449 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolma-wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1464772187 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolmino-stackexchange_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10335599308 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_finemath-4plus_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2781710 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_gsm8k_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9176535715 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_mathpile_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13280211413 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_olmo-algebraicstack_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22219529548 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_olmo-arxiv_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13395295861 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_olmo-openwebmath_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4744259830 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_wiki_0000_text_document"

WANDB_ENTITY="llm-jp"
WANDB_PROJECT="v3-8x1.8b"
WANDB_NAME="16expert_4granularity_0shared_top8_17b_active2.9b_z_loss_v4_exp3"


# Model arguments
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_HEADS}
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --num-query-groups 16
    --no-masked-softmax-fusion
    --rotary-base 10000
)

MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 8
    --moe-z-loss-coeff 1e-3
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $TRAIN_DATA_PATH
    --data-cache-path $CACHE_DIR
    --split 1,0,0
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --train-iters ${TRAIN_STEPS}
    --lr-decay-iters ${TRAIN_STEPS}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --weight-decay ${WEIGHT_DECAY}
    --lr-warmup-iters ${LR_WARMUP_STEPS}
    --clip-grad ${GRAD_CLIP}
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --bf16
    --use-flash-attn
    --transformer-impl "transformer_engine"
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --distributed-backend nccl
    --ckpt-format torch
)

# Model parameters
MODEL_PARALLEL_ARGS=(
   --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
   --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}
   --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --use-distributed-optimizer
   --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --moe-per-layer-logging
    --save-interval 500
    --eval-interval ${TRAIN_STEPS}
    --eval-iters 0
    --save ${CHECKPOINT_SAVE_DIR}
    --load ${CHECKPOINT_LOAD_DIR}
    --use-mpi
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_NAME}
    --wandb-entity ${WANDB_ENTITY}
)


export NVTE_FUSED_ATTN=0
python src/Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
