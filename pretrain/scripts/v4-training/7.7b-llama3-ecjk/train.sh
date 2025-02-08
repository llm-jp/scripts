#!/bin/bash

# LLM-jp v4 7.7B (Llama3 8B compatible) training script.
# Model card: https://github.com/llm-jp/model-cards/pull/30

set -eu -o pipefail

# Arguments from sbatch:
#   MASTER_ADDR
#   MASTER_PORT
#   NUM_NODES
#   NUM_GPUS_PER_NODE
#   ENV_DIR
#   MODEL_DIR
#   SCRIPT_DIR
#   WANDB_ENTITY
#   WANDB_PROJECT

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

MODEL_PARAMS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 500000
    --untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --disable-bias-linear
)

TOKENIZER_PARAMS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
)

OPTIMIZER_PARAMS=(
    --optimizer adam
    --lr 3e-4
    --min-lr 3e-5
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# ceil( 15.6T / 8192 / 1024 ) == 1859665
TRAIN_ITERS=1859665

SCHEDULER_PARAMS=(
    --train-iters ${TRAIN_ITERS}
    --lr-warmup-iters 2000
    --lr-decay-iters ${TRAIN_ITERS}
    --lr-decay-style cosine
    --eval-interval 999999999
    --eval-iters 0
)

BATCH_PARAMS=(
    --micro-batch-size 1
    --global-batch-size 1024
)

PARALLELISM_PARAMS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    # NOTE(odashi): Increasing timeout is required to prepare 15.6T dataset.
    --distributed-timeout-minutes 120
    --use-mpi
)

# Load TRAIN_DATA_PATH
source ${SCRIPT_DIR}/../train_data/llama3_simulation_15_6t.sh

DATASET_PARAMS=(
    --data-path ${TRAIN_DATA_PATH[@]}
    --data-cache-path ${MODEL_DIR}/cache
    --split 1,0,0
)

TASK_CHECKPOINT_DIR=${MODEL_DIR}/checkpoints

CHECKPOINT_PARAMS=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
    --save-interval 1000
)

IMPLEMENTATION_PARAMS=(
    --bf16
    --use-mcore-models
    --no-masked-softmax-fusion
    --use-flash-attn

    # NOTE(odashi): For adjusting throughput
    #--recompute-activations
    #--recompute-granularity selective
    #--overlap-grad-reduce
    #--overlap-param-gather

    --attention-softmax-in-fp32
    --transformer-impl transformer_engine

    # NOTE(odashi): Newer implementation requires to set attention backend by parameter.
    #--attention-backend flash
)

# NOTE(odashi): Disable fused attention for Sakura cluster due to some inconsistency.
export NVTE_FUSED_ATTN=0

LOGGING_PARAMS=(
    --log-interval 1
    --log-throughput
    --wandb-entity ${WANDB_ENTITY}
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name train_$(date '+%Y%m%d-%H%M%S')_${SLURM_JOB_ID}
)

python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
    ${MODEL_PARAMS[@]} \
    ${TOKENIZER_PARAMS[@]} \
    ${BATCH_PARAMS[@]} \
    ${OPTIMIZER_PARAMS[@]} \
    ${SCHEDULER_PARAMS[@]} \
    ${PARALLELISM_PARAMS[@]} \
    ${DATASET_PARAMS[@]} \
    ${CHECKPOINT_PARAMS[@]} \
    ${IMPLEMENTATION_PARAMS[@]} \
    ${LOGGING_PARAMS[@]}
