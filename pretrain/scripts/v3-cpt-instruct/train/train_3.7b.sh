#!/bin/bash

set -eu -o pipefail

# Arguments from sbatch:
#   EXPERIMENT_DIR
#   MASTER_ADDR
#   MASTER_PORT
#   NUM_NODES
#   NUM_GPUS_PER_NODE
#   TASK_DIR

ENV_DIR=${EXPERIMENT_DIR}/environments/train

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

# NOTE(odashi):
# Fix for the Sakura cluster: fused attention doesn't work for some reason.
export NVTE_FUSED_ATTN=0

MODEL_PARAMS=(
  --num-layers 28
  --hidden-size 3072
  --ffn-hidden-size 8192
  --num-attention-heads 24
  --seq-length 4096
  --max-position-embeddings 4096
  --position-embedding-type rope
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

BATCH_PARAMS=(
  --micro-batch-size 2
  --global-batch-size 1024
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
  --use-z-loss
)

SCHEDULER_PARAMS=(
  --train-iters 100000
  --lr-warmup-iters 2000
  --lr-decay-iters 100000
  --lr-decay-style cosine
  --eval-interval 999999999
  --eval-iters 0
)

PARALLELISM_PARAMS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --sequence-parallel
  --use-distributed-optimizer
  --distributed-backend nccl
  --use-mpi
)

# Load TRAIN_DATA_PATH
source ${TASK_DIR}/train_data.sh

DATASET_PARAMS=(
  --data-path ${TRAIN_DATA_PATH[@]}
  --data-cache-path ${TASK_DIR}/cache
  --split 1,0,0
)

BASE_CHECKPOINT_DIR=$(cat ${TASK_DIR}/base_checkpoint_dir.txt)
TASK_CHECKPOINT_DIR=${TASK_DIR}/checkpoints
mkdir -p ${TASK_CHECKPOINT_DIR}

if [ -e ${TASK_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
  # Continue existing training
  CHECKPOINT_PARAMS=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
elif [ -e ${BASE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
  # Start new training based on specified checkpoint
  CHECKPOINT_PARAMS=(
    --load ${BASE_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
  SCHEDULER_PARAMS+=(
    --finetune
    --override-opt_param-scheduler
  )
else
  # Start new training from scratch
  CHECKPOINT_PARAMS=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
fi

IMPLEMENTATION_PARAMS=(
  --bf16
  --use-mcore-models
  --no-masked-softmax-fusion
  --use-flash-attn
  --recompute-activations
  --recompute-granularity "selective"
  --attention-softmax-in-fp32
  --transformer-impl "transformer_engine"
)

LOGGING_PARAMS=(
  --log-interval 1
  --log-throughput
  --wandb-entity llm-jp
  --wandb-project 0095_cpt
  --wandb-name train_$(basename ${TASK_DIR})
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
