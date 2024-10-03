#!/bin/bash
#
# For details about the model, see:
# https://github.com/llm-jp/model-cards/pull/22
set -eu -o pipefail

# EXPERIMENT_DIR=  # set by sbatch
# SCRIPT_ROOT # set by sbatch
# CONF_DIR # set by sbatch
MODEL_SIZE="3.7B"
EXP_NAME="${MODEL_SIZE}-${CONF_DIR}"
ENV_DIR=${EXPERIMENT_DIR}/environment
WORK_DIR=${EXPERIMENT_DIR}/${EXP_NAME}
CACHE_DIR=${WORK_DIR}/cache
SCRIPT_DIR=${SCRIPT_ROOT}/${CONF_DIR}
mkdir -p "$WORK_DIR"

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

# model config
HIDDEN_SIZE=3072
FFN_HIDDEN_SIZE=8192
NUM_LAYERS=28
NUM_HEADS=24
SEQ_LENGTH=4096


# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024

LR_WARMUP_INIT=3e-5
LR=3e-5
MIN_LR=3e-6
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# data config
DATA_CONFIG="$WORK_DIR/data_config.sh"
python3 "${SCRIPT_ROOT}/megatron_data_formatter.py" "${SCRIPT_DIR}/data_config.yaml" > "$DATA_CONFIG" 2> /dev/null
source "$DATA_CONFIG"

# load $TRAIN_DATA_PATH and $TOTAL_TOKEN_SIZE


# validation set
VALID_DATA_PATH="" # Skip validation

# total number of iterations
# 210,033,012,552 (number of tokens) / 4096 (seq len) / 512 (batch size) = 100151.54 -> 100152
STEP_DETAIL=$((TOTAL_TOKEN_SIZE / SEQ_LENGTH / GLOBAL_BATCH_SIZE))
LR_DECAY_ITERS=$(awk "BEGIN {print int($STEP_DETAIL + 0.5)}")
LR_DECAY_STYLE=cosine
LR_WARMUP_STEPS=0
TRAIN_STEPS=$((LR_WARMUP_STEPS + LR_DECAY_ITERS))

# model config
TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model

CHECKPOINT_ROOT=${WORK_DIR}/checkpoints
CHECKPOINT_SAVE_DIR=${CHECKPOINT_ROOT}

CHECKPOINT_ARGS=""
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_LOAD_DIR=${CHECKPOINT_SAVE_DIR}
else
  # first training
  CHECKPOINT_LOAD_DIR=${EXPERIMENT_DIR}/pretrained_checkpoints/3.bb-exp1
  CHECKPOINT_ARGS="--finetune"
fi

mkdir -p ${CHECKPOINT_SAVE_DIR}

# job name
WANDB_ENTITY="llm-jp"
WANDB_PROJECT="high-quality-cpt"
WANDB_NAME=$EXP_NAME

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
  $CHECKPOINT_ARGS \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 1,0,0 \
  --data-cache-path ${CACHE_DIR} \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr-warmup-init ${LR_WARMUP_INIT} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style ${LR_DECAY_STYLE} \
  --lr-decay-iters ${LR_DECAY_ITERS} \
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
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-name ${WANDB_NAME}
