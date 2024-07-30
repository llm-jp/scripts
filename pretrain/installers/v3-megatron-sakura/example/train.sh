#!/bin/bash
# Node-level Megatron-LM launcher
#
# Environment variables that the script expects to be passed from mpirun:
# * MASTER_ADDR: Address of the master node
# * MASTER_PORT: Port number of the master node
# * NUM_NODES: Number of nodes assigned for this task
# * NUM_GPUS_PER_NODE: Number of GPUs in the node assined for this task

set -eu -o pipefail

source scripts/environment.sh
source scripts/mpi_variables.sh
source venv/bin/activate

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
FFN_HIDDEN_SIZE=7168
NUM_LAYERS=24
NUM_HEADS=16
NUM_QUERY_GROUPS=8
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024

LR=1e-4
MIN_LR=1e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

LR_WARMUP_STEPS=1000
LR_DECAY_ITERS=5000
TRAIN_STEPS=$((${LR_WARMUP_STEPS} + ${LR_DECAY_ITERS}))

# model config
TOKENIZER_MODEL=src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_SAVE_DIR=checkpoints

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0
DATA_PATH=""
DATA_PATH="${DATA_PATH} 2563804308 ${DATASET_DIR}/train/ja/wiki_0000.jsonl_text_document"

# job name
JOB_NAME="test-odashi"

# checkpoint load
# first training
CHECKPOINT_ARGS="" #"--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"

python src/Megatron-LM/pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_QUERY_GROUPS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
    ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --split 1,0,0 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --log-interval 1 \
  --eval-interval 1000000000 \
  --eval-iters 1 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
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
  --wandb-name ${JOB_NAME} \
  --wandb-project "sakura-test" \
  --wandb-entity "llm-jp"
