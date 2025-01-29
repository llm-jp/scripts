#!/bin/bash
# Node-level moe-recipes launcher
#
# Environment variables that the script expects to be passed from mpirun:
# * TORCH_NCCL_ASYNC_ERROR_HANDLING: Enable/disable async error handling for NCCL (1 for enable, 0 for disable)
# * LD_LIBRARY_PATH: Library path for dynamic linking
# * PATH: System path for executable files
# * MASTER_ADDR: Address of the master node
# * MASTER_PORT: Port number of the master node
# * NUM_NODES: Number of nodes assigned for this task
# * NUM_GPUS_PER_NODE: Number of GPUs in the node assined for this task
# * GRADIENTS_ACCUMULATION_STEPS: Number of gradient accumulation steps
# * MICRO_BATCH_SIZE: Micro batch size for training
# * GLOBAL_BATCH_SIZE: Global batch size for training
# * DEEPSPEED_CONFIG: Path to DeepSpeed configuration file
# * DEEPSPEED_ZERO_STAGE: DeepSpeed ZeRO stage
# * PYTHONPATH: Python module search path

set -eu -o pipefail

source scripts/environment.sh
source scripts/mpi_variables.sh
source venv/bin/activate

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

# training config
SEQ_LENGTH=4096
SLIDING_WINDOW_SIZE=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

TRAIN_STEPS=5000

# optimizer config
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=5000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

ADAMW_BETA1=0.9
ADAMW_BETA2=0.95
ADAMW_EPS=1E-8

# model config
TOKENIZER_MODEL=src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model

CHECKPOINT_DIR=Mixtral-llm-jp-v3-8x1.8B-initial-checkpoint
CHECKPOINT_SAVE_DIR=checkpoints

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0
DATA_PATH=""
DATA_PATH="${DATA_PATH} 2563804308 ${DATASET_DIR}/train/ja/wiki_0000.jsonl_text_document"

# job name
JOB_NAME="test-$(whoami)"

python src/moe-recipes/examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SLIDING_WINDOW_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 99999,1,0 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 $ADAMW_BETA1 \
  --adam-beta2 $ADAMW_BETA2 \
  --adam-eps $ADAMW_EPS \
  --save-interval 10 \
  --eval-interval 1000000000 \
  --eval-iters 1 \
  --bf16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --use-zero \
  --zero-config ${DEEPSPEED_CONFIG} \
  --zero-stage ${DEEPSPEED_ZERO_STAGE} \
  --no-meta-device \
  --output-router-logits \
  --use-mpi \
  --continual-pretraining \
  --wandb-entity "llm-jp" \
  --wandb-project "sakura-test-moe" \
  --wandb-name "${JOB_NAME}"