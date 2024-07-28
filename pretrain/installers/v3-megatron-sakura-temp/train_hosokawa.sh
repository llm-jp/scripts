#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=megatron-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks-per-node=1

set -eu -o pipefail

# cd ./llm-jp-Megatron-LM

module load cuda/12.1
module load /data/cudnn-tmp-install/modulefiles/8.9.4
module load hpcx/2.17.1-gcc-cuda12/hpcx
module load nccl/2.20.5

# distributed settings
export MASTER_ADDR="$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=12800

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="h100"
export NUM_GPU_PER_NODE=1

NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# model config
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=7168 # intermediate size (HuggingFace)
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
TRAIN_STEPS=61000 #今回は約250B Tokens
LR_DECAY_ITERS=61000

LR=1e-4
MIN_LR=1e-5
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_DIR=/dev/null
CHECKPOINT_SAVE_DIR=checkpoints

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0
DATA_PATH=""
DATA_PATH="${DATA_PATH} 2563804308 ${DATASET_DIR}/train/ja/wiki_0000.jsonl_text_document"

# job name
JOB_NAME="test-odashi"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# checkpoint load
# first training
CHECKPOINT_ARGS="" #"--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"

# run
mpirun \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  -x CUDNN_PATH \
  -x CPATH \
  -x LD_LIBRARY_PATH \
  -x CUDNN_PATH \
  -x LIBRARY_PATH \
  -x CUDA_HOME \
  -x CUDA_LAUNCH_BLOCKING \
  -x CUDNN_LOGDEST_DBG=stderr \
  -x CUDNN_LOGERR_DBG=1 \
  python pretrain_gpt.py \
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
  --split 949,50,1 \
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
  --eval-interval 100 \
  --eval-iters 10 \
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
