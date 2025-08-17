# Pretraining hyperparameters for v4 7.7B.
# Model card: https://github.com/llm-jp/model-cards/pull/30
# Ref: https://github.com/llm-jp/scripts/blob/ec3516a38f93047b7bc0d8305879d62a375e6ee2/pretrain/scripts/v4-training/params/7.7b-cont1.sh

ENV_NAME="$(basename ${ENV_DIR})"
RUN_NAME="8x1.3B--${ATTN_BACKEND}--${ENV_NAME}"

ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 7168
    --num-attention-heads 16
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

# Tokenizer
ALL_PARAMS+=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
)

# Optimizer hyperparameters
ALL_PARAMS+=(
    --optimizer adam
    # --lr 3e-4 # will be defined later
    # --min-lr 3e-5 # will be defined later
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --override-opt_param-scheduler
    # --no-load-optim
)

# pretrain_iters: 1,859,665
# 50B: ceil( 55,797,411,281 / 8192 / 1024 ) == 6652
# 50B sum: 1,859,665 + 6,652 = 1,866,317
# 100B: ceil( 113,460,356,693 / 8192 / 1024 ) == 13,526
# 100B sum: 1,859,665 + 13,526 = 1,873,191
# 300B: ceil( 337,681,167,151 / 8192 / 1024 ) == 40,255
# 300B sum: 1,859,665 + 40,255 = 1,899,920
MIDTRAIN_START=1859665
# TRAIN_ITERS=$(cat ${TASK_DIR}/${PARAM_NAME}/${DATASET_SIZE}/train_iters.txt)
DATASET_SIZE=50B
TRAIN_ITERS=500 # Stop with 500 steps
# MIDTRAIN_ITERS=$((TRAIN_ITERS - MIDTRAIN_START))

# Scheduler
# Scheduler
ALL_PARAMS+=(
    --lr 3e-5   # Start LR
    --min-lr 3e-5  # End LR
    # --min-lr 0  # End LR
    # --lr-warmup-iters ${MIDTRAIN_START} # No warmup
    --lr-warmup-iters 0 # No warmup
    --lr-decay-iters ${TRAIN_ITERS}
    # --lr-decay-iters ${MIDTRAIN_ITERS}
    --lr-decay-style linear
    --train-iters ${TRAIN_ITERS}
    --eval-interval 999999999
    --eval-iters 0
)

# Batch sizes
ALL_PARAMS+=(
    --micro-batch-size 1
    --global-batch-size 1024
)

# Parallelism
ALL_PARAMS+=(
    # model parallel size is set to 2 for 2 node training.
    # (World size (=8 GPUs)) % ((model parallel size) x (moe parallel size)) should be 0.
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    # NOTE(odashi): Increasing timeout is required to prepare 15.6T dataset.
    --distributed-timeout-minutes 120
    --use-mpi
)

# Load TRAIN_DATA_PATH
source ${TASK_DIR}/train_data_${DATASET_SIZE}.sh # options: 50B, 100B, and 300B
SEED=42
# Dataset
ALL_PARAMS+=(
    --data-path ${TRAIN_DATA_PATH[@]}
    --data-cache-path ${TASK_DIR}/${RUN_NAME}/cache
    --split 1,0,0
    --seed ${SEED}
)

TASK_CHECKPOINT_DIR=${TASK_DIR}/${RUN_NAME}/checkpoints
mkdir -p ${TASK_CHECKPOINT_DIR}

# Always start from scratch and disable saving
# if [ -e ${TASK_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
#   # Continue existing training
#   ALL_PARAMS+=(
#     --load ${TASK_CHECKPOINT_DIR}
#     --save ${TASK_CHECKPOINT_DIR}
#   )
#   echo "Continue existing training"
# else
  # Start new training from scratch
  ALL_PARAMS+=(
    --save ${TASK_CHECKPOINT_DIR}
  )
  echo "Start new training from scratch"
# fi
ALL_PARAMS+=(
    --save-interval 1000
)

# Other implementation-related parameters
ALL_PARAMS+=(
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
    --attention-backend ${ATTN_BACKEND}
)

# MoE args
# See https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
ALL_PARAMS+=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-grouped-gemm
    # --moe-permute-fusion # Not compatible with `TE < 2.1.0`
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --use-distributed-optimizer
    --moe-token-dispatcher-type alltoall
)

# NOTE(odashi): Disable fused attention for Sakura cluster due to some inconsistency.
# export NVTE_FUSED_ATTN=0

# Logging
ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --wandb-entity llm-jp
    --wandb-project 0176_merge_megatron_upstream
    --wandb-exp-name "${RUN_NAME}"
)
