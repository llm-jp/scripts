# Pretraining hyperparameters for v4 1.3B.
# Model card: https://github.com/llm-jp/model-cards/pull/31
# Ref: https://github.com/llm-jp/scripts/blob/ec3516a38f93047b7bc0d8305879d62a375e6ee2/pretrain/scripts/v4-training/params/1.3b-cont1.sh

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

# ceil( 55,797,411,281 / 8192 / 1024 ) == 6652
# pretrain_iters: 1859665
# sum: 1859665+6652=1,866,317
# TRAIN_ITERS=1866317
TRAIN_ITERS=$(cat ${TASK_DIR}/train_iters.txt)

# Scheduler
ALL_PARAMS+=(
    --train-iters ${TRAIN_ITERS}
    --lr-warmup-iters 2000
    --lr-decay-iters ${TRAIN_ITERS}
    --lr-decay-style cosine
    --eval-interval 999999999
    --eval-iters 0
)

# Batch sizes
ALL_PARAMS+=(
    --micro-batch-size 2
    --global-batch-size 1024
)

# Parallelism
ALL_PARAMS+=(
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
source ${TASK_DIR}/train_data.sh
# Dataset
ALL_PARAMS+=(
    --data-path ${TRAIN_DATA_PATH[@]}
    --data-cache-path ${TASK_DIR}/cache
    --split 1,0,0
)

    TASK_CHECKPOINT_DIR=${TASK_DIR}/checkpoints
mkdir -p ${TASK_CHECKPOINT_DIR}

if [ -e ${TASK_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
  # Continue existing training
  ALL_PARAMS+=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
  echo "Continue existing training"
else
  # Start new training from scratch
  ALL_PARAMS+=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
  echo "Start new training from scratch"
fi
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
    #--attention-backend flash
)

# NOTE(odashi): Disable fused attention for Sakura cluster due to some inconsistency.
export NVTE_FUSED_ATTN=0

# Logging
ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    # --wandb-entity llm-jp
    # --wandb-project 0156_midtrain
    # --wandb-exp-name train_$(basename ${TASK_DIR})
)
