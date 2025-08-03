# Pretraining hyperparameters for v4 7.7B.
# Model card: https://github.com/llm-jp/model-cards/pull/30
# Ref: https://github.com/llm-jp/scripts/blob/ec3516a38f93047b7bc0d8305879d62a375e6ee2/pretrain/scripts/v4-training/params/7.7b-cont1.sh

ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
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

# Tokenizer
ALL_PARAMS+=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.1/llm-jp-tokenizer-100k.ver3.1.model # TODO
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
    --no-load-optim
)

# pretrain_iters: 432,581
# 80B: ceil( 83,527,699,000/ 8192 / 1024) == 9958
# 80B sum: 432,581 + 9,958 = 442,539
MIDTRAIN_START=432581
TRAIN_ITERS=$(cat ${TASK_DIR}/${PARAM_NAME}/${DATASET_SIZE}/train_iters.txt) # 442539
MIDTRAIN_ITERS=$((TRAIN_ITERS - MIDTRAIN_START))

# Scheduler
# Scheduler
ALL_PARAMS+=(
    --lr 3e-5   # Start LR
    --min-lr 3e-5  # End LR
    # --min-lr 0  # End LR
    # --lr-warmup-iters ${MIDTRAIN_START} # No warmup
    --lr-warmup-iters 0 # No warmup
    # --lr-decay-iters ${TRAIN_ITERS}
    --lr-decay-iters ${MIDTRAIN_ITERS}
    --lr-decay-style linear
    --train-iters ${TRAIN_ITERS}
    --eval-interval 999999999
    --eval-iters 0
)

# Batch sizes
ALL_PARAMS+=(
    --micro-batch-size 1
    # --global-batch-size 512
    # --micro-batch-size 2
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
source ${TASK_DIR}/train_data_${DATASET_SIZE}.sh # options: 80B
SEED=42
# Dataset
ALL_PARAMS+=(
    --data-path ${TRAIN_DATA_PATH[@]}
    --data-cache-path ${TASK_DIR}/${PARAM_NAME}/${DATASET_SIZE}/cache
    --split 1,0,0
    --seed ${SEED}
)

    TASK_CHECKPOINT_DIR=${TASK_DIR}/${PARAM_NAME}/${DATASET_SIZE}/checkpoints
mkdir -p ${TASK_CHECKPOINT_DIR}

if [ -e ${TASK_CHECKPOINT_DIR}/${PARAM_NAME}/${DATASET_SIZE}/latest_checkpointed_iteration.txt ]; then
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
    --wandb-entity llm-jp
    --wandb-project 0193_midtrain
    --wandb-exp-name train_$(basename ${TASK_DIR})
)
