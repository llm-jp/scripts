ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
    --num-layers 40
    --hidden-size 5120
    --ffn-hidden-size 13824
    --num-attention-heads 40
    --seq-length 4096
    --max-position-embeddings 4096
    --position-embedding-type rope
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
    --micro-batch-size 4
    --global-batch-size 1024
)

# Parallelism
ALL_PARAMS+=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    # NOTE(odashi): Increasing timeout is required to prepare 15.6T dataset.
    --distributed-timeout-minutes 5
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

BASE_CHECKPOINT_DIR=$(cat ${TASK_DIR}/base_checkpoint_dir.txt)
TASK_CHECKPOINT_DIR=${TASK_DIR}/checkpoints
mkdir -p ${TASK_CHECKPOINT_DIR}

if [ -e ${TASK_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
  # Continue existing training
  ALL_PARAMS+=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
  echo "Continue existing training"
elif [ -e ${BASE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
  # Start new training based on specified checkpoint
  ALL_PARAMS+=(
    --load ${BASE_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
    --finetune
    --override-opt_param-scheduler
  )
  echo "Start new training based on specified checkpoint"
else
  # Start new training from scratch
  ALL_PARAMS+=(
    --load ${TASK_CHECKPOINT_DIR}
    --save ${TASK_CHECKPOINT_DIR}
  )
  echo "Start new training from scratch"
fi
ALL_PARAMS+=(
    --save-interval 100
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
#    --attention-backend flash
)

# Logging
ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --wandb-entity llm-jp
    --wandb-project 0130_instruction_pretraining
    --wandb-exp-name $(basename ${TASK_DIR})
)