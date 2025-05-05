# Pretraining hyperparameters for v4 7.7B.
# Model card: https://github.com/llm-jp/model-cards/pull/30

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

# ceil( 15.6T / 8192 / 1024 ) == 1859665
TRAIN_ITERS=1859665

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
    --micro-batch-size 1
    --global-batch-size 1024
)

# Parallelism
ALL_PARAMS+=(
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

# Dataset
ALL_PARAMS+=(
    --data-cache-path ${MODEL_DIR}/cache
    --split 1,0,0
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
)