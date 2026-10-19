# LLM-jp v4 model 2B

ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
    --num-layers 24
    --hidden-size 3072
    --ffn-hidden-size 7168
    --num-attention-heads 24
    --group-query-attention
    --num-query-groups 8

    # NOTE(odashi): We set 4096 (not 8192) for context length to award more training steps
    --seq-length 4096
    --max-position-embeddings 4096

    --position-embedding-type rope
    --rotary-base 500000
    #--untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
)

# Tokenizer
ALL_PARAMS+=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${ENV_DIR}/src/llm-jp-tokenizer-v4/v4_alpha_1.0.model
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

# Scheduler
ALL_PARAMS+=(
    --train-iters 4000000
    --lr-warmup-iters 2000
    --lr-decay-iters 4000000
    --lr-decay-style WSD

    # NOTE(odashi): We run stable training: don't apply decay until the last step.
    --lr-wsd-decay-style linear
    --lr-wsd-decay-iters 1

    --eval-interval 999999999
    --eval-iters 0
)

# Batch sizes
ALL_PARAMS+=(
    # Requires at most 16 nodes = 128 GPUs
    --micro-batch-size 4
    --global-batch-size 512
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

# Dataset
ALL_PARAMS+=(
    --data-path ${TRAIN_DATA_PATH[@]}
    --data-cache-path ${TASK_DIR}/cache
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
    --overlap-grad-reduce
    --overlap-param-gather

    --attention-softmax-in-fp32
    --transformer-impl transformer_engine

    # NOTE(sosuke): According to my experiments, fused attention backend is the fastest in available options (including flash_attn_3).
    # ref: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html
    --attention-backend fused
)

# NOTE(odashi):
# https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#communication-overlaps-and-tuning
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

# Overrides for scheduler decay
ALL_PARAMS+=(
    --override-opt_param-scheduler
    --min-lr 0
    --train-iters 3000000
    --lr-decay-iters 3000000
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-iters 100000
)
