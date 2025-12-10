# LLM-jp v4 model 32B

ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
    --num-layers 64
    --hidden-size 5120
    --ffn-hidden-size 27648
    --num-attention-heads 40
    --group-query-attention
    --num-query-groups 8

    # NOTE(odashi): We set 4096 (not 8192) for context length to award more training steps
    --seq-length 4096
    --max-position-embeddings 4096

    --position-embedding-type rope
    --rotary-base 500000
    --untie-embeddings-and-output-weights
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
    --lr 2e-4
    --min-lr 2e-5
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
# At least 10_526_120 steps required to train all tokens.
# ( == ceil[22_074_871_647_659 / 512 / 4096] )
ALL_PARAMS+=(
    --train-iters 12000000
    --lr-warmup-iters 2000
    --lr-decay-iters 12000000
    --lr-decay-style WSD

    # NOTE(odashi): We run stable training: don't apply decay until the last step.
    --lr-wsd-decay-style linear
    --lr-wsd-decay-iters 1

    --eval-interval 999999999
    --eval-iters 0
)

# Batch sizes
ALL_PARAMS+=(
    --micro-batch-size 2
    --global-batch-size 512
)

# Parallelism
ALL_PARAMS+=(
    --tensor-model-parallel-size 4
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