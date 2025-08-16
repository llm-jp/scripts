# Pretraining hyperparameters for DeepSeek-V2-Lite.
# Model details: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# 15.7B total parameters, 2.4B activated per token

ENV_NAME="$(basename ${ENV_DIR})"
RUN_NAME="deepseek-v2-lite-gqa--${ATTN_BACKEND}--${ENV_NAME}"

ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
    --num-layers 27
    --hidden-size 2048
    # FFN hidden size will be set by MoE configuration
    --num-attention-heads 16
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-base 10000
    --rotary-percent 1.0
    --rotary-scaling-factor 40
    --mscale 0.707
    --mscale-all-dim 0.707
    --untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    # Disable Multi-Latent Attention (MLA) parameters
    # --multi-latent-attention
    # Grouped Query Attention (GQA) parameters
    --group-query-attention
    --num-query-groups 8
    --kv-channels 16
    --kv-lora-rank 512
    --v-head-dim 128
    --qk-head-dim 128
    --qk-layernorm
    --qk-pos-emb-head-dim 64
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

# Dataset configuration
DATASET_SIZE=50B
TRAIN_ITERS=200 # Training speed saturates until 200 steps

# Scheduler
ALL_PARAMS+=(
    --lr 3e-5   # Start LR
    --min-lr 3e-5  # End LR
    --lr-warmup-iters 0 # No warmup
    --lr-decay-iters ${TRAIN_ITERS}
    --lr-decay-style linear
    --train-iters ${TRAIN_ITERS}
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

# Start new training from scratch
ALL_PARAMS+=(
    --save ${TASK_CHECKPOINT_DIR}
)
echo "Start new training from scratch"
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

# MoE args for DeepSeek-V2-Lite architecture
# DeepSeek-V2-Lite: 2 shared experts + 64 routed experts, 6 experts activated per token
# See the following related resources:
# - Megatron-LM MoE docs: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
# - Megatron-LM DeepSeek-V2-Lite config: https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/conf/deepseek-ai/DeepSeek-V2-Lite.sh
# - Hugging Face DeepSeek-V2-Lite repo: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
ALL_PARAMS+=(
    --num-experts 64
    --expert-model-parallel-size 8
    --moe-layer-freq "([0]+[1]*26)"  # MoE on all layers except first
    --moe-ffn-hidden-size 1408
    --moe-grouped-gemm
    # --moe-permute-fusion # Not compatible with `TE < 2.1.0`
    --moe-router-score-function softmax
    --moe-router-topk 6  # 6 experts activated per token in DeepSeek-V2-Lite
    --moe-router-topk-scaling-factor 1.0
    --moe-router-pre-softmax
    --moe-shared-expert-intermediate-size 2816  # 2 * 1408
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall
    # --moe-token-dispatcher-type flex
    # --moe-enable-deepep # DeepEP: https://github.com/deepseek-ai/deepep
    # --moe-router-dtype fp32
    # --external-cuda-graph
    # --cuda-graph-scope all
    # --moe-expert-capacity-factor 1.0
    # --moe-pad-expert-input-to-capacity
    --moe-token-drop-policy probs
    --moe-router-load-balancing-type seq_aux_loss
    --use-distributed-optimizer
    # FFN hidden size for MoE layers
    --ffn-hidden-size 10944
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
