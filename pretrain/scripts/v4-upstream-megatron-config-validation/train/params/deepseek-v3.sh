# Pretraining hyperparameters for DeepSeek-V3 (R1-equivalent hparams).
# V3 model hyperparameters largely match DeepSeek-R1.

ENV_NAME="$(basename ${ENV_DIR})"
RUN_NAME="deepseek-v3-16layer"

ALL_PARAMS=()

# =========================
# Model hyperparameters (R1/V3 style)
# =========================
N_LAYERS=16  # Original DeepSeek-V3 has 61 layers, but reduced to avoid OOM
ALL_PARAMS+=(
    # --num-layers 61
    --num-layers "$N_LAYERS"  # Set custom number of layers
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128

    --seq-length 4096
    --max-position-embeddings 163840

    --position-embedding-type rope
    --no-rope-fusion
    --rotary-base 10000
    --rotary-percent 1.0
    --rotary-scaling-factor 40

    --mscale 1.0
    --mscale-all-dim 1.0

    --untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear

    # Multi-Latent Attention (MLA) parameters (R1/V3)
    # --multi-latent-attention
    # Grouped Query Attention (GQA)
    --group-query-attention
    --kv-channels 128
    --kv-lora-rank 512
    --v-head-dim 128
    --q-lora-rank 1536
    --qk-head-dim 128
    --qk-layernorm
    --qk-pos-emb-head-dim 64

    # Vocab padding for tensor-parallel friendliness (R1/V3)
    --make-vocab-size-divisible-by 1280
)

# =========================
# Tokenizer
# =========================
ALL_PARAMS+=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
)

# =========================
# Optimizer hyperparameters (unchanged)
# =========================
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

# =========================
# Dataset configuration (unchanged)
# =========================
DATASET_SIZE=50B
TRAIN_ITERS=300 # Training speed saturates until 200 steps

# =========================
# Scheduler (unchanged)
# =========================
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

# =========================
# Batch sizes (unchanged)
# =========================
ALL_PARAMS+=(
    --micro-batch-size 1
    --global-batch-size 1024
)

# =========================
# Parallelism (for 16 nodes)
# =========================
ALL_PARAMS+=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --num-layers-per-virtual-pipeline-stage 2
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

# =========================
# Implementation-related parameters
# =========================
ALL_PARAMS+=(
    --bf16
    --use-mcore-models
    --no-masked-softmax-fusion
    --use-flash-attn

    # --recompute-activations
    # --moe-layer-recompute

    --attention-softmax-in-fp32
    --transformer-impl transformer_engine
    --attention-backend ${ATTN_BACKEND}
)

# =========================
# MoE args for DeepSeek-V3 (R1-equivalent)
# =========================
# DeepSeek-V3 / R1: many small experts, top-8 routing, shared experts overlap enabled
ALL_PARAMS+=(
    --num-experts 256
    --expert-model-parallel-size 8

    # 61 layers (original)
    # --moe-layer-freq "([0]*3+[1]*58)"
    # For simplicity, we set all layer to MoE in testing.
    --moe-layer-freq "([1]*$N_LAYERS)"

    # Expert dimensions (R1/V3)
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-shared-expert-overlap

    # Router & load balancing (R1/V3)
    --moe-router-dtype fp32
    --moe-router-score-function sigmoid
    --moe-router-topk 8
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-topk-scaling-factor 2.5
    --moe-router-pre-softmax
    --moe-router-bias-update-rate 1e-3
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4

    # Token dispatch
    --moe-token-dispatcher-type alltoall
    --moe-token-drop-policy probs

    --moe-grouped-gemm
    # --moe-permute-fusion # if TE >= 2.1.0 and compatible

    # Expert capacity
    --moe-expert-capacity-factor 1.2
    --moe-pad-expert-input-to-capacity
)

# NOTE(odashi): Disable fused attention for Sakura cluster due to some inconsistency.
# export NVTE_FUSED_ATTN=0

# =========================
# Logging (unchanged)
# =========================
ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --wandb-entity llm-jp
    --wandb-project 0176_merge_megatron_upstream
    --wandb-exp-name "${RUN_NAME}"
)

