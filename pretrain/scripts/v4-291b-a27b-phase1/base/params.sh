# Pretraining hyperparameters for LLM-jp v4 MoE 291B-A27B.
# Based on the 332B-A31B shape, reduced to 70 transformer layers.
# ALL_PARAMS holds everything except --data-path / --data-cache-path (train_data/),
# --wandb-* and --load/--save (sbatch_train.sh), and the launcher.

ALL_PARAMS=()

# Model hyperparameters
ALL_PARAMS+=(
    --num-layers 70
    --hidden-size 5120
    --ffn-hidden-size 2048
    --num-attention-heads 80
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 500000
    --untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --qk-layernorm
)

# MoE
ALL_PARAMS+=(
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 2048
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-router-dtype fp32
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
)

# Tokenizer (LLM-jp v4 alpha 1.0; vocab 196608)
ALL_PARAMS+=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${ENV_DIR}/src/llm-jp-tokenizer-v4/v4_alpha_1.0.model
)

# Optimizer. Megatron's adam path is AdamW by default via decoupled weight decay.
ALL_PARAMS+=(
    --optimizer adam
    --lr 4e-4
    --min-lr 4e-5
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# Scheduler (WSD). Hardcoded here (the source of truth); override via EXTRA_PARAMS.
ALL_PARAMS+=(
    --train-iters 12000000
    --lr-warmup-iters 2000
    --lr-decay-iters 12000000
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-wsd-decay-iters 1
    --eval-interval 999999999
    --eval-iters 0
)

# Batch sizes. DP = world / (TP*PP*CP). Default TP1/PP6/CP1 -> DP=48:
#   mbs=2, gbs=2880 -> 30 microbatches
# (Old TP1/PP4 default gave DP=72: mbs=2 -> 20, mbs=4 -> 10.)
ALL_PARAMS+=(
    --micro-batch-size ${MICRO_BATCH_SIZE:-4}
    --global-batch-size ${GLOBAL_BATCH_SIZE:-2880}
)

# Parallelism. 70 layers; DP = world / (TP*PP*CP), and EP must divide DP.
#   PP=4 (default): explicit 17/18/18/17 decoder split.
#   PP=6 + VPP=2 + ACCOUNT_EMB=1: embedding+loss each count as a layer
#     (70+2=72), giving an even 72/(PP*VPP)=6 layers per virtual stage.
ALL_PARAMS+=(
    --tensor-model-parallel-size ${TP:-1}
    --pipeline-model-parallel-size ${PP:-4}
    --expert-model-parallel-size ${EP:-8}
    --expert-tensor-parallel-size 1
    --context-parallel-size ${CP:-1}
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES:-120}
    --num-dataset-builder-threads ${NUM_DATASET_BUILDER_THREADS:-1}
)

# Pipeline layer split: account-for-embedding/loss (even PP*VPP division) vs the
# explicit PP=4 first/last split.
if [ "${ACCOUNT_EMB:-0}" = "1" ]; then
    ALL_PARAMS+=(
        --account-for-embedding-in-pipeline-split
        --account-for-loss-in-pipeline-split
    )
else
    ALL_PARAMS+=(
        --decoder-first-pipeline-num-layers 17
        --decoder-last-pipeline-num-layers 17
    )
fi

# Virtual pipeline parallelism (interleaved 1F1B) when VPP>1.
if [ "${VPP:-1}" -gt 1 ]; then
    ALL_PARAMS+=( --num-virtual-stages-per-pipeline-rank ${VPP} )
fi

# Distributed-optimizer communication overlap. Set COMM_OVERLAP=0 to disable
# (e.g. to rule out an overlap-related NCCL hang at iteration 2).
if [ "${COMM_OVERLAP:-0}" = "1" ]; then
    ALL_PARAMS+=(
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi

# Recompute for initial memory headroom.
ALL_PARAMS+=(
    --recompute-granularity selective
    --recompute-modules moe
)

# Disable the rerun/result-validation engine (determinism reruns); not needed
# for training and avoids re-running steps. Override via EXTRA_PARAMS if wanted.
ALL_PARAMS+=( --rerun-mode disabled )

# Precision / kernels / attention backend
ALL_PARAMS+=(
    --bf16
    --use-mcore-models
    --transformer-impl transformer_engine
    --attention-backend flash
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --cross-entropy-fusion-impl native
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 100
)

# Checkpoint I/O format
ALL_PARAMS+=(
    --async-save
    --ckpt-format torch_dist
)

if [ "${USE_PERSISTENT_CKPT_WORKER:-1}" = "1" ]; then
    ALL_PARAMS+=( --use-persistent-ckpt-worker )
fi

# Logging
ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --moe-per-layer-logging
)
