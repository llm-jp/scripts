#!/usr/bin/env bash
# Smoke test: train Qwen3-30B-A3B (MoE) for a few iters on mock data, 1 node
# (8x B200), to confirm the env trains end-to-end. Arch flags mirror
# examples/rl/model_configs/qwen3_30b_a3b_moe.sh (HF tokenizer from cache).
#
# Run on a gpu node:
#   srun -p gpu --nodes=1 --ntasks=1 --gres=gpu:8 -c 144 -t 00:30:00 \
#     bash verify/smoke_qwen3_30b_a3b.sh
set -euo pipefail

ENV="${ENV:-$HOME/envs/megatron-lm-b200}"
PY="${ENV}/venv/bin/python"
MLM="${ENV}/src/Megatron-LM"

# Runtime env: cu13 wheels on the loader path; CUDA_HOME -> venv nvidia dir.
export NVTE_FRAMEWORK=pytorch
export HF_HUB_OFFLINE=1     # tokenizer is already in ~/.cache/huggingface
LD_EXTRA=$("${PY}" - <<'PY'
import os, glob, torch
sp = os.path.dirname(os.path.dirname(torch.__file__))
d = glob.glob(os.path.join(sp, "nvidia", "*", "lib")); d.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
print(":".join(d))
PY
)
export LD_LIBRARY_PATH="${LD_EXTRA}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${ENV}/venv/lib/python3.12/site-packages/nvidia"
export CUDA_PATH="${CUDA_HOME}"
# Blackwell + distributed optimizer: do NOT set CUDA_DEVICE_MAX_CONNECTIONS=1.

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL_ARGS="
  --num-layers 48
  --hidden-size 2048
  --ffn-hidden-size 6144
  --num-attention-heads 32
  --kv-channels 128
  --group-query-attention --num-query-groups 4
  --seq-length 4096
  --max-position-embeddings 8192
  --normalization RMSNorm --norm-epsilon 1e-6
  --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
  --swiglu --disable-bias-linear --untie-embeddings-and-output-weights
  --qk-layernorm
  --attention-dropout 0.0 --hidden-dropout 0.0
  --no-masked-softmax-fusion --attention-softmax-in-fp32
  --vocab-size 151936 --make-vocab-size-divisible-by 128
"
MOE_ARGS="
  --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.001
  --moe-token-dispatcher-type alltoall --moe-layer-freq 1
"
PARALLEL_ARGS="
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 8
  --use-distributed-optimizer
  --transformer-impl transformer_engine
  --attention-backend flash
"
TRAIN_ARGS="
  --bf16
  --micro-batch-size 1 --global-batch-size 8
  --train-iters 5 --log-interval 1
  --eval-interval 1000 --eval-iters 0
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8
  --lr 1e-5 --min-lr 1e-6 --lr-warmup-samples 0 --clip-grad 1.0 --weight-decay 0.01
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model Qwen/Qwen3-30B-A3B
  --mock-data
  --distributed-backend nccl
  --no-load-optim --no-load-rng
"

echo "=== smoke host=$(hostname) $(date) | $(nvidia-smi -L | wc -l) GPUs ==="
cd "${MLM}"
"${PY}" -m torch.distributed.run \
  --nnodes=1 --nproc-per-node="${GPUS_PER_NODE}" --master-port="${MASTER_PORT:-29501}" \
  pretrain_gpt.py ${MODEL_ARGS} ${MOE_ARGS} ${PARALLEL_ARGS} ${TRAIN_ARGS}
echo "=== smoke exit=$? $(date) ==="
