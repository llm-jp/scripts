#!/bin/bash
# Node-level training batch replay launcher.
#
# Environment variables:
# * ENV_DIR: Directory that contains scripts/environment.sh, venv, and src/Megatron-LM.

set -eu -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ENV_DIR=${ENV_DIR:-$(pwd)}

source "${ENV_DIR}/scripts/environment.sh"
source "${ENV_DIR}/venv/bin/activate"

export LOGLEVEL=${LOGLEVEL:-INFO}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}

# Training data range to replay. Defaults to the whole training loop.
ITER_INDEX=${ITER_INDEX:-}
ITER_START=${ITER_START:-}
ITER_END=${ITER_END:-}

# Training config from environment/example/train.sh.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-1000}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-5000}
TRAIN_STEPS=${TRAIN_STEPS:-$((${LR_WARMUP_STEPS} + ${LR_DECAY_ITERS}))}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
SEED=${SEED:-1234}

# Data config.
TOKENIZER_MODEL=${TOKENIZER_MODEL:-src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model}
DATA_PATH=${DATA_PATH:-"2563804308 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/wiki_0000.jsonl_text_document 1826105478 /data/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/train/ja/kaken_0000.jsonl_text_document"}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-cache}
OUTPUT=${OUTPUT:-outputs/global-batches.jsonl.gz}
MEGATRON_PATH=${MEGATRON_PATH:-src/Megatron-LM}

mkdir -p "$(dirname "${OUTPUT}")"

ITER_RANGE_ARGS=()
if [[ -n "${ITER_INDEX}" ]]; then
  ITER_RANGE_ARGS=(--iter-index "${ITER_INDEX}")
else
  if [[ -n "${ITER_START}" ]]; then
    ITER_RANGE_ARGS+=(--iter-start "${ITER_START}")
  fi
  if [[ -n "${ITER_END}" ]]; then
    ITER_RANGE_ARGS+=(--iter-end "${ITER_END}")
  fi
fi

EXTRA_REPLAY_ARGS=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_REPLAY_ARGS=(${EXTRA_ARGS})
fi

# shellcheck disable=SC2206
DATA_PATH_ARGS=(${DATA_PATH})

python "${SCRIPT_DIR}/replay_training_batch.py" \
  --megatron-path "${MEGATRON_PATH}" \
  "${ITER_RANGE_ARGS[@]}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --train-iters "${TRAIN_STEPS}" \
  --seq-length "${SEQ_LENGTH}" \
  --seed "${SEED}" \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --make-vocab-size-divisible-by 128 \
  --data-path "${DATA_PATH_ARGS[@]}" \
  --data-cache-path "${DATA_CACHE_PATH}" \
  --split 1,0,0 \
  --output "${OUTPUT}" \
  "${EXTRA_REPLAY_ARGS[@]}"

echo "Wrote replayed batch to ${OUTPUT}"
