#!/bin/bash
# Submit LLM-jp-4 291B-A27B on Sakura/B200.
#
# Defaults reproduce the validated fast config (36 nodes, 288 GPUs):
#   TP1/PP6/CP1/VPP2/EP8, mbs=2, gbs=2880, recompute=moe, comm-overlap OFF,
#   account-for-embedding/loss pipeline split, save-interval=100.
# A bare `bash submit_291b_a27b_36n.sh <ENV_DIR> <MODEL_DIR>` runs this as-is;
# override any knob via the environment variables listed below.

set -eu -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)

ENV_DIR=${ENV_DIR:-${1:-}}
MODEL_DIR=${MODEL_DIR:-${2:-}}
if [ -z "${ENV_DIR}" ] || [ -z "${MODEL_DIR}" ]; then
    >&2 cat <<'USAGE'
Usage:
  bash submit_291b_a27b_36n.sh <ENV_DIR> <MODEL_DIR>

Environment overrides (defaults in parens):
  NODES=36
  TP=1  PP=6  CP=1  VPP=2  EP=8  ACCOUNT_EMB=1
  MICRO_BATCH_SIZE=2  GLOBAL_BATCH_SIZE=2880
  COMM_OVERLAP=1           # overlap grad-reduce/param-gather (ON; set 0 if a config deadlocks)
  SAVE_INTERVAL=100
  NUM_DATASET_BUILDER_THREADS=16
  DISTRIBUTED_TIMEOUT_MINUTES=1440
  WANDB_ENTITY=llm-jp
  WANDB_PROJECT=0279_wsd_test
  DATA_CACHE_DIR=<shared cache path>
  TIME_LIMIT=UNLIMITED
  EXTRA_PARAMS="..."
USAGE
    exit 2
fi

export NODES=${NODES:-36}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TP=${TP:-1}
export PP=${PP:-6}
export EP=${EP:-8}
export CP=${CP:-1}
export VPP=${VPP:-2}
export ACCOUNT_EMB=${ACCOUNT_EMB:-1}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2880}
export SAVE_INTERVAL=${SAVE_INTERVAL:-100}
export NUM_DATASET_BUILDER_THREADS=${NUM_DATASET_BUILDER_THREADS:-16}
export DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-1440}
export DATA_CACHE_DIR=${DATA_CACHE_DIR:-${MODEL_DIR}/data_cache}
export WANDB_ENTITY=${WANDB_ENTITY:-llm-jp}
export WANDB_PROJECT=${WANDB_PROJECT:-0279_wsd_test}
export WANDB_EXP_NAME=${WANDB_EXP_NAME:-llm-jp-4-291b-a27b-pp${PP}-cp${CP}-vpp${VPP}-ep${EP}-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}}

VARIANT=${VARIANT:-base}
JOB_NAME=${JOB_NAME:-v4_291b_pp${PP}cp${CP}vpp${VPP}mbs${MICRO_BATCH_SIZE}}
TIME_LIMIT=${TIME_LIMIT:-UNLIMITED}
PARTITION=${PARTITION:-gpu}

WORLD_SIZE=$((NODES * GPUS_PER_NODE))
MODEL_PAR=$((TP * PP * CP))
if [ $((WORLD_SIZE % MODEL_PAR)) -ne 0 ]; then
    >&2 echo "Invalid parallelism: WORLD_SIZE=${WORLD_SIZE} is not divisible by TP*PP*CP=${MODEL_PAR}"
    exit 1
fi

DP=$((WORLD_SIZE / MODEL_PAR))
if [ $((DP % EP)) -ne 0 ]; then
    >&2 echo "Invalid parallelism: DP=${DP} (=world/(TP*PP*CP)) is not divisible by EP=${EP}"
    exit 1
fi

GBS_UNIT=$((MICRO_BATCH_SIZE * DP))
if [ $((GLOBAL_BATCH_SIZE % GBS_UNIT)) -ne 0 ]; then
    >&2 echo "Invalid batch: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by MICRO_BATCH_SIZE*DP=${GBS_UNIT}"
    exit 1
fi
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / GBS_UNIT))
EXPERT_DP=$((DP / EP))

mkdir -p "${SCRIPT_DIR}/logs" "${DATA_CACHE_DIR}"
cd "${SCRIPT_DIR}"

echo "Submitting ${JOB_NAME}"
echo "  ENV_DIR=${ENV_DIR}"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  nodes=${NODES} world=${WORLD_SIZE} tp=${TP} pp=${PP} cp=${CP} vpp=${VPP} dp=${DP} ep=${EP} expert_dp=${EXPERT_DP} account_emb=${ACCOUNT_EMB}"
echo "  mbs=${MICRO_BATCH_SIZE} gbs=${GLOBAL_BATCH_SIZE} microbatches=${NUM_MICROBATCHES}"
echo "  data_cache=${DATA_CACHE_DIR}"
echo "  wandb=${WANDB_ENTITY}/${WANDB_PROJECT}:${WANDB_EXP_NAME}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: validation passed; not submitting."
    exit 0
fi

# shellcheck disable=SC2086
JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --nodes="${NODES}" \
    --job-name="${JOB_NAME}" \
    --time="${TIME_LIMIT}" \
    --export=ALL \
    ${SBATCH_EXTRA_ARGS:-} \
    "${SCRIPT_DIR}/sbatch_train.sh" \
    "${ENV_DIR}" "${MODEL_DIR}" "${VARIANT}" \
    "${WANDB_ENTITY}" "${WANDB_PROJECT}")

echo "Submitted ${JOB_ID}"
