#!/bin/bash
#SBATCH --job-name=v4-moe-32b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#
# LLM-jp v4 MoE 291B-A27B trainer for sakura/B200 (torchrun, no MPI).
# Port of ABCI 0279. A VARIANT directory holds the config pair
# (<VARIANT>/params.sh and <VARIANT>/train_data.sh); "base" is the default.
# Launch:
#   sbatch --nodes=<N> --job-name=0279_train sbatch_train.sh \
#     <ENV_DIR> <MODEL_DIR> base <WANDB_ENTITY> <WANDB_PROJECT>
set -eu -o pipefail

if [ $# -ne 5 ]; then
    >&2 echo "Usage: $0 ENV_DIR MODEL_DIR VARIANT WANDB_ENTITY WANDB_PROJECT"
    exit 1
fi
export ENV_DIR=$(realpath -eP "$1"); shift
export MODEL_DIR=$(realpath -m "$1"); shift
VARIANT=$1; shift
WANDB_ENTITY=$1; shift
WANDB_PROJECT=$1; shift

# Resolve script dir (works under sbatch and plain bash/srun).
if [ -n "${SLURM_JOB_ID:-}" ] && scontrol show job "${SLURM_JOB_ID}" >/dev/null 2>&1; then
    SCRIPT_PATH=$(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}' | cut -d ' ' -f1)
    [ -f "${SCRIPT_PATH}" ] || SCRIPT_PATH=$(realpath "$0")
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
mkdir -p "${SCRIPT_DIR}/logs"
echo "ENV_DIR=${ENV_DIR} MODEL_DIR=${MODEL_DIR} VARIANT=${VARIANT}"

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
source "${SCRIPT_DIR}/setup.sh"
source "${SCRIPT_DIR}/${VARIANT}/params.sh"
source "${SCRIPT_DIR}/${VARIANT}/train_data.sh"

# W&B: enabled whenever an entity other than "-" is given.
# NOTE: in this mcore, the scalar-logging block in training_log is gated by
# `if writer (== tensorboard) ...`, and the wandb_writer.log() calls live inside
# it. So a TensorBoard writer MUST exist for wandb to receive loss/grad-norm/etc.
# -> always pair --wandb-* with --tensorboard-dir.
if [ "${WANDB_ENTITY}" != "-" ]; then
    ALL_PARAMS+=(
        --wandb-entity "${WANDB_ENTITY}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_EXP_NAME:-train_$(date '+%Y%m%d-%H%M%S')_${SLURM_JOB_ID:-local}}"
        --tensorboard-dir "${MODEL_DIR}/tensorboard"
    )
fi

# Checkpointing: on unless NO_SAVE=1.
if [ "${NO_SAVE:-0}" != "1" ]; then
    CKPT="${MODEL_DIR}/checkpoints"
    ALL_PARAMS+=( --load "${CKPT}" --save "${CKPT}" --save-interval "${SAVE_INTERVAL:-1000}" )
fi

# Extra ad-hoc flags for variant runs (e.g. EXTRA_PARAMS="--moe-router-fusion ...").
ALL_PARAMS+=( ${EXTRA_PARAMS:-} )

LAUNCH="${ENV_DIR}/venv/bin/python -m torch.distributed.run \
  --nnodes=${SLURM_JOB_NUM_NODES:-1} --nproc-per-node=${GPUS_PER_NODE} \
  --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT}"

# One srun task per node; torchrun fans out the local ranks. NODE_RANK=SLURM_NODEID.
srun --ntasks="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 bash -c "
  ${LAUNCH} --node-rank=\${SLURM_NODEID:-0} \
    ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
    ${ALL_PARAMS[*]} ${TRAIN_DATA_PATH[*]}
"
echo "Done"
