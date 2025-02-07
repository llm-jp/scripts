#!/bin/bash
#SBATCH --job-name=0111_train
#SBATCH --partition=FIXME
#SBATCH --nodes=0
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# LLM-jp v4 7.7B launcher script.
# Usage: sbatch --partition=FIXME --nodes=FIXME --outpout=FIXME sbatch.sh ENV_DIR MODEL_DIR

set -eu -o pipefail

# Arguments
if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 ENV_DIR MODEL_DIR"
    exit 1
fi
ENV_DIR=$(realpath $1); shift
MODEL_DIR=$(realpath $1); shift
echo "ENV_DIR=${ENV_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"

# Find the script directory
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(
        scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/Command=/{print $2}' \
        | cut -d ' ' -f 1
    )
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
echo "SCRIPT_DIR=${SCRIPT_DIR}"

# Determine master address:port
MASTER_ADDR=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n1)
MASTER_PORT=$((10000 + (${SLURM_JOBID} % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Determine amount of employed devices
NUM_NODES=${SLURM_JOB_NUM_NODES}
NUM_GPUS_PER_NODE=$(echo ${SLURM_TASKS_PER_NODE} | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"

# Preparation for mpirun (Python)
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

# Run the trainer script
mpirun \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    -x MASTER_ADDR=${MASTER_ADDR} \
    -x MASTER_PORT=${MASTER_PORT} \
    -x NUM_NODES=${NUM_NODES} \
    -x NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} \
    -x ENV_DIR=${ENV_DIR} \
    -x MODEL_DIR=${MODEL_DIR} \
    -x SCRIPT_DIR=${SCRIPT_DIR} \
    bash ${SCRIPT_DIR}/train.sh
