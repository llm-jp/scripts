#!/bin/bash
#SBATCH --job-name=0031_train
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

# PLEASE run this script from the root of the experiment directory.


set -eu -o pipefail

if [ $# -ne 9 ]; then
    >&2 echo "Usage $0 ENABLED FORMAT MARGIN INTERVAL AMAX_HIST_LEN AMAX_ALGO WGRAD ITER STOP"
    exit 1
fi

FP8_ENABLED=$1; shift
FP8_FORMAT=$1; shift
FP8_MARGIN=$1; shift
FP8_INTERVAL=$1; shift
FP8_AMAX_HISTORY_LEN=$1; shift
FP8_AMAX_COMPUTE_ALGO=$1; shift
FP8_WGRAD=$1; shift
LOAD_ITER=$1; shift
FORCE_STOP_ITER=$1; shift

EXPERIMENT_DIR=$(pwd)
ENV_DIR=${EXPERIMENT_DIR}/environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

mpirun \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -bind-to none \
  -map-by slot \
  -x EXPERIMENT_DIR=$EXPERIMENT_DIR \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NUM_NODES=$NUM_NODES \
  -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
  \
  -x FP8_ENABLED=$FP8_ENABLED \
  -x FP8_FORMAT=$FP8_FORMAT \
  -x FP8_MARGIN=$FP8_MARGIN \
  -x FP8_INTERVAL=$FP8_INTERVAL \
  -x FP8_AMAX_HISTORY_LEN=$FP8_AMAX_HISTORY_LEN \
  -x FP8_AMAX_COMPUTE_ALGO=$FP8_AMAX_COMPUTE_ALGO \
  -x FP8_WGRAD=$FP8_WGRAD \
  -x LOAD_ITER=$LOAD_ITER \
  -x FORCE_STOP_ITER=${FORCE_STOP_ITER} \
  \
  bash scripts/pretrain/scripts/fp8-behavior-check/train_3.8b.sh
