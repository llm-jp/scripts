#!/bin/bash
#SBATCH --job-name=0095_train
#SBATCH --partition=FIXME
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

if [ $# -ne 2 ] ; then
    >&2 echo Usage: "$0 <TRAIN_SCRIPT> <TASK_DIR>"
    exit 1
fi

TRAIN_SCRIPT=$1; shift
TASK_DIR=$1; shift

EXPERIMENT_DIR=/home/shared/experiments/0095_instruction_pretraining
ENV_DIR=${EXPERIMENT_DIR}/environments/train

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo MASTER_ADDR=${MASTER_ADDR}

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
  -x TASK_DIR=$TASK_DIR \
  bash ${TRAIN_SCRIPT}
