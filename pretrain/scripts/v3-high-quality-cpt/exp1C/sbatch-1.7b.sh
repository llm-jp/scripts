#!/bin/bash
#SBATCH --job-name=0022_1.7b-high-qaulity-cpt-exp1c
#SBATCH --partition=gpu-small
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

# change directory if each experiment will be handled as one experintal issue
EXPERIMENT_DIR=/home/shared/experiments/0022_v3-high-quality-cpt
CONF_DIR=exp1C

ENV_DIR=${EXPERIMENT_DIR}/environment
SCRIPT_ROOT=${EXPERIMENT_DIR}/scripts/pretrain/scripts/v3-high-quality-cpt

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + (SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

mpirun \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -bind-to none \
  -map-by slot \
  -x EXPERIMENT_DIR=$EXPERIMENT_DIR \
  -x SCRIPT_ROOT=$SCRIPT_ROOT \
  -x CONF_DIR=$CONF_DIR \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  bash ${SCRIPT_ROOT}/train-1.7b.sh
