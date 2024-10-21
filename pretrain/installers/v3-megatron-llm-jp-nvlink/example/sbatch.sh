#!/bin/bash
#
# Example sbatch launcher script of pretraining tasks.
# 
# This script only constructs cluster-related environment variables, and immediately
# calls mpirun with train.sh, which implements an actual invocation of the Megatron-LM
# trainer script.
#
# This script is installed together with other tools so that you can check if the
# installed environment works as expected by launching the job using this script.
#
# Usage:
# 1. cd {root directory that you installed training scripts}
# 2. sbatch example/sbatch.sh

#SBATCH --job-name=pretrain-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eu -o pipefail

source scripts/environment.sh
source venv/bin/activate

export MASTER_ADDR="$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=12800

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
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NUM_NODES=$NUM_NODES \
  -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
  bash example/train.sh

