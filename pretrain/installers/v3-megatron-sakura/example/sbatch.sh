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

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-small
#SBATCH --job-name=megatron-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks-per-node=1

set -eu -o pipefail

source scripts/environment.sh
source venv/bin/activate

export MASTER_ADDR="$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=12800

echo "MASTER_ADDR=${MASTER_ADDR}"

mpirun \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NUM_NODES=1 \
  -x NUM_GPUS_PER_NODE=1 \
  -bind-to none \
  -map-by slot \
  bash example/train.sh
