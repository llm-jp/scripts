#!/bin/bash
#SBATCH --job-name=p2p-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out
#SBATCH --partition=example

set -eu -o pipefail

ENV_DIR=environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

# 直接起動するときは以下の3行を有効にする
#export SLURM_JOB_NODELIST=b[034-035]
#export SLURM_JOB_NUM_NODES=2
#export SLURM_TASKS_PER_NODE=1

export MASTER_ADDR="$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)"
export MASTER_PORT=12801

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

# mpitx \
# 直接起動するときは -machinefile machines を追加
#    -machinefile machines \
mpirun \
    -np $NUM_GPUS \
    --npernode $NUM_GPUS_PER_NODE \
    -bind-to none \
    -map-by slot \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NUM_NODES=$NUM_NODES \
    -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
    -- ./scripts/server/gpu/p2p_test/p2p.sh $1 $2