#!/bin/bash
#
# Example launcher script of pretraining tasks.
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
# 2. bash example/mpi_wrapper.sh

set -eu -o pipefail

source scripts/environment.sh
source venv/bin/activate

export MASTER_ADDR="$(ip -br addr | sed -n 2p | awk '{print $3}' | cut -d'/' -f1)"
export MASTER_PORT=12800

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=1
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
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

