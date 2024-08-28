#!/bin/bash

set -eu -o pipefail

ENV_DIR=environment

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/scripts/mpi_variables.sh
source ${ENV_DIR}/venv/bin/activate

#export NCCL_P2P_LEVEL=NVL
#export NCCL_P2P_DISABLE=1
#export NCCL_NET_GDR_LEVEL=SYS
#export NCCL_NET_GDR_READ=1

# env > p2p_${OMPI_COMM_WORLD_RANK}.txt

export NCCL_DEBUG=INFO
#export NCCL_DEBUG=TRACE

python ./scripts/server/gpu/p2p_test/p2p.py $1 $2
# 2>&1 | tee p2p_${OMPI_COMM_WORLD_RANK}.txt