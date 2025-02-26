#!/bin/bash
#PBS -P gcg51557
#PBS -q R10415
#PBS -v RTYPE=rt_HF
#PBS -N all_reduce
#PBS -l walltime=1:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

cd $PBS_O_WORKDIR

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/all_reduce-${MASTER_ADDR}-${JOBID}.out
ERRFILE=./logs/all_reduce-${MASTER_ADDR}-${JOBID}.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

echo "hostname: ${MASTER_ADDR}"

EXPERIMENT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts/pretrain/scripts/v3-instruct-pretrain-abci/pretrain
ENV_DIR=${EXPERIMENT_DIR}/environments/pretrain-test

# Setup environment
source ${SCRIPT_DIR}/common/setup.sh

NUM_NODES=$(wc -l < $PBS_NODEFILE)
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "nnodes: ${NUM_NODES}; ngpus: ${NUM_GPUS}"
echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

cat $PBS_NODEFILE

mpirun \
  --display-allocation \
  --report-bindings \
  --oversubscribe \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -bind-to none \
  -map-by slot \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x NUM_NODES=$NUM_NODES \
  -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
  python scripts/server/gpu/all_reduce_bench/all_reduce_bench.py
