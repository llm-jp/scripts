#!/bin/bash
#PBS -P gcg51557
#PBS -q R10415
#PBS -N 0130_test
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -m n

cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/${HOST_NAME}-${JOBID}.out
ERRFILE=${TASK_DIR}/logs/${HOST_NAME}-${JOBID}.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

EXPERIMENT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining
SCRIPT_DIR=/groups/gcg51557/experiments/0130_instruction_pretraining/scripts/pretrain/scripts/v3-instruct-pretrain-abci/pretrain
ENV_DIR=${EXPERIMENT_DIR}/environments/pretrain_torch_v2.5.1

# Setup environment
source ${SCRIPT_DIR}/common/setup_torch_v2.5.1.sh

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))
echo "hostname: ${MASTER_ADDR}"

NUM_NODES=$(wc -l < $PBS_NODEFILE)
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "nnodes: ${NUM_NODES}; ngpus: ${NUM_GPUS}"
echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

cat $PBS_NODEFILE

# Load ALL_PARAMS
source ${SCRIPT_DIR}/params/${PARAM_NAME}.sh
echo "ALL_PARAMS: ${ALL_PARAMS[@]}"

mpirun \
  --display-allocation \
  --report-bindings \
  --oversubscribe \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -map-by slot \
  -x PATH \
  python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
    ${ALL_PARAMS[@]}

