#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0176_merge_megatron_upstream
#PBS -l walltime=10000:00:00
#PBS -m n

cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/train-${JOBID}.out
ERRFILE=${TASK_DIR}/logs/train-${JOBID}.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

EXPERIMENT_DIR=/home/ach17726fj/experiments/0176_megatron_upstream_merge/
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts/pretrain/scripts/v5-test/train
# Takes $ENV_DIR from the environment variable
# ENV_DIR=${EXPERIMENT_DIR}/environments
# ENV_DIR=${EXPERIMENT_DIR}/environment2
# ENV_DIR=${EXPERIMENT_DIR}/test_environment

# Setup environment
source ${SCRIPT_DIR}/common/setup.sh

source ${ENV_DIR}/venv/bin/activate

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

# Load TRAIN_DATA_PATH
source ${TASK_DIR}/train_data_50B.sh
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"

# Load ALL_PARAMS
source ${SCRIPT_DIR}/params/${PARAM_NAME}.sh
echo "ALL_PARAMS: ${ALL_PARAMS[@]}"

# export NVTE_FUSED_ATTN=0

mpirun \
  --display-allocation \
  --report-bindings \
  --oversubscribe \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -bind-to none \
  -map-by slot \
  python ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
    ${ALL_PARAMS[@]}
