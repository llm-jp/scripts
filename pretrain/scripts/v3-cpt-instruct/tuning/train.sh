#!/bin/bash
#SBATCH --job-name=0095_sft
#SBATCH --partition=FIXME
#SBATCH --nodes=0
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eu -o pipefail

if [ $# -ne 4 ]; then
  >&2 echo Usage: $0 ENV_DIR TASK_DIR MODEL_SIZE ITER
  >&2 echo Example: $0 /path/to/environment /path/to/tasks/1.8b_inst0.2 1.8b 100000
  exit 1
fi

ENV_DIR=$1; shift
TASK_DIR=$1; shift
MODEL_SIZE=$1; shift
ITER=$(printf %07d $1); shift

WORK_DIR=${TASK_DIR}/sft
mkdir -p ${WORK_DIR}

export TMPDIR=${WORK_DIR}/tmp

# module load
export MODULEPATH=/data/modules:${MODULEPATH}
module load cuda-12.1.1
module load cudnn-8.9.7
module load hpcx-2.17.1
module load nccl-2.18.3

# open file limit
ulimit -n 65536 1048576

source ${ENV_DIR}/venv/bin/activate

# distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# GPU settings
NUM_GPUS_PER_NODE=8
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "NUM_NODES=${NUM_NODES}, NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}, NUM_GPUS=${NUM_GPUS}"

# run
mpirun \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none \
  -map-by slot \
  -x PATH \
  python scripts/pretrain/scripts/v3-cpt-instruct/tuning/train_sft.py \
  work_dir=${WORK_DIR} \
  trainer.num_nodes=${NUM_NODES} \
  use_mpi=True \
  use_slurm=True \
  name=$(basename ${TASK_DIR})-iter_${ITER} \
  model=llm-jp-3-${MODEL_SIZE} \
  mbs=4 \
  model.restore_from_path=${TASK_DIR}/checkpoints_nemo/iter_${ITER}
