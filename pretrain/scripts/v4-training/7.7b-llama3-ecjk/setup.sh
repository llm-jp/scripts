# Script for setup trainer environment.

# Determine master address:port
export MASTER_ADDR=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n1)
export MASTER_PORT=$((10000 + (${SLURM_JOBID} % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Determine amount of employed devices
NUM_NODES=${SLURM_JOB_NUM_NODES}
NUM_GPUS_PER_NODE=$(echo ${SLURM_TASKS_PER_NODE} | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"

# Setup Python environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/scripts/mpi_variables.sh
source ${ENV_DIR}/venv/bin/activate

# open file limit
ulimit -n 65536 1048576

# Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1