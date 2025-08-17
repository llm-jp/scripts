# Script for setup trainer environment.

source /etc/profile.d/modules.sh
# echo $(module list)
source ${ENV_DIR}/scripts/environment.sh

loaded=$(module -t list 2>&1)
echo "-----"
echo "Modules: $loaded"
echo "-----"


source ${ENV_DIR}/venv/bin/activate

## Debug/logging flags
export LOGLEVEL=INFO
# export NCCL_DEBUG=WARN
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

# For debugging
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
