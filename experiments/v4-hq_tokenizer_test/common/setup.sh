# Script for setup trainer environment.

module load cuda/12.8/12.8.1
module load cudnn/9.5/9.5.1
module load hpcx/2.20
module load nccl/2.25/2.25.1-1
# (cliu) Only for cuda/12.8; there is no folder for cuda/12.8 in cudnn/9.5.1
export CUDNN_HOME=/apps/cudnn/9.5.1/cuda12.0
export CUDNN_PATH=$CUDNN_HOME
export LD_LIBRARY_PATH=/apps/cudnn/9.5.1/cuda12.0/lib:$LD_LIBRARY_PATH
export CPATH=/apps/cudnn/9.5.1/cuda12.0/include:$CPATH
export LIBRARY_PATH=/apps/cudnn/9.5.1/cuda12.0/lib:$LIBRARY_PATH
echo $(module list)

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

export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2