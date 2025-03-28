# Script for setup trainer environment.

source /etc/profile.d/modules.sh
module load cuda/12.4/12.4.1
module load cudnn/9.5/9.5.1
module load hpcx/2.20
module load nccl/2.25/2.25.1-1
echo $(module list)

ENV_DIR=${EXPERIMENT_DIR}/environments/pretrain_torch_v2.6.0

source ${ENV_DIR}/venv/bin/activate

## Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1
