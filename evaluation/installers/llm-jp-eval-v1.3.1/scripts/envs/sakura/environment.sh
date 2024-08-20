export CUDA_VERSION_MAJOR=12
export CUDA_VERSION_MINOR=1
export CUDA_VERSION=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
export CUDNN_VERSION=8.9.4
module load cuda/${CUDA_VERSION}
module load /data/cudnn-tmp-install/modulefiles/${CUDNN_VERSION}
