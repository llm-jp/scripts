#!/bin/bash
# List of environment variables and module loads for pretrain tasks on SYNKA
#
# SYNKA hardware: gpusv[01-50], NVIDIA H200 x8 (Hopper architecture)
# SYNKA CUDA toolkit: /home/apps/cuda/12.6 (via module load cuda/12.6)
# Using CUDA 12.6 to match PyTorch cu126 wheel exactly (avoids nvcc version mismatch)

export PRETRAIN_CUDA_VERSION_MAJOR=12
export PRETRAIN_CUDA_VERSION_MINOR=6
export PRETRAIN_CUDA_VERSION_PATCH=0

export PRETRAIN_CUDA_VERSION=${PRETRAIN_CUDA_VERSION_MAJOR}.${PRETRAIN_CUDA_VERSION_MINOR}
export PRETRAIN_CUDA_VERSION_SHORT=${PRETRAIN_CUDA_VERSION_MAJOR}${PRETRAIN_CUDA_VERSION_MINOR}

# SYNKA module versions
export PRETRAIN_NCCL_VERSION=2.25.1-cuda12
export PRETRAIN_HPCX_VERSION=2.18.1-gcc-cuda12/hpcx

export PRETRAIN_SYSTEM_PYTHON=/usr/bin/python3.10

export PRETRAIN_TORCH_VERSION=2.6.0

export PRETRAIN_APEX_COMMIT=312acb44f9fe05cab8c67bba6daa0e64d3737863
# FA3 commit (hopper/ subdirectory); H200 = Hopper (sm_90), same as H100
export PRETRAIN_FLASH_ATTENTION_VERSION=27f501d
export PRETRAIN_TRANSFORMER_ENGINE_VERSION=2.3.0

export PRETRAIN_MEGATRON_TAG=main
export PRETRAIN_TOKENIZER_COMMIT=925b87aa2b55cebcb66a36a80431e28c533f70b5

# HTTP Proxy (required for all external network access on SYNKA)
export HTTP_PROXY=http://172.22.116.58:8080
export HTTPS_PROXY=http://172.22.116.58:8080
export http_proxy=http://172.22.116.58:8080
export https_proxy=http://172.22.116.58:8080

# Load SYNKA modules
source /etc/profile.d/modules.sh
module purge
module load cuda/${PRETRAIN_CUDA_VERSION}
module load nccl/${PRETRAIN_NCCL_VERSION}
module load hpcx/${PRETRAIN_HPCX_VERSION}

# CUDA_HOME is set by module load cuda/12.6 to /home/apps/cuda/12.6
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}
