#!/bin/bash
# List of environment variables and module loads for pretrain tasks

export PRETRAIN_CUDA_VERSION_MAJOR=11
export PRETRAIN_CUDA_VERSION_MINOR=8
export PRETRAIN_CUDA_VERSION=${PRETRAIN_CUDA_VERSION_MAJOR}.${PRETRAIN_CUDA_VERSION_MINOR}
export PRETRAIN_CUDA_VERSION_SHORT=${PRETRAIN_CUDA_VERSION_MAJOR}${PRETRAIN_CUDA_VERSION_MINOR}

export PRETRAIN_PYTHON_VERSION=3.10.14
export PRETRAIN_TORCH_VERSION=2.3.1
export PRETRAIN_TORCHVISION_VERSION=0.18.1
export PRETRAIN_APEX_VERSION=24.04.01
export PRETRAIN_TRANSFORMER_ENGINE_VERSION=1.4
export PRETRAIN_MEGATRON_TAG=nii-geniac
# Ensure the appropriate Huggingface tokenizer is included
# https://github.com/llm-jp/scripts/pull/12#discussion_r1708415209
export PRETRAIN_TOKENIZER_TAG=v3.0b2

source /etc/profile.d/openmpi.sh
export CUDACXX="/usr/local/cuda-${PRETRAIN_CUDA_VERSION}/bin/nvcc"
