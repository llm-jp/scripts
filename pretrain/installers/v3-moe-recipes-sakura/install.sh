#!/bin/bash
#
# moe-recipes installation script for pretrain jobs on the Sakura cluster
#
# Usage:
# 1. Set the working directory to the directory this file is located.
# 2. Run `sbatch install.sh TARGET_DIR` with setting TARGET_DIR to the actual path.
#
# This script consumes 1 node on the `cpu` partition on the cluster.
#
# CAUTION:
# DO NOT change the content of this file and any other materials in the installer
# directory while the installation is being processed.

#SBATCH --job-name=pretrain-install
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eux -o pipefail

if [ $# -ne 1 ]; then
  >&2 echo Usage: sbatch install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$1; shift

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR

mkdir ${TARGET_DIR}
pushd ${TARGET_DIR}

# copy basic scripts
cp -a ${INSTALLER_DIR}/{install.sh,requirements.txt,scripts,example} .

source scripts/environment.sh

# record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir src
pushd src

# install Python
git clone https://github.com/python/cpython -b v${PRETRAIN_PYTHON_VERSION}
pushd cpython
./configure --prefix="${TARGET_DIR}/python" --enable-optimizations
make -j 64
make install
popd

popd  # src

# prepare venv
python/bin/python3 -m venv venv
source venv/bin/activate
python -m pip install --no-cache-dir -U pip

# install PyTorch
python -m pip install \
  --no-cache-dir \
  --find-links https://download.pytorch.org/whl/torch_stable.html \
  torch==${PRETRAIN_TORCH_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT} \
  torchvision==${PRETRAIN_TORCHVISION_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT}

# install Transformers
pushd src
git clone https://github.com/llm-jp/transformers -b ${PRETRAIN_TRANSFORMERS_TAG}
pushd transformers
python -m pip install -e .
popd  # transformers
popd  # src

# install other requirements
python -m pip install --no-cache-dir -U -r requirements.txt

# install for flash attention
python -m pip install flash-attn=="${PRETRAIN_FLASH_ATTENTION_VERSION}" --no-build-isolation

pushd src

# download our moe-recipes and build helper library
git clone https://github.com/llm-jp/moe-recipes.git -b ${PRETRAIN_MOE_RECIPES_TAG}
pushd moe-recipes/megatron_lm/megatron/core/datasets/
# NOTE(odashi):
# Original makefile in the above directory uses the system's (or pyenv's) python3-config.
# But we need to invoke python3-config installed on our target directory.
MEGATRON_HELPER_CPPFLAGS=(
  -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
  $(python -m pybind11 --includes)
)
MEGATRON_HELPER_EXT=$(${TARGET_DIR}/python/bin/python3-config --extension-suffix)
g++ ${MEGATRON_HELPER_CPPFLAGS[@]} helpers.cpp -o helpers${MEGATRON_HELPER_EXT}
popd

# download our tokeniser
# Tokenizer
git clone https://github.com/llm-jp/llm-jp-tokenizer -b ${PRETRAIN_TOKENIZER_TAG}

# Clone and build CUTLASS
git clone https://github.com/NVIDIA/cutlass.git -b ${PRETRAIN_CUTLASS_TAG}
pushd cutlass
export CUDACXX=${CUDA_HOME}/bin/nvcc
mkdir build && pushd build
cmake .. -DCUTLASS_NVCC_ARCHS=90
make -j $(nproc)
popd  # build
popd  # cutlass
popd  # src
popd  # ${TARGET_DIR}
