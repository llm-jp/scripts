#!/bin/bash
#
# Megatron installation script for pretrain jobs on the Sakura cluster
#
# Usage:
# 1. Set the working directory to the directory this file is located.
# 2. Run `sbatch install.sh TARGET_DIR` with setting TARGET_DIR to the actual path.
#
# This script consumes 1 CPU node on the cluster.

#SBATCH --job-name=pretrain-install
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eux -o pipefail

if [ $# != 1 ]; then
  >&2 echo Usage: sbatch install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$1; shift

echo INSTALLER_DIR=$INSTALLER_DIR
echo TARGET_DIR=$TARGET_DIR

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
python -m pip install -U pip

# install PyTorch
python -m pip install \
  --find-links https://download.pytorch.org/whl/torch_stable.html \
  torch==${PRETRAIN_TORCH_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT} \
  torchvision==${PRETRAIN_TORCHVISION_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT}

# install other requirements
pip install -U -r requirements.txt

pushd src

# install apex
git clone https://github.com/NVIDIA/apex -b ${PRETRAIN_APEX_VERSION}
pushd apex
git submodule update --init --recursive
pip install \
  -v \
  --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  ./
popd

# install transformer engine
# NOTE(odashi):
# This implicitly installs flash-attn with their recommended version.
# If the auto-installed flash-attn causes some problems, we need to re-install it.
pip install \
  --recursive \
  git+https://github.com/NVIDIA/TransformerEngine.git@v${PRETRAIN_TRANSFORMER_ENGINE_VERSION}

# download our Megatron and build helper library
git clone https://github.com/llm-jp/Megatron-LM -b ${PRETRAIN_MEGATRON_TAG}
pushd Megatron-LM/megatron/core/datasets/
make
popd

# download our tokeniser
# Tokenizer
git clone https://github.com/llm-jp/llm-jp-tokenizer -b ${PRETRAIN_TOKENIZER_TAG}

popd  # src
popd  # ${TARGET_DIR}
