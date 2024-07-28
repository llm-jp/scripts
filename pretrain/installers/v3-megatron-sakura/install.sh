#!/bin/bash
#
# Megatron installation script for pretrain jobs on the Sakura cluster
#
# Usage:
# 1. Set the working directory to the directory this file is located.
# 2. Run `sbatch install.sh TARGET_DIR` with setting TARGET_DIR to the actual path.
#
# This script consumes 1 CPU node on the cluster.

#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --job-name=megatron-install
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks-per-node=1

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

cp -a ${INSTALLER_DIR}/* .

source scripts/environment.sh

# record current environment variables
set > installer_envvar.log

# install Python
if ! which pyenv; then
    >&2 echo ERROR: pyenv not found.
    exit 1
fi
pyenv install -s ${PRETRAIN_PYTHON_VERSION}
pyenv local ${PRETRAIN_PYTHON_VERSION}
if [ "$(python --version)" != "Python ${PRETRAIN_PYTHON_VERSION}" ]; then
    >&2 echo ERROR: Python version mismatch: $(python --version) != ${PRETRAIN_PYTHON_VERSION}
    exit 1
fi
python -m venv venv
source venv/bin/activate
python -m pip install -U pip

# install PyTorch
python -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==${PRETRAIN_TORCH_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT} \
    torchvision==${PRETRAIN_TORCHVISION_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT}

# install other requirements
pip install -U -r requirements.txt

mkdir src
pushd src

# install apex
git clone https://github.com/NVIDIA/apex -b ${PRETRAIN_APEX_VERSION}
pushd apex
MAX_JOBS=32 pip install \
    -v \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
popd

# install transformer engine
git clone https://github.com/NVIDIA/TransformerEngine -b v${PRETRAIN_TRANSFORMER_ENGINE_VERSION}
pushd TransformerEngine
MAX_JOBS=32 NVTE_FRAMEWORK=pytorch pip install \
    -v \
    --no-cache-dir \
    --no-build-isolation \
    ./
popd

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
