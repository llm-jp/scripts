#!/bin/bash
# Installation script for LLM-jp Megatron-LM on Sakura cluster
# Usage: bash install.sh /path/to/myspace
set -euxo pipefail

PYTHON_VERSION=3.10.14
CUDA_VERSION_MAJOR=12
CUDA_VERSION_MINOR=1
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDNN_VERSION=8.9.4
HPCX_VERSION=2.17.1
NCCL_VERSION=2.20.5
PIP_VERSION=24.1.2
APEX_VERSION=24.04.01
FLASH_ATTENTION_VERSION=2.4.2
TRANSFORMER_ENGINE_VERSION=1.4
MEGATRON_TAG=nii-geniac
TOKENIZER_TAG=Release-ver3.0b1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INSTALL_DIR=$1; shift

module load cuda/$CUDA_VERSION
module load /data/cudnn-tmp-install/modulefiles/$CUDNN_VERSION
module load hpcx/${HPCX_VERSION}-gcc-cuda${CUDA_VERSION_MAJOR}/hpcx
module load nccl/$NCCL_VERSION

set  # print environment variables

mkdir $INSTALL_DIR
pushd $INSTALL_DIR

# install Python
if ! which pyenv; then
    >&2 echo ERROR: pyenv not found.
    exit 1
fi
pyenv install -s $PYTHON_VERSION
pyenv local $PYTHON_VERSION
if [ "$(python --version)" != "Python $PYTHON_VERSION" ]; then
    >&2 echo ERROR: Python version mismatch: $(python --version) != $PYTHON_VERSION
    exit 1
fi
python -m venv venv
source venv/bin/activate
python -m pip install -U pip==$PIP_VERSION
python -m pip install -U -r ${SCRIPT_DIR}/requirements.txt

mkdir src
pushd src

# install apex
git clone https://github.com/NVIDIA/apex -b $APEX_VERSION
pushd apex
MAX_JOBS=32 pip install \
    -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
popd

# install flash attention
pip install flash-attn==$FLASH_ATTENTION_VERSION --no-build-isolation

# install transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v$TRANSFORMER_ENGINE_VERSION

# download our Megatron and build helper library
git clone https://github.com/llm-jp/Megatron-LM -b $MEGATRON_TAG
pushd Megatron-LM/megatron/core/datasets/
make
popd

# download our tokeniser
# Tokenizer
git clone git@github.com:llm-jp/llm-jp-tokenizer -b $TOKENIZER_TAG

popd  # src
popd  # $INSTALL_DIR
