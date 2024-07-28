#!/bin/bash
# Installation script for LLM-jp Megatron-LM on Sakura cluster
# Usage: bash install.sh /path/to/myspace
set -eux -o pipefail

TARGET_DIR=$1; shift

INSTALLER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SCRIPTS_DIR=${INSTALLER_DIR}/../../scripts/v3-megatron-sakura

mkdir $TARGET_DIR
pushd $TARGET_DIR

# copy common scripts
cp -a ${INSTALLER_DIR} installer
cp -a ${SCRIPTS_DIR} scripts

source scripts/environment.sh
set  # print environment variables

# install Python
if ! which pyenv; then
    >&2 echo ERROR: pyenv not found.
    exit 1
fi
pyenv install -s $INSTALLER_PYTHON_VERSION
pyenv local $INSTALLER_PYTHON_VERSION
if [ "$(python --version)" != "Python $INSTALLER_PYTHON_VERSION" ]; then
    >&2 echo ERROR: Python version mismatch: $(python --version) != $INSTALLER_PYTHON_VERSION
    exit 1
fi
python -m venv venv
source venv/bin/activate

python -m pip install -U pip==$INSTALLER_PIP_VERSION
python -m pip install -U -r installer/requirements_initial.txt

mkdir src
pushd src

# pytorch
git clone git@github.com:pytorch/pytorch -b v$PYTORCH_VERSION
pushd pytorch
git submodule update --init --recursive
export PYTORCH_BUILD_VERSION=${PYTORCH_VERSION}+cu${CUDA_VERSION_SHORT}
export PYTORCH_BUILD_NUMBER=1
python setup.py install
popd

# torchvision
git clone git@github.com:pytorch/vision -b v$TORCHVISION_VERSION
pushd vision
python setup.py install
popd

# install pytorch
#python -m pip install \
#    --find-links https://download.pytorch.org/whl/torch_stable.html \
#    torch==${INSTALLER_PYTORCH_VERSION}+cu${INSTALLER_CUDA_VERSION_SHORT}
#python -m pip install \
#    --find-links https://download.pytorch.org/whl/torch_stable.html \
#    torchvision==${INSTALLER_TORCHVISION_VERSION}+cu${INSTALLER_CUDA_VERSION_SHORT}

popd  # src

# install other requirements
python -m pip install -U -r installer/requirements_final.txt

pushd src

# install apex
git clone https://github.com/NVIDIA/apex -b $INSTALLER_APEX_VERSION
pushd apex
MAX_JOBS=32 python -m pip install \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    .
popd

# install transformer engine
MAX_JOBS=32 python -m pip install \
    --no-build-isolation \
    git+https://github.com/NVIDIA/TransformerEngine.git@v$INSTALLER_TRANSFORMER_ENGINE_VERSION

# reinstall flash attention
#python -m pip uninstall -y flash-attn
#git clone https://github.com/Dao-AILab/flash-attention -b v$INSTALLER_FLASH_ATTENTION_VERSION
#pushd flash-attention
#MAX_JOBS=32 pip install --no-build-isolation -e .
#popd

# download our Megatron and build helper library
git clone https://github.com/llm-jp/Megatron-LM -b $INSTALLER_MEGATRON_TAG
pushd Megatron-LM/megatron/core/datasets/
make
popd

# download our tokeniser
# Tokenizer
git clone git@github.com:llm-jp/llm-jp-tokenizer -b $INSTALLER_TOKENIZER_TAG

popd  # src
popd  # $TARGET_DIR
