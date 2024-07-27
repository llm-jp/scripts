#!/bin/bash
# Installation script for LLM-jp Megatron-LM on Sakura cluster
# Usage: bash install.sh /path/to/myspace
set -euxo pipefail

SCRIPT_RELPATH=../../scripts/v3-megatron-sakura
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/${SCRIPT_RELPATH}" &> /dev/null && pwd)
INSTALL_DIR=$1; shift

source ${SCRIPT_DIR}/scripts/environment.sh

set  # print environment variables

mkdir $INSTALL_DIR
pushd $INSTALL_DIR

# copy common scripts
cp -a ${SCRIPT_DIR}/scripts .

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
python -m pip install -U -r ${SCRIPT_DIR}/requirements.txt

mkdir src
pushd src

# install apex
git clone https://github.com/NVIDIA/apex -b $INSTALLER_APEX_VERSION
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
pip install flash-attn==$INSTALLER_FLASH_ATTENTION_VERSION --no-build-isolation

# install transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v$INSTALLER_TRANSFORMER_ENGINE_VERSION

# download our Megatron and build helper library
git clone https://github.com/llm-jp/Megatron-LM -b $INSTALLER_MEGATRON_TAG
pushd Megatron-LM/megatron/core/datasets/
make
popd

# download our tokeniser
# Tokenizer
git clone git@github.com:llm-jp/llm-jp-tokenizer -b $INSTALLER_TOKENIZER_TAG

popd  # src
popd  # $INSTALL_DIR
