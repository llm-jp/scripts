#!/bin/bash

set -eu -o pipefail

CUDNN_VERSION=8.9.4
NVTE_VERSION=v1.4 # change for stable, v1.4, etc ..

module load cuda/12.1
module load /data/cudnn-tmp-install/modulefiles/8.9.4
module load hpcx/2.17.1-gcc-cuda12/hpcx
module load nccl/2.20.5

cat << EOF > requirements.txt
pybind11

--find-links https://download.pytorch.org/whl/torch_stable.html
torch==2.3.1+cu121
torchvision==0.18.1+cu121

six
regex
numpy

deepspeed
wandb
tensorboard

# mpirun
mpi4py

# tokenizer v2
sentencepiece

# tokenize
nltk

# flash-attn
ninja
packaging
wheel

# checkpoint convert
transformers
accelerate
safetensors

# transformer
einops
EOF

# install Python
INSTALLER_PYTHON_VERSION=3.10.14
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

# install basic requirements
pip install -U pip cmake wheel ninja
pip install -r requirements.txt
pip install zarr tensorstore

# install apex
git clone https://github.com/NVIDIA/apex
cd apex/
MAX_JOBS=32 pip install \
    -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
cd ../

# install transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@${NVTE_VERSION}

# download our Megatron and build helper library
INSTALLER_MEGATRON_TAG=nii-geniac
git clone https://github.com/llm-jp/Megatron-LM -b $INSTALLER_MEGATRON_TAG
pushd Megatron-LM/megatron/core/datasets/
make
popd

# download our tokeniser
# Tokenizer
export INSTALLER_TOKENIZER_TAG=Release-ver3.0b1
git clone git@github.com:llm-jp/llm-jp-tokenizer -b $INSTALLER_TOKENIZER_TAG

popd  # src
popd  # $TARGET_DIR
