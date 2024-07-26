#!/bin/bash
set -eu -o pipefail

module load cuda/12.1
module load /data/cudnn-tmp-install/modulefiles/8.9.4
module load hpcx/2.17.1-gcc-cuda12/hpcx
module load nccl/2.20.5

# install basic requirements
pip install -U pip==24.1.2
pip install -U -r requirements.txt

# install apex
git clone https://github.com/NVIDIA/apex -b 24.04.01
cd apex/
MAX_JOBS=32 pip install \
    -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
cd ../

# install flash attention
pip install flash-attn==2.4.2 --no-build-isolation

# install transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.4

# download our Megatron and build helper library
git clone https://github.com/llm-jp/Megatron-LM -b nii-geniac
cd Megatron-LM/megatron/core/datasets/
make

