#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF
#PBS -l select=2
#PBS -l walltime=4:00:00
#PBS -k n
#PBS -N 0163_install

set -euo pipefail

EXP_DIR="/groups/gcg51557/experiments/0163_math_midtraining"
SCRIPT_DIR="${EXP_DIR}/scripts/experiments/v4-hq_tokenizer_test/installer"

mkdir -p ${EXP_DIR}/logs/installer

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBID=${PBS_JOBID%%.*}
LOGFILE=${EXP_DIR}/logs/installer/$TIMESTAMP-$JOBID.out
ERRFILE=${EXP_DIR}/logs/installer/$TIMESTAMP-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

source ${SCRIPT_DIR}/../common/setup.sh

cd ${EXP_DIR}
mkdir -p src
pushd src

echo "Install Python"
mkdir -p python
git clone https://github.com/python/cpython -b v3.12.8
PYTHONPATH=$(pwd)/python
pushd cpython
  ./configure --prefix=${PYTHONPATH} --enable-optimizations
  make -j 64
  make altinstall
popd

echo "Setup venv"
${PYTHONPATH}/bin/python3.12 -m venv ../venv
source ../venv/bin/activate
pip install --upgrade pip

echo "Install torch"
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "Install requirements"
pip install -r ${SCRIPT_DIR}/requirements.txt

echo "Install apex"
git clone --recurse-submodules https://github.com/NVIDIA/apex
pushd apex
  pip install -v \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
popd

# echo "Install flash-attn"
git clone https://github.com/Dao-AILab/flash-attention.git
pushd flash-attention
  git checkout 27f501d && cd hopper/ && python setup.py install
  python_path=`python -c "import site; print(site.getsitepackages()[0])"`
  mkdir -p $python_path/flash_attn_3
  wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py
popd

# pip install \
#     --no-build-isolation \
#     --no-cache-dir \
#     flash-attn

# echo "Install transformer_engine"
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
pushd TransformerEngine
  export NVTE_FRAMEWORK=pytorch
  pip install .
popd

# pip install \
#     --no-build-isolation \
#     --no-cache-dir \
#     transformer_engine[pytorch]

echo "Install Megatron-LM"
git clone https://github.com/llm-jp/Megatron-LM -b v4-old
pushd Megatron-LM/megatron/core/datasets
  MEGATRON_HELPER_CPPFLAGS=(
    -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
    $(python -m pybind11 --includes)
  )
  MEGATRON_HELPER_EXT=$(${PYTHONPATH}/bin/python3.12-config --extension-suffix)
  g++ ${MEGATRON_HELPER_CPPFLAGS[@]} helpers.cpp -o helpers_cpp${MEGATRON_HELPER_EXT}
popd

deactivate
popd

echo "Done"
