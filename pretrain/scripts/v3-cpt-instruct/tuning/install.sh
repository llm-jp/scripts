#!/bin/bash

set -eu -o pipefail

if [ $# -ne 1 ]; then
  >&2 echo Usage: sbatch install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$1; shift

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR

#mkdir ${TARGET_DIR}
pushd ${TARGET_DIR}

# module load
export MODULEPATH=/data/modules:${MODULEPATH}
module load cuda-12.1.1
module load cudnn-8.9.7
module load hpcx-2.17.1
module load nccl-2.18.3

# record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir src
pushd src

# install Python
git clone https://github.com/python/cpython -b v3.10.14
pushd cpython
./configure --prefix="${TARGET_DIR}/python" --enable-optimizations
make -j 64
make install
popd

popd  # src

# python
${TARGET_DIR}/python/bin/python3 -m venv ${TARGET_DIR}/venv
source venv/bin/activate

pip install --upgrade pip wheel cython
pip install setuptools==69.5.1
pip install packaging
pip install torch
pip install -r ${INSTALLER_DIR}/requirements_cuda12.txt

pushd src

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout b7a4acc1
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
cd ..

# transformer engine install
pip install git+https://github.com/NVIDIA/TransformerEngine.git@c81733f1032a56a817b594c8971a738108ded7d0

popd  # src
popd  # ${TARGET_DIR}
