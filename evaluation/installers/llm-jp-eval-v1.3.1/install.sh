#!/bin/bash
#
# Megatron installation script for pretrain jobs on the Sakura cluster
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

#SBATCH --job-name=install-llm-jp-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
cp -a ${INSTALLER_DIR}/{install.sh,scripts} .

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
popd # src
popd # $TARGET_DIR

# prepare venv
python/bin/python3 -m venv venv
source venv/bin/activate
python -m pip install --no-cache-dir -U pip setuptools
pip install poetry

# download & install llm-jp-eval
pushd src
git clone https://github.com/llm-jp/llm-jp-eval -b ${LLM_JP_EVAL_TAG}
git cherry-pick ${LLM_JP_EVAL_BUG_FIX_COMMIT_IDS}
pushd llm-jp-eval
poetry install

# downlaod & preprocess dataset
poetry run python scripts/preprocess_dataset.py  \
  --dataset-name all  \
  --output-dir ${TARGET_DIR}/llm-jp-eval/dataset \
  --version-name ${LLM_JP_EVAL_TAG}

popd  # src
popd  # ${TARGET_DIR}

# check sha256sum on evaluation dataset
HASH_FILE=scripts/hash.tsv
DEV_DATASET_DIR=$TARGET_DIR/llm-jp-eval/dataset/$LLM_JP_EVAL_TAG/evaluation/dev

declare -A hash_map

while IFS=$'\t' read -r filename hash; do
  hash_map["$filename"]="$hash"
done < "$HASH_FILE"

for file in "$DEV_DATASET_DIR/*"; do
  filename=$(basename "$file")
  calculated_hash=$(sha256sum "$file" | awk '{print $1}')

  if [[ "${hash_map[$filename]}" ~= "$calculated_hash" ]]; then
    echo "NG: $filename"
    echo "Expected: ${hash_map[$filename]}"
    echo "Got: $calculated_hash"
  fi
done