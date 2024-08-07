#!/bin/bash
#
# llm-jp-eval v1.3.1 installation script on any cluster
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --paratition {partition} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-eval.out 2> logs/install-eval.err`
# - TARGET_DIR: Instalation directory
# 
# This script consumes 1 node on the `cpu` partition on the cluster.
#
#SBATCH --job-name=install-llm-jp-eval
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux 

source scripts/environment.sh

if [ $# -ne 1 ]; then
  case "$HOSTNAME" in
    "login2" | "llm-jp-nvlink")
      >&2 echo Usage: sbatch install.sh TARGET_DIR
      ;;
    "llm-jp")
      >&2 echo Usage: bash install.sh TARGET_DIR
  esac 
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$1; shift

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR

mkdir $TARGET_DIR
pushd $TARGET_DIR

# copy basic scripts
cp -a ${INSTALLER_DIR}/{install.sh,run_llm-jp-eval.sh,scripts} .

# record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir src
pushd src

# install Python
git clone https://github.com/python/cpython -b v${PYTHON_VERSION}
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

# install llm-jp-eval
pushd src
git clone https://github.com/llm-jp/llm-jp-eval -b v${LLM_JP_EVAL_TAG}
pushd llm-jp-eval
if [ -n "$LLM_JP_EVAL_BUG_FIX_COMMIT_IDS" ]; then
  git cherry-pick -m 1 ${LLM_JP_EVAL_BUG_FIX_COMMIT_IDS}
fi
poetry install

# preprocess dataset
poetry run python scripts/preprocess_dataset.py  \
  --dataset-name all  \
  --output-dir ${TARGET_DIR}/dataset/llm-jp-eval \
  --version-name $LLM_JP_EVAL_TAG
# set config
cp ${INSTALLER_DIR}/configs/config.yaml ./configs/
popd  # src
popd  # ${TARGET_DIR}

# check sha256sum on evaluation dataset
HASH_FILE=${INSTALLER_DIR}/configs/hashs.tsv
DEV_DATASET_DIR=${TARGET_DIR}/dataset/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev

set +x
declare -A hash_map

while IFS=$'\t' read -r filename hash; do
  hash_map["$filename"]="$hash"
done < "$HASH_FILE"

for file in ${DEV_DATASET_DIR}/*; do
  filename=$(basename "$file")
  calculated_hash=$(sha256sum "$file" | awk '{print $1}')

  if [[ "${hash_map[$filename]}" != "$calculated_hash" ]]; then
    >&2 echo "NG: $filename"
    >&2 echo "Expected: ${hash_map[$filename]}"
    >&2 echo "Got: $calculated_hash"
  fi
done
