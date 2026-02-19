#!/bin/bash
#
# llm-jp-eval v2.1.0 installation script
#
# This script use CPU on a cluster.
#  - In a SLURM environment, it is recommend to use CPU nodes.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --partition {FIX_ME} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-eval.out 2> logs/install-eval.err`
# - TARGET_DIR: Installation directory
#
#SBATCH --job-name=install-llm-jp-eval
#SBATCH --partition={FIX_ME}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

if [ $# -ne 1 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\)  install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$(realpath $1)

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR

mkdir -p $TARGET_DIR
pushd $TARGET_DIR

# Copy basic scripts for llm-jp-eval
cp ${INSTALLER_DIR}/scripts/run_llm-jp-eval.sh .
mkdir -p resources
cp ${INSTALLER_DIR}/resources/*.yaml resources/
mkdir -p logs

ENV_DIR=${TARGET_DIR}/environment
mkdir -p $ENV_DIR
pushd $ENV_DIR

export UV_PYTHON_INSTALL_DIR="${ENV_DIR}/python"

# Copy environment scripts
cp ${INSTALLER_DIR}/install.sh .
mkdir -p scripts
cp ${INSTALLER_DIR}/scripts/environment.sh scripts/
source scripts/environment.sh

cp ${INSTALLER_DIR}/scripts/update_result_json.py scripts/
cp -r ${INSTALLER_DIR}/conf .

# Record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir -p src
pushd src

if [[ ! -d llm-jp-eval ]]; then
  git clone --depth 1 https://github.com/llm-jp/llm-jp-eval
fi
pushd llm-jp-eval
git fetch origin $LLM_JP_EVAL_COMMIT_HASH
git checkout $LLM_JP_EVAL_COMMIT_HASH
if [ -n "$LLM_JP_EVAL_BUG_FIX_COMMIT_IDS" ]; then
  git cherry-pick -m 1 ${LLM_JP_EVAL_BUG_FIX_COMMIT_IDS}
fi
uv sync --no-cache --python $PYTHON_VERSION

if [[ ! -d llm-jp-eval-inference ]]; then
  git clone --depth 1 https://github.com/llm-jp/llm-jp-eval-inference
fi
pushd llm-jp-eval-inference
git fetch origin $LLM_JP_EVAL_INFERENCE_COMMIT_HASH
git checkout $LLM_JP_EVAL_INFERENCE_COMMIT_HASH

pushd inference-modules/vllm
uv sync --no-cache --python $PYTHON_VERSION

popd  # inference-modules/vllm
popd  # llm-jp-eval-inference

# Preprocess dataset
# --version-name は実のところデータセットの出力先のディレクトリの名称としてだけ使われる。
# この値は `llm_jp_eval.__version__` に一致させなければならない。dumpやeval時にこの値から
# データセットを読みに行くので、不一致があるとエラーになる。
# llm-jp-evalのパッケージとしてのバージョンはもう >2.0.0 であるが、`__version__` が依然として
# 2.0.0 であるため、やむを得ず 2.0.0 を指定している。
uv run python scripts/preprocess_dataset.py \
  --dataset-name all \
  --output-dir ${ENV_DIR}/data/llm-jp-eval \
  --version-name 2.0.0

popd  # llm-jp-eval
popd  # src
popd  # $ENV_DIR
popd  # $TARGET_DIR

# TODO: Check integrity
# # Check integrity of evaluation dataset
# HASH_FILE=${INSTALLER_DIR}/resources/sha256sums.csv
# DEV_DATASET_DIR=${ENV_DIR}/data/llm-jp-eval/${LLM_JP_EVAL_TAG}/evaluation/dev

# set +x
# declare -A hash_map

# while IFS=$'\t' read -r filename hash; do
#   hash_map["$filename"]="$hash"
# done < "$HASH_FILE"

# for file in ${DEV_DATASET_DIR}/*; do
#   filename=$(basename "$file")
#   calculated_hash=$(sha256sum "$file" | awk '{print $1}')

#   if [[ "${hash_map[$filename]}" != "$calculated_hash" ]]; then
#     >&2 echo "NG: $filename"
#     >&2 echo "Expected: ${hash_map[$filename]}"
#     >&2 echo "Got: $calculated_hash"
#     echo "Aborted." | tee >(cat >&2)
#     exit 1
#   fi
# done

echo "Installation done." | tee >(cat >&2)
