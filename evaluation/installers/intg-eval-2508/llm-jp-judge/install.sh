#!/bin/bash
#
# llm-jp-judge installation script
#
# This script uses CPU on a cluster (a GPU is not required for installation).
#  - In a SLURM environment, it is recommended to use CPU nodes.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --partition {FIX_ME} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-llm-jp-judge.out 2> logs/install-llm-jp-judge.err`
# - TARGET_DIR: Installation directory
#
# Environment variables:
#   HF_TOKEN is required for the AnswerCarefully datasets (gated; also needs
#   an approved access request on Hugging Face). Without it, those datasets
#   are skipped with a warning and the corresponding safety benchmarks are
#   unavailable at run time.
#
#SBATCH --job-name=install-llm-jp-judge
#SBATCH --partition={FIX_ME}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

if [ $# -ne 1 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\) install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$(realpath $1)

LLM_JP_JUDGE_TAG=v2.0.0

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR

mkdir -p $TARGET_DIR
pushd $TARGET_DIR

# Copy basic scripts for llm-jp-judge
cp ${INSTALLER_DIR}/scripts/run_llm-jp-judge.sh .
mkdir -p logs

ENV_DIR=${TARGET_DIR}/environment
mkdir -p $ENV_DIR
pushd $ENV_DIR

# Copy environment scripts
cp ${INSTALLER_DIR}/install.sh .

# llm-jp-judge is uv-based (uv sync --locked). Use uv from PATH, or install a
# standalone uv under the environment when unavailable.
export UV_PYTHON_INSTALL_DIR="${ENV_DIR}/python"
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="${ENV_DIR}/uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${UV_INSTALL_DIR}:${PATH}"
fi

# Record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir -p src
pushd src

if [[ ! -d llm-jp-judge ]]; then
  git clone https://github.com/llm-jp/llm-jp-judge
fi
pushd llm-jp-judge
git fetch origin tag ${LLM_JP_JUDGE_TAG}
git checkout ${LLM_JP_JUDGE_TAG}

# The vllm extra provides `vllm serve`, used by run_llm-jp-judge.sh to host
# the generation target model (and optionally a local judge model).
uv sync --locked --extra vllm

# Datasets are cached under ./data/cache and referenced by run_llm-jp-judge.sh.
bash scripts/download_llm_jp_instructions_v1.0.sh
bash scripts/download_llm_jp_instructions_jculture_v1.0.sh
if [[ ! -d data/cache/safety-boundary-test ]]; then
  bash scripts/download_sbi_safety_boundary.sh
fi

# AnswerCarefully is gated: it requires an approved access request on
# Hugging Face and HF_TOKEN. Keep the rest of the installation usable when
# unavailable; run_llm-jp-judge.sh skips benchmarks whose dataset is missing.
if ! bash scripts/download_ac_v2.0.sh; then
  >&2 echo "WARNING: failed to download AnswerCarefully v2.0 (gated; check the access request and HF_TOKEN). safety_ja will be skipped at run time."
fi
if ! bash scripts/download_ac_borderline_v1.0.sh; then
  >&2 echo "WARNING: failed to download AnswerCarefully borderline v1.0 (gated; check the access request and HF_TOKEN). safety_borderline_ja will be skipped at run time."
fi

popd  # llm-jp-judge
popd  # src
popd  # $ENV_DIR
popd  # $TARGET_DIR

echo "Installation done." | tee >(cat >&2)
