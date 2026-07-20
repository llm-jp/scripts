#!/bin/bash
#
# swallow v202411 (transformers v5 variant) installation script
#
# EXPERIMENTAL: Same evaluation code and run scripts as swallow_v202411, but
# the harness venv uses transformers 5.x / vllm 0.19.x / torch 2.10 so that
# models requiring transformers>=5.6 (e.g. Gemma 4 tokenizer configs) can be
# evaluated. Run scripts and patches are taken verbatim from ../swallow_v202411
# so the evaluation behavior itself is unchanged.
#
# Validated on CPU (hf backend: hellaswag/gsm8k/openbookqa/xwinograd_en/
# squadv2/triviaqa with --limit); the vllm backend on GPU is not yet verified.
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
#SBATCH --job-name=install-swallow-tf5
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
TARGET_DIR=$1
# Run scripts, helper scripts and harness patches are shared with the
# original installer so that the evaluation behavior stays identical.
BASE_INSTALLER_DIR=$INSTALLER_DIR/../swallow_v202411
INSTALLER_COMMON=$INSTALLER_DIR/../../../../common/installers.sh

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR
>&2 echo BASE_INSTALLER_DIR=$BASE_INSTALLER_DIR
>&2 echo INSTALLER_COMMON=$INSTALLER_COMMON
source $INSTALLER_COMMON

if [ ! -d "$BASE_INSTALLER_DIR" ]; then
  >&2 echo "ERROR: $BASE_INSTALLER_DIR not found; this installer must live next to swallow_v202411/."
  exit 1
fi

mkdir -p $TARGET_DIR

# Copy basic scripts for swallow eval (verbatim from the original installer)
cp $BASE_INSTALLER_DIR/scripts/run-eval.sh $TARGET_DIR
cp -r $BASE_INSTALLER_DIR/scripts/scripts $TARGET_DIR


pushd $TARGET_DIR

mkdir -p logs

ENV_DIR=${TARGET_DIR}/environment
mkdir -p $ENV_DIR
pushd $ENV_DIR # $ENV_DIR

# src is used to store all resources for from-scratch builds
mkdir -p src
# Retrieve swallow repository
SWALLOW_REPO_URL=https://github.com/llm-jp/swallow-eval-customization.git
if [[ ! -d src/swallow-evaluation ]]; then
  git clone $SWALLOW_REPO_URL src/swallow-evaluation -b v202411
fi

# Apply vLLM >=0.10 compatibility fixes to lm-evaluation-harness-en
# (ray-based data parallelism replaced with multiprocessing; TokensPrompt API)
pushd src/swallow-evaluation
PATCH_FILE=${BASE_INSTALLER_DIR}/patches/vllm_causallms-vllm010-compat.patch
if git apply --reverse --check $PATCH_FILE 2>/dev/null; then
  echo "Patch already applied; skipping."
else
  git apply $PATCH_FILE
fi
popd
# Copy enviroment scripts
cp ${INSTALLER_DIR}/install.sh .
mkdir -p scripts
cp ${INSTALLER_DIR}/scripts/environment.sh scripts/
source scripts/environment.sh

# Record current environment variables
set > installer_envvar.log


# Install Python
# Prefer a uv-managed standalone CPython: it bundles lzma/bz2/sqlite3, which
# a source build silently omits when the node lacks the dev headers.
pushd src
if command -v uv >/dev/null 2>&1; then
  export UV_PYTHON_INSTALL_DIR=${ENV_DIR}/python
  uv python install ${PYTHON_VERSION}
  PYTHON_BIN=$(uv python find --managed-python ${PYTHON_VERSION})
else
  ## Build CPython from source (function in $INSTALLER_COMMON)
  if [[ ! -d "${ENV_DIR}/python" ]]; then
    install_python v${PYTHON_VERSION} ${ENV_DIR}/python
  fi
  PYTHON_BIN=${ENV_DIR}/python/bin/python3
fi
popd # $ENV_DIR

# Prepare venv for swallow harness_en and bigcode
$PYTHON_BIN -m venv venv-harness venv-postprocessing

source venv-postprocessing/bin/activate
pip install wandb pandas
deactivate


# Install vllm (transformers v5 variant)
source venv-harness/bin/activate
pushd src/swallow-evaluation/lm-evaluation-harness-en
pip install --upgrade pip
pip install -e .[math]
pip install sentencepiece==0.2.0 "protobuf>=5.28.3"
pip install 'accelerate>=0.26.0'
# vllm 0.19.x brings torch 2.10 (cu128), which also covers Blackwell (sm_100).
pip install vllm==0.19.1
# transformers 5.6+ is required for models whose tokenizer configs use the
# v5 format (e.g. Gemma 4). NOTE: this leaves a metadata conflict with
# xgrammar (requires transformers<5); xgrammar is only used for guided
# decoding, which the swallow evaluation does not exercise.
pip install transformers==5.6.2
pip install datasets==2.21.0
deactivate
popd # $ENV_DIR
popd  # $TARGET_DIR

echo "Installation done." | tee >(cat >&2)
