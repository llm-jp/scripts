#!/bin/bash
#
# llm-jp-eval v1.4.1 installation script
#
# This script use CPU on a cluster.
#  - In a SLURM environment, it is recommend to use CPU nodes.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --paratition {FIX_ME} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-eval.out 2> logs/install-eval.err`
# - TARGET_DIR: Instalation directory
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
TARGET_DIR=$1
INSTALLER_COMMON=$INSTALLER_DIR/../../../common/installers.sh

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR
>&2 echo INSTALLER_COMMON=$INSTALLER_COMMON

set -eux -o pipefail

if [ $# -ne 1 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\)  install.sh TARGET_DIR
  exit 1
fi

INSTALLER_DIR=$(pwd)
TARGET_DIR=$1
INSTALLER_COMMON=$INSTALLER_DIR/../../../common/installers.sh

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR
>&2 echo INSTALLER_COMMON=$INSTALLER_COMMON
source $INSTALLER_COMMON

mkdir -p $TARGET_DIR

# Copy basic scripts for swallow eval
cp scripts/run-eval.sh $TARGET_DIR
cp -r scripts/scripts $TARGET_DIR


pushd $TARGET_DIR

mkdir -p logs

ENV_DIR=${TARGET_DIR}/environment
mkdir -p $ENV_DIR
pushd $ENV_DIR # $ENV_DIR

# src is used to store all resources for from-scratch builds
mkdir -p src
# Retrieve swallow repository
SWALLOW_REPO_URL=https://github.com/junjiechen-chris/swallow-eval-customization.git 
# TODO: switch to experimental branch for development
git clone $SWALLOW_REPO_URL src/swallow-evaluation -b swallow-gpqanmath
# Copy enviroment scripts
cp ${INSTALLER_DIR}/install.sh .
mkdir scripts
cp ${INSTALLER_DIR}/scripts/environment.sh scripts/
source scripts/environment.sh

# Record current environment variables
set > installer_envvar.log


# Install Python (function in $INSTALLER_COMMON)
pushd src
install_python v${PYTHON_VERSION} ${ENV_DIR}/python
popd # $ENV_DIR

# Prepare venv for swallow harness_en and bigcode
python/bin/python3 -m venv venv-harness venv-postprocessing

source venv-postprocessing/bin/activate
pip install wandb pandas
deactivate


# Install vllm
source venv-harness/bin/activate
pushd src/swallow-evaluation/lm-evaluation-harness-en
pip install --upgrade pip
pip install -e .[math]
pip install sentencepiece==0.2.0 protobuf==5.28.3 transformers==4.46.2
pip install 'accelerate>=0.26.0'
pip install datasets==2.21.0
pip install vllm==v0.6.3.post1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
deactivate
popd # $ENV_DIR
popd  # $TARGET_DIR

echo "Installation done." | tee >(cat >&2)