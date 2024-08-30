#!/bin/bash
#
# g-leaderboard installation script
#
# This script use CPU on a cluster.
#  - In a SLURM environment, it is recommended to use CPU nodes.
#
# Usage:
# On a cluster with SLURM:
#   Run `sbatch --paratition {partition} install.sh TARGET_DIR`
# On a cluster without SLURM:
#   Run `bash install.sh TARGET_DIR > logs/install-eval.out 2> logs/install-eval.err`
# - TARGET_DIR: Instalation directory
#
#SBATCH --job-name=install-g-leaderboard
#SBATCH --partition={FIX_ME}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eux -o pipefail

if [ $# -ne 1 ]; then
  set +x
  >&2 echo Usage: sbatch \(or bash\) install.sh TARGET_DIR
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
pushd $TARGET_DIR

# Copy basic scripts for g-leaderboard
cp ${INSTALLER_DIR}/scripts/run_g-leaderboard.sh .
mkdir resources
cp ${INSTALLER_DIR}/resources/config_base.yaml resources/
mkdir logs

ENV_DIR=${TARGET_DIR}/environment
mkdir $ENV_DIR
pushd $ENV_DIR

# Copy enviroment scripts
cp ${INSTALLER_DIR}/install.sh .
mkdir scripts

# Create environment.sh
BASE_ENV_SHELL=${INSTALLER_DIR}/scripts/environment.sh
NEW_ENV_SHELL=scripts/environment.sh
cp $BASE_ENV_SHELL $NEW_ENV_SHELL

source $NEW_ENV_SHELL

# Record current environment variables
set > installer_envvar.log

# src is used to store all resources for from-scratch builds
mkdir src
pushd src

# Install Python (function in $INSTALLER_COMMON)
install_python v${PYTHON_VERSION} ${ENV_DIR}/python
popd # $ENV_DIR

# Prepare venv
python/bin/python3 -m venv venv
source venv/bin/activate

# Install g-leaderboard
pushd src
git clone https://github.com/wandb/llm-leaderboard g-leaderboard -b g-leaderboard
pushd g-leaderboard
pip install --no-cache-dir -r requirements.txt

# Deploy blended run config
BLENDED_RUN_CONFIG=${INSTALLER_DIR}/resources/blended_run_config.yaml
cp $BLENDED_RUN_CONFIG blend_run_configs/config.yaml

echo "Installation done." | tee >(cat >&2)
