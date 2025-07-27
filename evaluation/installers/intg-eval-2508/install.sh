#!/bin/bash
#
# Integrated evaluation platform (version 2508) installation script
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

>&2 echo INSTALLER_DIR=$INSTALLER_DIR
>&2 echo TARGET_DIR=$TARGET_DIR

pushd ./llm-jp-eval-v1.4.1/
bash install.sh $TARGET_DIR \
  > ../logs/install-llm-jp-eval-v1.4.1.out \
  2> ../logs/install-llm-jp-eval-v1.4.1.err
