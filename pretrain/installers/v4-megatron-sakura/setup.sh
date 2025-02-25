#!/bin/bash
#
# Attention:
# This script requires much memory for compiling specific libraries (APEX and Flash Attention)
#
# Usage:
#    bash setup.sh TARGET_DIR
# 
# Usage for batch systems:
#    bash: bash setup.sh ...args...
#    srun: srun setup.sh ...args...
#    sbatch: sbatch --output=installer.slurm.out setup.sh ...args...

#SBATCH --job-name=0111_setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive

set -eu -o pipefail

# Check arguments
if [ $# -ne 1 ]; then
    >&2 echo "Usage: $0 TARGET_DIR"
    exit 1
fi

TARGET_DIR=$(realpath "$1"); shift
echo "TARGET_DIR=${TARGET_DIR}"

# Find the script directory
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(
        scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/Command=/{print $2}' \
        | cut -d ' ' -f 1
    )
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
echo "SCRIPT_DIR=${SCRIPT_DIR}"

mkdir ${TARGET_DIR}
mkdir ${TARGET_DIR}/src

# Copy necessary scripts
cp -r ${SCRIPT_DIR}/scripts ${TARGET_DIR}

# Set variables
source ${TARGET_DIR}/scripts/environment.sh
set > ${TARGET_DIR}/installer_envvar.log

# Install Libraries
source ${SCRIPT_DIR}/src/install_python.sh
source ${SCRIPT_DIR}/src/install_venv.sh
source ${SCRIPT_DIR}/src/install_pytorch.sh
source ${SCRIPT_DIR}/src/install_requirements.sh
source ${SCRIPT_DIR}/src/install_apex.sh
source ${SCRIPT_DIR}/src/install_flash_attention.sh
source ${SCRIPT_DIR}/src/install_transformer_engine.sh
source ${SCRIPT_DIR}/src/install_megatron_lm.sh
source ${SCRIPT_DIR}/src/install_tokenizer.sh

echo "Done"
