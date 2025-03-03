#!/bin/bash
#PBS -P gcg51557
#PBS -q R10415
#PBS -v RTYPE=rt_HF
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -o /dev/null
#PBS -e /dev/null

cd $PBS_O_WORKDIR

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBID=${PBS_JOBID%%.*}
mkdir -p logs
LOGFILE=logs/install-$JOBID.out
ERRFILE=logs/install-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

echo "TARGET_DIR=${TARGET_DIR}"

# Find the script directory
if [ -n "${PBS_JOBID:-}" ]; then
    SCRIPT_PATH="$PBS_O_WORKDIR/$(basename "$0")"
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
