#!/bin/bash
#SBATCH --job-name=0309
#SBATCH --partition=llmjp-pj
#SBATCH --account=llmjp-pj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/install_%j.out
#SBATCH --error=logs/install_%j.err

# Switch to the submission directory
cd ${SLURM_SUBMIT_DIR}

set -eu -o pipefail

TARGET_DIR=$1
echo "TARGET_DIR=${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/src"

# Find the script directory (SLURM_SUBMIT_DIR is where sbatch was called)
SCRIPT_DIR=${SLURM_SUBMIT_DIR}
echo "SCRIPT_DIR=${SCRIPT_DIR}"
# Copy environment scripts to target directory
cp -r "${SCRIPT_DIR}/scripts" "${TARGET_DIR}"

# Load environment variables and SYNKA modules
source "${TARGET_DIR}/scripts/environment.sh"
set > "${TARGET_DIR}/installer_envvar.log"

# Install Libraries (in order of dependency)
source ${SCRIPT_DIR}/src/install_venv.sh            # venv
source ${SCRIPT_DIR}/src/install_pytorch.sh        # PyTorch cu126
source ${SCRIPT_DIR}/src/install_requirements.sh   # accelerate, ninja, wandb, etc.
source ${SCRIPT_DIR}/src/install_apex.sh           # NVIDIA Apex (CUDA ext)
source ${SCRIPT_DIR}/src/install_flash_attention_3.sh  # FA3 (H200 = Hopper sm_90)
source ${SCRIPT_DIR}/src/install_transformer_engine.sh # Transformer Engine
source ${SCRIPT_DIR}/src/install_megatron_lm.sh   # Megatron-LM + C++ helper
source ${SCRIPT_DIR}/src/install_tokenizer.sh      # LLM-jp tokenizer

echo "Done"
