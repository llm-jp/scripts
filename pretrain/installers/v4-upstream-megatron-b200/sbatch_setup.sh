#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=mlm-b200-setup
#SBATCH --output=logs/install-%j.out
#SBATCH --error=logs/install-%j.err
#
# Slurm analog of ABCI's qsub_setup.sh. Builds the upstream Megatron-LM native
# env into ${TARGET_DIR}. Run via run_setup.sh, or directly on the cpu partition:
#   srun -p cpu -c 64 --pty bash sbatch_setup.sh <target_dir>
# (the cpu partition has internet + cores; gpu nodes may not have internet.)

set -eu -o pipefail

# TARGET_DIR comes from --export (sbatch) or $1 (direct srun/bash).
TARGET_DIR="${TARGET_DIR:-${1:-}}"
if [ -z "${TARGET_DIR}" ]; then
    >&2 echo "Usage: TARGET_DIR=<dir> sbatch sbatch_setup.sh   (or: bash sbatch_setup.sh <dir>)"
    exit 1
fi
echo "TARGET_DIR=${TARGET_DIR}"

# Resolve this script's directory (works under sbatch, srun, and plain bash).
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"

mkdir -p "${TARGET_DIR}/src" "${SCRIPT_DIR}/logs"

# Copy the env script into the target so the env is self-describing/reactivatable.
cp -r "${SCRIPT_DIR}/scripts/" "${TARGET_DIR}/"

# Load version pins + CUDA toolkit (also exports TARGET_DIR-derived PY).
export TARGET_DIR
source "${TARGET_DIR}/scripts/environment.sh"
set > "${TARGET_DIR}/installer_envvar.log"

# Install libraries (order matters: torch -> cuda libs -> FA4 -> TE -> apex ->
# requirements -> megatron -> tokenizer -> finalize).
source "${SCRIPT_DIR}/src/install_venv.sh"
source "${SCRIPT_DIR}/src/install_pytorch.sh"
source "${SCRIPT_DIR}/src/install_cuda_libs.sh"
source "${SCRIPT_DIR}/src/install_flash_attention_4.sh"
source "${SCRIPT_DIR}/src/install_transformer_engine.sh"
source "${SCRIPT_DIR}/src/install_apex.sh"
source "${SCRIPT_DIR}/src/install_requirements.sh"
source "${SCRIPT_DIR}/src/install_megatron_lm.sh"
source "${SCRIPT_DIR}/src/install_tokenizer.sh"
source "${SCRIPT_DIR}/src/install_finalize.sh"

echo "Done all installations. Verify on a GPU node:"
echo "  TARGET_DIR=${TARGET_DIR} srun -p gpu --gres=gpu:1 -c 16 bash ${SCRIPT_DIR}/verify/verify_env.sh"
