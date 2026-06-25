#!/bin/bash
# Submit the Megatron-LM B200 env build to the Slurm cpu partition.

set -eu -o pipefail

if [ $# -ne 1 ]; then
    >&2 echo "Usage: $0 <target-dir>"
    >&2 echo "Example: $0 \$HOME/envs/megatron-lm-b200"
    exit 1
fi

target_dir=$(realpath -m "$1"); shift

if [ -e "${target_dir}" ]; then
    >&2 echo "Error: ${target_dir} already exists. Choose a fresh path or remove it."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

sbatch \
  --export=ALL,TARGET_DIR="${target_dir}" \
  sbatch_setup.sh

echo "Submitted. TARGET_DIR=${target_dir}"
echo "Logs: ${SCRIPT_DIR}/logs/install-<jobid>.out"
