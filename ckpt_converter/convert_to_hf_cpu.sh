#!/bin/bash

# LLM-jp v4 model converter (PBS version)
# Usage:
#   bash convert_to_hf_cpu.sh /path/to/task 100000 /path/to/EXP_DIR /path/to/ENV_DIR

set -eu -o pipefail

task_dir=${1:?Usage: $0 TASK_DIR ITER EXP_DIR ENV_DIR}
iter=${2:?Usage: $0 TASK_DIR ITER EXP_DIR ENV_DIR}
exp_dir=${3:?Usage: $0 TASK_DIR ITER EXP_DIR ENV_DIR}
env_dir=${4:?Usage: $0 TASK_DIR ITER EXP_DIR ENV_DIR}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export for local environment (and also pass via qsub -v)
export EXP_DIR="${exp_dir}"
export ENV_DIR="${env_dir}"
export SCRIPT_DIR="${script_dir}"

echo "Submitting job for iteration: ${iter}"

job_output=$(qsub \
  -v TASK_DIR=${task_dir},ITER=${iter},EXP_DIR=${EXP_DIR},ENV_DIR=${ENV_DIR},SCRIPT_DIR=${SCRIPT_DIR},RTYPE=rt_HC \
  -m n \
  -k n \
  "${script_dir}/qsub_convert_cpu.sh")

job_id=$(echo "$job_output" | cut -d'.' -f1)
echo "Job submitted with ID: $job_id"