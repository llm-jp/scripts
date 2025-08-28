#!/bin/bash

# LLM-jp v4 model converter (PBS version)
# 6 args required. Usage:
#   bash convert_to_hf_cpu.sh TASK_DIR ITER VENV_DIR MEGATRON_PATH HF_TOKENIZER_PATH OUTPUT_DIR

set -eu -o pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 TASK_DIR ITER VENV_DIR MEGATRON_PATH HF_TOKENIZER_PATH OUTPUT_DIR PARALLEL_SIZE" >&2
  exit 2
fi

task_dir=$1
iter=$2
venv_dir=$3
megatron_path=$4
hf_tokenizer_path=$5
output_dir=$6
parrallel_size=$7

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Create env list in case cumulatively adding -v is not supported
env_list="SCRIPT_DIR=${script_dir},TASK_DIR=${task_dir},ITER=${iter},"
env_list+="VENV_DIR=${venv_dir},MEGATRON_PATH=${megatron_path},"
env_list+="HF_TOKENIZER_PATH=${hf_tokenizer_path},OUTPUT_DIR=${output_dir}"
env_list+="PARALLEL_SIZE=${parrallel_size}"

echo "Submitting job for iteration: ${iter}"

job_output=$(qsub \
  -v "${env_list},RTYPE=rt_HC" \
  -m n \
  -k n \
  "${script_dir}/qsub_convert_cpu.sh")

job_id=$(echo "$job_output" | cut -d'.' -f1)
echo "Job submitted with ID: $job_id"