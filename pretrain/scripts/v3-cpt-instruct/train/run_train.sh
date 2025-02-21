#!/bin/bash

set -eu -o pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 <task-dir> <trainer-suffix> <partition> <nodes>"
    >&2 echo "Example: $0 1.8b_inst0.1 1.8b gpu 8"
    exit 1
fi

task_dir=$1; shift
trainer_suffix=$1; shift
partition=$1; shift
num_nodes=$1; shift

script_root=scripts/pretrain/scripts/v3-cpt-instruct

source ${script_root}/.venv/bin/activate

python ${script_root}/preprocess/configure_corpus.py \
    --token-info token_info \
    --config ${task_dir}/config.yaml \
    --output ${task_dir}/train_data.sh

sbatch \
    --partition=${partition} \
    --nodes=${num_nodes} \
    --output="${task_dir}/train-%j.out" \
    --error="${task_dir}/train-%j.err" \
    ${script_root}/train/sbatch.sh \
    ${script_root}/train/train_${trainer_suffix}.sh \
    ${task_dir}
