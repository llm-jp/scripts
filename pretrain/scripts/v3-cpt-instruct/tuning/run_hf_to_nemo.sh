#!/bin/bash

if [ $# -ne 3 ]; then
    >&2 echo Usage: $0 TASK_DIR MODEL_SIZE ITER
    >&2 echo Example: $0 /path/to/tasks/1.8b_inst0.2 1.8b 100000
    exit 1
fi

task_dir=$1; shift
model_size=$1; shift
iter=$(printf %07d $1); shift

sbatch \
    --parsable \
    --partition=gpu \
    --output=${task_dir}/hf_to_nemo-%j.out \
    --error=${task_dir}/hf_to_nemo-%j.err \
    scripts/pretrain/scripts/v3-cpt-instruct/tuning/hf_to_nemo.sh \
    environments/tuning \
    scripts/pretrain/scripts/v3-cpt-instruct/tuning/convert_configs/llm-jp-3-${model_size}.yaml \
    ${task_dir}/checkpoints_hf/iter_${iter} \
    ${task_dir}/checkpoints_nemo/iter_${iter}
