#!/bin/bash

if [ $# -ne 2 ]; then
    >&2 echo Usage: $0 TASK_DIR ITER
    >&2 echo Example: $0 /path/to/tasks/1.8b_inst0.2 1.8b 100000
    exit 1
fi

task_dir=$1; shift
iter=$(printf %07d $1); shift

sbatch \
    --parsable \
    --partition=gpu \
    --output=${task_dir}/nemo_to_hf-%j.out \
    --error=${task_dir}/nemo_to_hf-%j.err \
    scripts/pretrain/scripts/v3-cpt-instruct/tuning/nemo_to_hf.sh \
    environments/tuning \
    ${task_dir}/sft/result/$(basename ${task_dir})-iter_${iter}/checkpoints \
    ${task_dir}/checkpoints_hf/iter_${iter} \
    ${task_dir}/checkpoints_hf_tuned/iter_${iter}
