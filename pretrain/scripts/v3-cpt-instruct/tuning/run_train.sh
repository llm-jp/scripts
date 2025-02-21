#!/bin/bash

if [ $# -ne 5 ]; then
    >&2 echo Usage: $0 TASK_DIR MODEL_SIZE ITER PARTITION NODES
    >&2 echo Example: $0 /path/to/tasks/1.8b_inst0.2 1.8b 100000 gpu 1
    exit 1
fi

task_dir=$1; shift
model_size=$1; shift
iter=$1; shift
partition=$1; shift
nodes=$1; shift

sbatch \
    --parsable \
    --partition=${partition} \
    --nodes=${nodes} \
    --output=${task_dir}/sft-%j.out \
    --error=${task_dir}/sft-%j.err \
    scripts/pretrain/scripts/v3-cpt-instruct/tuning/train.sh \
    environments/tuning \
    ${task_dir} \
    ${model_size} \
    ${iter}
