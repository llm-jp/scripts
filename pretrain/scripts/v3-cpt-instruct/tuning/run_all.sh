#!/bin/bash

set -eu -o pipefail

if [ $# -ne 5 ]; then
    >&2 echo "Usage: $0 TASK_DIR MODEL_SIZE ITER PARTITION NODES"
    >&2 echo "Example: $0 /path/to/tasks/1.8b_inst0.2 1.8b 100000 gpu 1"
    exit 1
fi

task_dir=$1; shift
model_size=$1; shift
iter_raw=$1; shift
partition=$1; shift
nodes=$1; shift

iter_padded=$(printf %07d ${iter_raw})

script_root=scripts/pretrain/scripts/v3-cpt-instruct/tuning

src_hf_dir=${task_dir}/checkpoints_hf/iter_${iter_padded}
if [ ! -d ${src_hf_dir} ]; then
    >&2 echo "Error: Source checkpoint doesn't exist: ${src_hf_dir}"
    exit 1
fi

jobid_hf_to_nemo=$(
    sbatch \
        --parsable \
        --partition=gpu \
        --output=${task_dir}/hf_to_nemo-%j.out \
        --error=${task_dir}/hf_to_nemo-%j.err \
        scripts/pretrain/scripts/v3-cpt-instruct/tuning/hf_to_nemo.sh \
        environments/tuning \
        scripts/pretrain/scripts/v3-cpt-instruct/tuning/convert_configs/llm-jp-3-${model_size}.yaml \
        ${src_hf_dir} \
        ${task_dir}/checkpoints_nemo/iter_${iter_padded}
)

jobid_train=$(
    sbatch \
        --dependency=afterok:${jobid_hf_to_nemo} \
        --parsable \
        --partition=${partition} \
        --nodes=${nodes} \
        --output=${task_dir}/sft-%j.out \
        --error=${task_dir}/sft-%j.err \
        scripts/pretrain/scripts/v3-cpt-instruct/tuning/train.sh \
        environments/tuning \
        ${task_dir} \
        ${model_size} \
        ${iter_raw}
)

jobid_nemo_to_hf=$(
    sbatch \
        --dependency=afterok:${jobid_train} \
        --parsable \
        --partition=gpu \
        --output=${task_dir}/nemo_to_hf-%j.out \
        --error=${task_dir}/nemo_to_hf-%j.err \
        scripts/pretrain/scripts/v3-cpt-instruct/tuning/nemo_to_hf.sh \
        environments/tuning \
        ${task_dir}/sft/result/$(basename ${task_dir})-iter_${iter_padded}/checkpoints \
        ${src_hf_dir} \
        ${task_dir}/checkpoints_hf_tuned/iter_${iter_padded}
)

echo Job IDs:
echo   hf_to_nemo: ${jobid_hf_to_nemo}
echo   train: ${jobid_train}
echo   nemo_to_hf: ${jobid_nemo_to_hf}
