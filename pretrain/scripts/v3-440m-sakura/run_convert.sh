#!/bin/bash

src_root=/home/shared/experiments/0087_llmjp3-440m/checkpoints
dest_root=/home/shared/experiments/0087_llmjp3-440m/checkpoints_hf

for src_ckpt_dir in ${src_root}/iter_???????; do
    ckpt_rel=$(basename ${src_ckpt_dir})
    dest_ckpt_dir=${dest_root}/${ckpt_rel}

    if [ -e ${dest_ckpt_dir} ]; then
        echo "Exists: ${ckpt_rel}"
        continue
    fi

    mkdir -p ${dest_ckpt_dir}

    sbatch \
        scripts/pretrain/scripts/v3-440m-sakura/convert.sh \
        ${src_ckpt_dir} \
        ${dest_ckpt_dir}
    sbatch_result=$?

    if [ ${sbatch_result} -eq 0 ]; then
        echo "Queued: ${ckpt_rel}"
    else
        echo "Error: ${ckpt_rel}"
        rmdir ${dest_ckpt_dir}
    fi
done
