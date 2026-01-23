#!/bin/bash

src_root=/home/shared/experiments/0066_v3-13b-cpt-lr/checkpoints
dest_root=/home/shared/experiments/0066_v3-13b-cpt-lr/checkpoints_hf

for src_exp_dir in ${src_root}/exp*; do
    exp_rel=$(basename ${src_exp_dir})

    for src_ckpt_dir in ${src_exp_dir}/iter_???????; do
        ckpt_rel=${exp_rel}/$(basename ${src_ckpt_dir})
        dest_ckpt_dir=${dest_root}/${ckpt_rel}

        if [ -e ${dest_ckpt_dir} ]; then
            echo "Exists: ${ckpt_rel}"
            continue
        fi

        mkdir -p ${dest_ckpt_dir}

        sbatch \
            --job-name=0066_convert \
            --partition=gpu-small \
            --priority=1 \
            --mem=200G \
            scripts/pretrain/scripts/v3-13b-exp4-cpt-lr-sakura/convert.sh \
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
done
