#!/bin/bash

for task_dir in tasks/*; do
    if [ ! -e ${task_dir}/checkpoints ]; then
        continue
    fi

    for src_dir in ${task_dir}/checkpoints/iter_???????; do
        stem=$(basename ${src_dir})
        dest_dir=${task_dir}/checkpoints_hf/${stem}

        if [[ ${stem} == 'iter_0000000' ]]; then
            #echo "Skip: ${src_dir}"
            continue
        fi

        if [ -e ${dest_dir} ]; then
            #echo "Exists: ${src_dir}"
            continue
        fi
        # Immediately create dest directory so that it is used as a marker
        mkdir -p ${dest_dir}

        echo "Enqueue: ${src_dir}"
        qsub -l select=1 \
            -v MEGATRON_CHECKPOINT_DIR=${src_dir},HF_CHECKPOINT_DIR=${dest_dir},RTYPE=rt_HF \
            -m n \
            -o /dev/null \
            -e /dev/null \
            scripts/pretrain/scripts/v3-instruct-pretrain-abci/converter/qsub_convert.sh
        sleep 1
    done
done
