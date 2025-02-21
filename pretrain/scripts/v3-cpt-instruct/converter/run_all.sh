#!/bin/bash

if [ $# -ne 1 ]; then
    >&2 echo "Usage: $0 <partition>"
    exit 1
fi

partition=$1; shift

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
        sbatch \
            --partition=${partition} \
            scripts/pretrain/scripts/v3-cpt-instruct/converter/convert.sh \
            ${src_dir} \
            ${dest_dir}
    done
done
