#!/bin/bash

CHECKPOINTS_DIR=checkpoints_hf/3.8b

mkdir -p processed

for d in $(ls ${CHECKPOINTS_DIR}); do
    if [[ -f processed/$d ]]; then
        echo "$d: already processed"
        continue
    fi
    sbatch \
        --partition=gpu-small \
        --priority=1 \
        eval_environment/run_llm-jp-eval.sh ${CHECKPOINTS_DIR}/$d $d \
    && touch processed/$d \
    && echo "$d: queued"
done

