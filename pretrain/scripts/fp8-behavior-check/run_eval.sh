#!/bin/bash

for cfg_file in $(find checkpoints_hf -name config.json | sort); do
    cfg=$(dirname $cfg_file | sed 's/checkpoints_hf\///')
    if [ -e processed/$cfg ]; then
        echo "Already processed: $cfg"
        continue
    fi

    sbatch \
        --partition=gpu-small-lp \
        scripts/pretrain/scripts/fp8-behavior-check/run_llm-jp-eval.sh checkpoints_hf/$cfg $cfg

    mkdir -p $(dirname processed/$cfg) && touch processed/$cfg
    echo "Started: $cfg"
done
