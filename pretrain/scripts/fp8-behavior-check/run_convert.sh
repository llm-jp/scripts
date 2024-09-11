#!/bin/bash

mkdir -p checkpoints_hf

for d in $(ls checkpoints/3.8b); do
    echo $d
    sbatch \
        --partition=gpu-small \
        scripts/pretrain/scripts/fp8-behavior-check/convert.sh \
        checkpoints/3.8b/$d \
        checkpoints_hf/3.8b/$d
done

