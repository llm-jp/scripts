#!/bin/bash

mkdir -p checkpoints_hf/{3.8b,13b}

# 3.8B

#for d in $(ls checkpoints/3.8b); do
#    echo $d
#    sbatch \
#        --partition=gpu-small \
#        scripts/pretrain/scripts/fp8-behavior-check/convert_3.8b.sh \
#        checkpoints/3.8b/$d \
#        checkpoints_hf/3.8b/$d
#done

# 13B

CONFIGS=(
    contd_0000000.fp8.hybrid.m0.i1.h1.most_recent.wgrad
    contd_0239000.fp8.hybrid.m0.i1.h1.most_recent.wgrad
)

SRC_ROOT=/home/shared/experiments/0031_fp8-behavior/checkpoints/13b
DEST_ROOT=/home/shared/experiments/0031_fp8-behavior/checkpoints_hf/13b

for c in ${CONFIGS[@]}; do
    s=${SRC_ROOT}/$c
    d=${DEST_ROOT}/$c

    for i in `ls $s | egrep '^iter_.{7}$'`; do
        if [ -e $d/$i ]; then
            echo "Exists: $s/$i"
            continue
        fi

        echo "Converting: $s/$i"
        sbatch \
            --job-name=0031_convert \
            --partition=gpu-small-lp \
            scripts/pretrain/scripts/fp8-behavior-check/convert_13b.sh \
            $s/$i \
            $d/$i
    done
done
