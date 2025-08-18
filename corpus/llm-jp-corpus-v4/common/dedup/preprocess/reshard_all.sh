#!/bin/bash

set -eux

work_dir="/model/experiments/0118_dedup_corpusv4_ja"
reshard_script=${work_dir}/scripts/subcorpus/reshard.sh

target_dirs=(
    aozorabunko
    cc
    ceek_news
    e-gov
    fineweb-2
    kaken
    kokkai_giji
    nwc2010
    nwjc
    patent
    sip_comprehensive_html
    sip_comprehensive_pdf-pdf2text
    sip_comprehensive_pdf-surya
    warp_html
    warp_pdf_e0
    warp_pdf_e0.2
    wiki
)

# reshard
reshard_script=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup/preprocess/reshard.sh
unit_size=1G

declare -A patterns=(
    ["kaken"]="train_*"
    ["wiki"]="*train*"
)

for _dir in "${target_dirs[@]}"; do
    trg_dir=$work_dir/data/subcorpus/${_dir}
    if [ ! -d "$trg_dir" ]; then
        echo "Directory does not exit. Skip: $trg_dir"
        continue
    fi
    continue

    pattern=${patterns[$_dir]:-""}

    bash $reshard_script \
        "${trg_dir}/raw" \
        "${trg_dir}/reshard_${unit_size}B" \
        "$unit_size" \
        "$pattern"
done
