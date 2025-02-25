#!/bin/bash

set -eux

work_dir="/home/shared/experiments/0118_dedup_corpusv4_ja"
reshard_script=${work_dir}/scripts/subcorpus/reshard.sh
minhash_script=${work_dir}/scripts/subcorpus/minhash_dedup.py
python_env=${work_dir}/environment/.venv

trg_dirs=(
    aozorabunko
#    cc
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

job_ids=()

wait_for_jobs() {
    local job_ids=("$@")
    set +x
    for job_id in "${job_ids[@]}"; do
        while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
            sleep 10
        done
    done
    set -x
}

# reshard
reshard_script=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup/subcorpus/reshard.sh
unit_size=1G

declare -A patterns=(
    ["kaken"]="train_*"
    ["wiki"]="*train*"
)

for _dir in "${trg_dirs[@]}"; do
    trg_dir=$work_dir/data/subcorpus/${_dir}
    if [ ! -d "$trg_dir" ]; then
        echo "Directory does not exit. Skip: $trg_dir"
        continue
    fi
    continue

    pattern=${patterns[$_dir]:-""}

    job_id=$(
        sbatch $reshard_script \
            "${trg_dir}/raw" \
            "${trg_dir}/reshard_${unit_size}B" \
            "$unit_size" \
            "$pattern"
    )
    job_ids+=("$job_id")
done
wait_for_jobs "${job_ids[@]}"
job_ids=()

# minhash
source ${python_env}/bin/activate
for _dir in "${trg_dirs[@]}"; do
    trg_dir=$work_dir/data/subcorpus/${_dir}
    if [ ! -d "$trg_dir" ]; then
        echo "Directory does not exit. Skip: $trg_dir"
        continue
    fi
    #continue

    python $minhash_script \
        --input "${trg_dir}/reshard_${unit_size}B" \
        --output "${trg_dir}"
done
