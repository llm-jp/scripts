#!/bin/bash

set -eux

stage=$1
work_dir="/model/experiments/0118_dedup_corpusv4_ja"
env_path="${work_dir}/environment/.venv/bin/activate"
log_root="${work_dir}/local_logs"

node_list=()
for i in $(seq 0 99); do
    node_list+=("z-cpu$i")
done

source $env_path

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
python_submit_script=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup/minhash/local_multi_node/submit_minhash.py
python_minhash_script=${work_dir}/scripts/corpus/llm-jp-corpus-v4/common/dedup/minhash/local_multi_node/minhash_dedup.py

for _dirname in "${target_dirs[@]}"; do
    _target_dir=$work_dir/data/subcorpus/${_dirname}
    if [ ! -d "$_target_dir" ]; then
        echo "Directory does not exit. Skip: $_target_dir"
        continue
    fi
    continue

    python $python_submit_script \
        --input_dir "${_target_dir}/reshard_1B" \
        --output_dir "$_target_dir" \
        --stage "${stage}" \
        --log_dir "${log_root}/${stage}/${_dirname}" \
        --venv_path $env_path \
        --python_script $python_minhash_script \
        --node_list "${node_list[@]}" \
        --max_node_worker 150
done
