#!/bin/bash
data_dir=/model/experiments/0118_dedup_corpusv4_ja/data

for dir in "${data_dir}"/subcorpus/*; do
    dir_name=$(basename "$dir")

    for file in "$dir/minhash-5gram-20buckets-10hashes/results/deduplicated_output/"*; do
        file_name=$(basename "$file")
        mkdir -p "$data_dir/all/deduped_subcorpus/$dir_name"
        ln -s "$file" "$data_dir/all/deduped_subcorpus/$dir_name/$file_name"
    done
done