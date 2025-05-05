#!/bin/bash

set -eu -o pipefail

DATA_ROOT="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/dolmino-mix-1124/data"
OUTPUT_ROOT="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/dolmino-mix-1124-extracted"

mkdir -p "$OUTPUT_ROOT"

extract_zstd() {
    local input_file="$1"
    local output_file="${input_file%.zst}"
    output_file="${output_file/$DATA_ROOT/$OUTPUT_ROOT}"
    mkdir -p "$(dirname "$output_file")"
    echo zstd -d "$input_file" -o "$output_file"
    zstd -f -d "$input_file" -o "$output_file"
}

extract_gzip() {
    local input_file="$1"
    local output_file="${input_file%.gz}"
    output_file="${output_file/$DATA_ROOT/$OUTPUT_ROOT}"
    mkdir -p "$(dirname "$output_file")"
    echo gunzip -c "$input_file" > "$output_file"
    gunzip -c "$input_file" > "$output_file"
}

copy_only() {
    local input_file="$1"
    local output_file="${input_file}"
    output_file="${output_file/$DATA_ROOT/$OUTPUT_ROOT}"
    mkdir -p "$(dirname "$output_file")"
    echo cp "$input_file" "$output_file"
    cp "$input_file" "$output_file"
}

# DCLM
for file in $(find "$DATA_ROOT/dclm" -name "*.json.zst" -type f); do
    extract_zstd "$file"
done

# flan
for file in $(find "$DATA_ROOT/flan" -name "*.json.gz" -type f); do
    extract_gzip "$file"
done

# pes2o
for file in $(find "$DATA_ROOT/pes2o" -name "*.json.gz" -type f); do
    extract_gzip "$file"
done

# stackexchange
for file in $(find "$DATA_ROOT/stackexchange" -name "*.json.gz" -type f); do
    extract_gzip "$file"
done

# wiki
for file in $(find "$DATA_ROOT/wiki" -name "*.json.gz" -type f); do
    extract_gzip "$file"
done

# math
## codesearchnet-owmfilter
for file in $(find "$DATA_ROOT/math/codesearchnet-owmfilter" -name "*.jsonl.gz" -type f); do
    extract_gzip "$file"
done

## gsm8k (train only)
for file in $(find "$DATA_ROOT/math/gsm8k/*/train" -name "*.jsonl.zst" -type f); do
    extract_zstd "$file"
done

## metamath-owmfilter
for file in $(find "$DATA_ROOT/math/metamath-owmfilter" -name "*.jsonl.gz" -type f); do
    extract_gzip "$file"
done

## tulu_math
for file in $(find "$DATA_ROOT/math/tulu_math" -name "*.jsonl" -type f); do
    copy_only "$file"
done

## dolmino_math_synth
for file in $(find "$DATA_ROOT/math/dolmino_math_synth" -name "*.jsonl" -type f); do
    copy_only "$file"
done

## mathcoder2-synthmath
for file in $(find "$DATA_ROOT/math/mathcoder2-synthmath" -name "*.jsonl" -type f); do
    copy_only "$file"
done

## tinyGSM-MIND
for file in $(find "$DATA_ROOT/math/tinyGSM-MIND" -name "*.jsonl.zst" -type f); do
    extract_gzip "$file"
done

