#!/bin/bash

# This script processes JSONL files in parallel using a specified Python script.
# It takes three arguments: an input directory, an output directory, and the number of parallel processes.

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <num_parallel>"
    exit 1
fi

input_dir="$1"
output_dir="$2"
num_parallel="$3"

python_script="scripts/corpus/llm-jp-corpus-v4/ja/patent/parse/parse_patent_jsonl.py"

mkdir -p "$output_dir"

process_file() {
    local file="$1"
    local filename=$(basename "$file")

    python3 "$python_script" < "$file" >"$output_dir/$filename"
}

export -f process_file
export output_dir python_script

find "$input_dir" -type f -print0 | xargs -0 -I {} -P "$num_parallel" bash -c 'process_file "$@"' _ {}

echo "Processing completed."
