#!/bin/bash
#SBATCH --job-name=0118_reshard_corpus
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=slurm_logs/%j-%x.out
#SBATCH --error=slurm_logs/%j-%x.err

# This script splits files from an input directory into smaller compressed chunks,
# preserving directory structure. Supports optional file pattern filtering and .gz input.
#
# Usage:
#   sbatch this_script.sh <input_dir> <output_dir> [unit_size] [pattern]
# Example:
#   sbatch this_script.sh ./data ./sharded 500M '\.jsonl$'

set -eux

input_dir=$1
output_dir=$2
unit_size=${3:-"1G"} # Target size per split chunk (default: 1G)
pattern=${4:-""}     # Optional pattern to filter files

input_dir=$(realpath "$input_dir")
mkdir -p "$output_dir"

# Get list of all files (respecting directory structure)
all_files=$(find -L "$input_dir" -type f)

# Filter files if a pattern is specified
if [[ -n "$pattern" ]]; then
    all_files=$(echo "$all_files" | grep -E "$pattern" || true)
fi

# Exit if no files match the pattern
if [[ -z "$all_files" ]]; then
    echo "No matching files found. Exiting."
    exit 1
fi

# Group files by their parent directory (relative to input_dir)
declare -A dir_files_map
while IFS= read -r file; do
    relative_dir=$(dirname "${file#$input_dir/}")
    output_subdir="$output_dir/$relative_dir"

    mkdir -p "$output_subdir"
    dir_files_map["$output_subdir"]+="$file "
done <<<"$all_files"

# For each group of files, perform splitting
for subdir in "${!dir_files_map[@]}"; do
    file_list=${dir_files_map["$subdir"]}

    split_args=(
        --suffix-length=4
        --additional-suffix=.jsonl
        --line-bytes="$unit_size"
        -d
        --filter='zstd -T0 -o $FILE.zst'
        --verbose
    )

    # Concatenate and split each file; decompress if .gz
    for file in $file_list; do
        if [[ "$file" == *.gz ]]; then
            gunzip -c "$file"
        else
            cat "$file"
        fi
    done | split "${split_args[@]}" - "$subdir/"
done
