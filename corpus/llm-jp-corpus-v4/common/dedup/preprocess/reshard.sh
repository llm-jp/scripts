#!/bin/bash
#SBATCH --job-name=0118_reshard_corpus
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=slurm_logs/%j-%x.out
#SBATCH --error=slurm_logs/%j-%x.err

set -eux

input_dir=$1
output_dir=$2
unit_size=${3:-"1G"}
pattern=${4:-""}

input_dir=$(realpath "$input_dir")
mkdir -p "$output_dir"

# ファイル一覧取得（フォルダ構造を考慮）
all_files=$(find -L "$input_dir" -type f)

# パターンが指定された場合はフィルタリング
if [[ -n "$pattern" ]]; then
    all_files=$(echo "$all_files" | grep -E "$pattern" || true)
fi

# ファイルが見つからない場合の処理
if [[ -z "$all_files" ]]; then
    echo "No matching files found. Exiting."
    exit 1
fi

# 各フォルダごとに処理するため、ディレクトリ単位でグループ化
declare -A dir_files_map
while IFS= read -r file; do
    # ファイルの属するディレクトリ（input_dir からの相対パス）を取得
    relative_dir=$(dirname "${file#$input_dir/}")
    output_subdir="$output_dir/$relative_dir"

    # ディレクトリ構造を再現
    mkdir -p "$output_subdir"

    # ディレクトリごとにファイルリストを格納
    dir_files_map["$output_subdir"]+="$file "
done <<< "$all_files"

# 各ディレクトリごとに分割処理を実行
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

    # ファイルを適切に処理（.gz ファイルは解凍）
    for file in $file_list; do
        if [[ "$file" == *.gz ]]; then
            gunzip -c "$file"
        else
            cat "$file"
        fi
    done | split "${split_args[@]}" - "$subdir/"
done