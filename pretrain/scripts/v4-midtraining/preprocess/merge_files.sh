#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0156_preprocess_merge_files
#PBS -l select=1
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n
#PBS -v RTYPE=rt_HC

set -eu -o pipefail
shopt -s globstar
shopt -s nullglob
shopt -s failglob

EXP_DIR="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction"
DATA_ROOT="${EXP_DIR}/dolmino-mix-1124-extracted"
OUTPUT_ROOT="${EXP_DIR}/dolmino-mix-1124-extracted-merged"

JOBID=${PBS_JOBID:-shell}
JOBID=${JOBID%%.*}
LOG_DIR="${EXP_DIR}/task/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/merge_files-$JOBID.log" 2>&1

min() {
    a="$1"
    b="$2"
    if [ "$a" -lt "$b" ]; then
        echo "$a"
    else
        echo "$b"
    fi
}

# Workaround for `codesearchnet-ownfilter` and `dolmino-mathsynth`
merge_jsonl() {
  for f in "$@"; do
    cat "$f"
    # If the file is not empty and does not have new line at the end, add a new line.
    [ -s "$f" ] && [ "$(tail -c1 "$f")" != $'\n' ] && printf '\n'
  done
}

# DCLM
## 0000 - 0009, 0010 - 0019, ..., 0240 - 0246
# DCLM_DIR="$DATA_ROOT/dclm"
# max_num=246
# increment=10
# for i in $(seq 0 $increment $max_num); do
#     # cat "$DCLM_DIR/0000/*.json" "$DCLM_DIR/0001/*.json" ... "$DCLM_DIR/0009/*.json" > "$OUTPUT_ROOT/dclm/dclm-0000-0009.jsonl"
#     start=$i
#     end=$(min $(($i + $increment - 1)) $max_num)
#     echo "Merging DCLM files from $start to $end"
#     dir_list=$(seq -f "${DCLM_DIR}/%04g" -s " " $start $end)
#     concat_files=$(find $dir_list -name "*.json" | sort)
#     output_file="$OUTPUT_ROOT/dclm/dclm-$(printf '%04d' $start)-$(printf '%04d' $end).jsonl"
#     mkdir -p "$(dirname "$output_file")"
#     cat $concat_files > $output_file
#     echo "Output file: $output_file"
# done
# 
# # flan
# echo "Merging FLAN files"
# output_flan="$OUTPUT_ROOT/flan/flan-all.jsonl"
# mkdir -p "$(dirname "$output_flan")"
# cat $DATA_ROOT/flan/*.json > "$output_flan"
# echo "Output file: $output_flan"
# 
# # pes2o
# echo "Merging PES2O files"
# output_pes2o="$OUTPUT_ROOT/pes2o/pes2o-all.jsonl"
# mkdir -p "$(dirname "$output_pes2o")"
# cat $DATA_ROOT/pes2o/*.json > "$output_pes2o"
# echo "Output file: $output_pes2o"
# 
# # stackexchange
# echo "Merging StackExchange files"
# output_stackexchange="$OUTPUT_ROOT/stackexchange/stackexchange-all.jsonl"
# mkdir -p "$(dirname "$output_stackexchange")"
# cat $DATA_ROOT/stackexchange/*.json > "$output_stackexchange"
# echo "Output file: $output_stackexchange"
# 
# # wiki
# echo "Merging Wiki files"
# output_wiki="$OUTPUT_ROOT/wiki/wiki-all.jsonl"
# mkdir -p "$(dirname "$output_wiki")"
# cat $DATA_ROOT/wiki/*.json > "$output_wiki"
# echo "Output file: $output_wiki"

# math
## codesearchnet-owmfilter
echo "Merging codesearchnet-owmfilter files"
output_codesearchnet="$OUTPUT_ROOT/math/codesearchnet-owmfilter-all.jsonl"
mkdir -p "$(dirname "$output_codesearchnet")"
merge_jsonl $DATA_ROOT/math/codesearchnet-owmfilter/**/*.jsonl > "$output_codesearchnet"
echo "Output file: $output_codesearchnet"

# ## gsm8k
# echo "Merging gsm8k files"
# output_gsm8k="$OUTPUT_ROOT/math/gsm8k-all.jsonl"
# mkdir -p "$(dirname "$output_gsm8k")"
# cat $DATA_ROOT/math/gsm8k/**/*.jsonl > "$output_gsm8k"
# echo "Output file: $output_gsm8k"
# 
# ## metamath-owmfilter
# echo "Merging metamath-owmfilter files"
# output_metamath="$OUTPUT_ROOT/math/metamath-owmfilter-all.jsonl"
# mkdir -p "$(dirname "$output_metamath")"
# cat $DATA_ROOT/math/metamath-owmfilter/**/*.jsonl > "$output_metamath"
# echo "Output file: $output_metamath"
# 
# ## tulu_math
# echo "Merging tulu_math files"
# output_tulu_math="$OUTPUT_ROOT/math/tulu_math-all.jsonl"
# mkdir -p "$(dirname "$output_tulu_math")"
# cat $DATA_ROOT/math/tulu_math/**/*.jsonl > "$output_tulu_math"
# echo "Output file: $output_tulu_math"

## dolmino_math_synth
echo "Merging dolmino_math_synth files"
output_dolmino_math_synth="$OUTPUT_ROOT/math/dolmino_math_synth-all.jsonl"
mkdir -p "$(dirname "$output_dolmino_math_synth")"
merge_jsonl $DATA_ROOT/math/dolmino_math_synth/**/*.jsonl > "$output_dolmino_math_synth"
echo "Output file: $output_dolmino_math_synth"

# ## mathcoder2-synthmath
# echo "Merging mathcoder2-synthmath files"
# output_mathcoder2="$OUTPUT_ROOT/math/mathcoder2-synthmath-all.jsonl"
# mkdir -p "$(dirname "$output_mathcoder2")"
# cat $DATA_ROOT/math/mathcoder2-synthmath/**/*.jsonl > "$output_mathcoder2"
# echo "Output file: $output_mathcoder2"
# 
# ## tinyGSM-MIND
# echo "Merging tinyGSM-MIND files"
# output_tinygsm_mind="$OUTPUT_ROOT/math/tinyGSM-MIND-all.jsonl"
# mkdir -p "$(dirname "$output_tinygsm_mind")"
# cat $DATA_ROOT/math/tinyGSM-MIND/**/*.jsonl > "$output_tinygsm_mind"
# echo "Output file: $output_tinygsm_mind"
