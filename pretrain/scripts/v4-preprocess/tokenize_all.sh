#!/bin/bash

if [ $# -ne 1 ]; then
    >&2 echo "Usage: $0 PRETRAIN_ENV_DIR"
    exit 1
fi

PRETRAIN_ENV_DIR=$1; shift

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
echo "SCRIPT_DIR=${SCRIPT_DIR}"

if [ ! -d corpus/json ]; then
    >&2 echo "Error: Corpus directory doesn't exist. Please run this script on the parent directory of \`corpus\`."
    exit 1
fi

corpus=(
    # code
    corpus/json/code/code_stack_0000.jsonl.gz
    corpus/json/code/code_olmo-starcoder_0000.jsonl.gz

    # en
    corpus/json/en/en_wiki_*.jsonl.gz
    corpus/json/en/en_dolma-books_*.jsonl.gz
    corpus/json/en/en_dolma-wiki_*.jsonl.gz
    corpus/json/en/en_dolma-pes2o_*.jsonl.gz
    corpus/json/en/en_dolma-reddit_*.jsonl.gz
    corpus/json/en/en_olmo-arxiv_*.jsonl.gz
    corpus/json/en/en_olmo-openwebmath_*.jsonl.gz
    corpus/json/en/en_olmo-algebraicstack_*.jsonl.gz
    corpus/json/en/en_mathpile_*.jsonl.gz
    corpus/json/en/en_gsm8k_*.jsonl.gz
    corpus/json/en/en_math_*.jsonl.gz
    corpus/json/en/en_finemath-4plus_*.jsonl.gz
    corpus/json/en/en_dolmino-stackexchange_*.jsonl.gz
    corpus/json/en/en_fineweb-eduscore2_*.jsonl.gz
    corpus/json/en/en_fineweb-rest_*.jsonl.gz

    # ja
    corpus/json/ja/ja_fineweb2_*.jsonl.gz
    corpus/json/ja/ja_wiki_*.jsonl.gz

    # ko
    corpus/json/ko/ko_fineweb2_*.jsonl.gz
    corpus/json/ko/ko_wiki_*.jsonl.gz

    # zh
    corpus/json/zh/zh_fineweb2_*.jsonl.gz
    corpus/json/zh/zh_wiki_*.jsonl.gz
)

# 2 tasks per node
workers=32
mem=800G

for src in ${corpus[@]}; do
    dest=$(echo $src | sed 's/^corpus\/json/corpus\/tokenized/' | sed 's/\.jsonl\.gz$//')
    mkdir -p $(dirname $dest)

    if [ -f $dest.num_tokens ]; then
        echo "$src --> $dest: Exists"
        continue
    fi

    echo "$src --> $dest: Queue"
    sbatch \
        --job-name=0111_tokenize \
        --partition=cpu \
        --cpus-per-task=$workers \
        --mem=$mem \
        --output=$dest.slurm.out \
        ${SCRIPT_DIR}/tokenize_sbatch.sh ${PRETRAIN_ENV_DIR} $src $dest $workers
done
