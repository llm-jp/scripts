#!/bin/bash
#SBATCH --job-name=0095_tokenize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

experiment_dir=/home/shared/experiments/0095_instruction_pretraining
environment_dir=${experiment_dir}/environments/train
tokenizer_path=${environment_dir}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
#src_dir=/home/shared/experiments/0084_instruction_pretraining_dataset/v0.1
src_dir=/home/shared/experiments/0102_synthetic-data/datasets/japanese-cosmopedia-v20250120
corpus_dir=${experiment_dir}/corpus/japanese-cosmopedia-v20250120
token_info_path=${experiment_dir}/token_info/japanese-cosmopedia-v20250120.csv

source ${environment_dir}/scripts/environment.sh
source ${environment_dir}/venv/bin/activate

mkdir -p ${corpus_dir}
mkdir -p $(dirname ${token_info_path})

for src_file in $(cd ${src_dir} && find . -name '*.jsonl'); do
    src_fullpath=${src_dir}$(echo ${src_file} | sed 's/^\.//')
    dest_prefix=${corpus_dir}$(echo ${src_file%.*} | sed 's/^\.//')

    echo "Processing: ${src_fullpath} --> ${dest_prefix}"

    mkdir -p $(dirname ${dest_prefix})

    python ${environment_dir}/src/Megatron-LM_tokenize/tools/preprocess_data.py \
        --input ${src_fullpath} \
        --output-result-total-token-info ${token_info_path} \
        --output-prefix ${dest_prefix} \
        --tokenizer-model ${tokenizer_path} \
        --tokenizer-type Llama2Tokenizer \
        --workers 64 \
        --append-eod
done

echo "Done"

