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
src_dir=/home/shared/experiments/0095_instruction_pretraining/raw_corpus/instruction-pretraining-v0.1.1
corpus_dir=${experiment_dir}/corpus/instruction-pretraining-v0.1.1
token_info_path=${experiment_dir}/token_info/instruction-pretraining-v0.1.1_jaster.csv

source ${environment_dir}/scripts/environment.sh
source ${environment_dir}/venv/bin/activate

mkdir -p ${corpus_dir}
mkdir -p $(dirname ${token_info_path})

for src_file in ${src_dir}/jaster*.jsonl; do
    stem=$(basename ${src_file} .jsonl)
    dest_prefix=${corpus_dir}/${stem}
    echo Processing: ${stem}

    python ${environment_dir}/src/Megatron-LM_tokenize/tools/preprocess_data.py \
        --input $src_file \
        --output-result-total-token-info ${token_info_path} \
        --output-prefix ${dest_prefix} \
        --tokenizer-model ${tokenizer_path} \
        --tokenizer-type Llama2Tokenizer \
        --workers 64 \
        --append-eod
done

echo "Done"

