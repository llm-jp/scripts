#!/bin/bash
#
# Tokenization script using llm-jp-tokenizer v3.0b1 for Megatron-LM
#
# Usage: sbatch (or bash) megatron_tokenizer_v3.0b1.sh
#
# Prerequisites:
# * Set the paths for ENV_DIR and DATA_ROOT to appropriate locations before executing.
# 
#SBATCH --job-name=tokenize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euxo pipefail

# Set model `environment` path
ENV_DIR="environment"
# Set dataset path
DATA_ROOT="/path/to/dataset"
SOURCE_DIR="${DATA_ROOT}/raw"
OUTPUT_ROOT="${DATA_ROOT}/tokenized/v3.0b1"
OUTPUT_DIR="${OUTPUT_ROOT}/data"
OUTPUT_INFO="${OUTPUT_ROOT}/token_info.csv"

source ${ENV_DIR}/venv/bin/activate

# Clone Megatron-LM for tokenize data (branch: llmjp0-mdx)
TOKENIZER_DIRNAME=Megatron-LM-tokenizer
cd ${ENV_DIR}/src
if [ ! -d $TOKENIZER_DIRNAME ]; then
  git clone https://github.com/llm-jp/Megatron-LM.git -b llmjp0-mdx $TOKENIZER_DIRNAME
fi
cd $TOKENIZER_DIRNAME

# Tokenize settings
MODEL_PATH="${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model"
TOKENIZER_TYPE="Llama2Tokenizer"
WORKERS=64

# Tokenize
echo "Tokenizer: $MODEL_PATH"
mkdir -p $OUTPUT_DIR

find $SOURCE_DIR -name "*.jsonl" | while read -r file; do
  relative_path="${file#$SOURCE_DIR/}"
  output_path="$OUTPUT_DIR/$relative_path"
  mkdir -p $(dirname "$output_path")

  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info $OUTPUT_INFO \
    --output-prefix "${output_path%.jsonl}" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type $TOKENIZER_TYPE \
    --workers $WORKERS \
    --append-eod
done

echo "Tokenization done"

