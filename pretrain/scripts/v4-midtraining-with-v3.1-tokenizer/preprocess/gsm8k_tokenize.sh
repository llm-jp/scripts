#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -l walltime=240:00:00
#PBS -N 0193_tokenize_gsm8k_clean
#PBS -l select=1
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n
#PBS -v RTYPE=rt_HF

# --- 初期設定 ---
cd $PBS_O_WORKDIR

# 実験ディレクトリなどの設定 (元のスクリプトから流用)
EXPERIMENT_DIR=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction
ENV_DIR=${EXPERIMENT_DIR}/environment3
MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM

# ログディレクトリの設定
JOBID=${PBS_JOBID%%.*}
TASK_DIR="$EXPERIMENT_DIR/task"
TOKENIZE_LOG_DIR="${TASK_DIR}/logs/tokenize-$JOBID/"
mkdir -p ${TOKENIZE_LOG_DIR}
LOGFILE=${TOKENIZE_LOG_DIR}/stdout.log
ERRFILE=${TOKENIZE_LOG_DIR}/stderr.log
exec > $LOGFILE 2> $ERRFILE # スクリプト全体の出力をログファイルにリダイレクト

set -eu -o pipefail # エラー時にスクリプトを終了

# --- 環境のロード ---
echo "Loading environments..."
source ${ENV_DIR}/venv/bin/activate
source ${ENV_DIR}/scripts/environment.sh
echo "Environments loaded."

# --- トークナイザ設定 ---
export TOKENIZER_MODEL="${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.1/llm-jp-tokenizer-100k.ver3.1.model"
export TOKENIZER_TYPE=Llama2Tokenizer

# --- 【修正点】入出力パスの直接指定 ---
# 処理したい単一の入力ファイルを指定
INPUT_FILE="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/dolmino-tmp/math/gsm8k-all_clean.jsonl"

# 出力先のディレクトリを指定
OUTPUT_DIR="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/dolmino-mix-1124-v3.1-tokenized/math"

# 出力ファイル名のプレフィックスを定義 (拡張子 .jsonl を除いたもの)
OUTPUT_FILENAME="gsm8k-all_clean"
OUTPUT_PREFIX="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# 出力ディレクトリが存在しない場合は作成
mkdir -p ${OUTPUT_DIR}

echo "--- Configuration ---"
echo "INPUT_FILE: ${INPUT_FILE}"
echo "OUTPUT_PREFIX: ${OUTPUT_PREFIX}"
echo "TOKENIZER_MODEL: ${TOKENIZER_MODEL}"
echo "TOKENIZER_TYPE: ${TOKENIZER_TYPE}"
echo "MEGATRON_PATH: ${MEGATRON_PATH}"
echo "---------------------"

# --- 【修正点】トークン化処理の簡略化 ---
# findとxargsによるループを削除し、単一のコマンドで実行

echo "Starting tokenization for ${INPUT_FILE}..."

python $MEGATRON_PATH/tools/preprocess_data.py \
  --input "${INPUT_FILE}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --tokenizer-type "${TOKENIZER_TYPE}" \
  --workers "$(nproc)" \
  --append-eod

echo "Tokenization completed successfully."
echo "Output files have been saved as ${OUTPUT_PREFIX}.bin and ${OUTPUT_PREFIX}.idx"
