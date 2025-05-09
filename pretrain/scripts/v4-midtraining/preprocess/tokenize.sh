#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -l walltime=12:00:00
#PBS -N 0156_tokenize
#PBS -l select=1
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n
#PBS -v RTYPE=rt_HF

cd $PBS_O_WORKDIR

EXPERIMENT_DIR=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction
SCRIPT_DIR=${EXPERIMENT_DIR}/scripts/pretrain/scripts/v4-midtraining
ENV_DIR=${EXPERIMENT_DIR}/environment
MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM

JOBID=${PBS_JOBID%%.*}
TASK_DIR="$EXPERIMENT_DIR/task"
TOKENIZE_LOG_DIR="${TASK_DIR}/logs/tokenize-$JOBID/"
mkdir -p ${TOKENIZE_LOG_DIR}
LOGFILE=${TOKENIZE_LOG_DIR}/stdout.log
ERRFILE=${TOKENIZE_LOG_DIR}/stderr.log
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

# Arguments
echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"

# Load environment
source ${ENV_DIR}/venv/bin/activate
source ${ENV_DIR}/scripts/environment.sh

# Tokenizer config
export TOKENIZER_MODEL="${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model"
export TOKENIZER_TYPE=Llama2Tokenizer

export WORKERS_PER_PROC=16
N_PROCS=$(($(nproc) / $WORKERS_PER_PROC))

export DATA_DIR=${EXPERIMENT_DIR}/dolmino-mix-1124-extracted-merged
export OUTPUT_DIR=${EXPERIMENT_DIR}/dolmino-mix-1124-tokenized
mkdir -p ${OUTPUT_DIR}
export MEGATRON_PATH
export TOKENIZE_LOG_DIR

# Tokenize
find ${DATA_DIR} -name "*.jsonl" -print0 | \
  sort -z | \
  xargs -0 -P${N_PROCS} -I "{}" bash -c '
    file="{}"
    echo "Tokenizing ${file}"
    relative_path="${file#${DATA_DIR}/}"
    output_path="${OUTPUT_DIR}/${relative_path}"
    tokenize_log_file="${TOKENIZE_LOG_DIR}/${relative_path}.log"
    mkdir -p "$(dirname "$output_path")"
    mkdir -p "$(dirname "$tokenize_log_file")"

    python $MEGATRON_PATH/tools/preprocess_data.py \
      --input "$file" \
      --output-prefix "${output_path%.jsonl}" \
      --tokenizer-model "$TOKENIZER_MODEL" \
      --tokenizer-type "$TOKENIZER_TYPE" \
      --workers "$WORKERS_PER_PROC" \
      --append-eod > "$tokenize_log_file" 2>&1

    echo "Tokenization completed for ${file}"
  '

