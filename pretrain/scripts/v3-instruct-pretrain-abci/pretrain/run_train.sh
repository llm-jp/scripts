#!/bin/bash

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <task-dir> <param-name> <num-nodes>"
    >&2 echo "Example: $0 1.8b_inst0.1 v3-1.8b 8"
    exit 1
fi

task_dir=$1; shift
param_name=$1; shift
num_nodes=$1; shift

script_root=scripts/pretrain/scripts/v3-instruct-pretrain-abci

source ${script_root}/venv/bin/activate

python ${script_root}/preprocess/configure_corpus.py \
    --token-info token_info \
    --config ${task_dir}/config.yaml \
    --output ${task_dir}/train_data.sh


qsub \
  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  ${script_root}/pretrain/qsub_train.sh

