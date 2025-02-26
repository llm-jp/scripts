#!/bin/bash

set -eu -o pipefail

#if [ $# -ne 1 ]; then
#    >&2 echo "Usage: $0 <hostname>"
#    >&2 echo "Example: $0 hnode040"
#    exit 1
#fi

#host_name=$1; shift
task_dir="tasks/test"
param_name="v3-1.8b"

script_root=scripts/pretrain/scripts/v3-instruct-pretrain-abci

source ${script_root}/venv/bin/activate

host_list=(
    hnode126
    hnode151
    hnode152
    hnode153
    hnode154
    hnode155
    hnode156
    hnode157
    hnode158
)

#python ${script_root}/preprocess/configure_corpus.py \
#    --token-info token_info \
#    --config ${task_dir}/config.yaml \
#    --output ${task_dir}/train_data.sh

#qsub -l select=1:host=${host_name} \
#  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},RTYPE=rt_HF,HOST_NAME=${host_name} \
#  -o /dev/null -e /dev/null \
#  -m n \
#  ${script_root}/pretrain/qsub_test_train.sh

for host_name in "${host_list[@]}"; do
    qsub -l select=1:host=${host_name} \
        -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},RTYPE=rt_HF,HOST_NAME=${host_name} \
        -o /dev/null -e /dev/null \
        -m n \
        ${script_root}/pretrain/qsub_test_train.sh
done

