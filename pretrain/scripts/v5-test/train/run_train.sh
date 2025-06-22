#!/bin/bash

set -eu -o pipefail

if [ $# -ne 5 ]; then
    >&2 echo "Usage: $0 <task-dir> <param-name> <num-nodes> <env-dir> <attn-backend>"
    >&2 echo "Example: $0 v4-high-quality v3-13b 32"
    exit 1
fi

task_dir=$1; shift
param_name=$1; shift
num_nodes=$1; shift
env_dir=$1; shift
attn_backend=$1; shift

script_root=/home/ach17726fj/experiments/0176_megatron_upstream_merge/scripts/pretrain/scripts/v5-test

qsub -l select=${num_nodes} \
  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},RTYPE=rt_HF,ENV_DIR=${env_dir},ATTN_BACKEND=${attn_backend} \
  -o /dev/null -e /dev/null \
  -m n \
  ${script_root}/train/qsub_train.sh
