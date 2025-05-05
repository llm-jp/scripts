#!/bin/bash

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <task-dir> <param-name> <num-nodes>"
    >&2 echo "Example: $0 v4-high-quality v3-13b 32"
    exit 1
fi

task_dir=$1; shift
param_name=$1; shift
num_nodes=$1; shift

script_root=/groups/gcg51557/experiments/0150_corpus-v3-vs-v4/scripts/pretrain/scripts/v4-high-quality-abci

qsub -l select=${num_nodes} \
  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  ${script_root}/pretrain/qsub_train.sh
