#!/bin/bash

set -eu -o pipefail

if [ $# -ne 5 ]; then
    >&2 echo "Usage: $0 <task-dir> <param-name> <dataset_size> <num-nodes> <job_dependency>"
    >&2 echo "Example: $0 v4-high-quality v3-13b 32 afterok:xxxx.pbs1"
    exit 1
fi

task_dir=$1; shift
param_name=$1; shift
dataset_size=$1; shift # 50B or 100B or 300B
num_nodes=$1; shift

# qsub -W depend="$job_dependency" ...
# See `man qsub`
job_dependency=$1; shift

script_root=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining

qsub -l select=${num_nodes} \
  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},DATASET_SIZE=${dataset_size},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  -W depend="$job_dependency" \
  ${script_root}/midtrain/qsub_train.sh
