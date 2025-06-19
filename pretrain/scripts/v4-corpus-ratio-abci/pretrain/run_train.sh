#!/bin/bash

set -eu -o pipefail

if [ $# -ne 7 ]; then
    >&2 echo "Usage: $0 <reservation id> <experiment_id> <env-dir> <task-root-dir> <task-name> <wandb-project> <num-nodes>"
    >&2 echo "Example: $0 R0123456789 0123 /path/to/installed/env /path/to/tasks taskname 0174_corpus_ratio 32"
    exit 1
fi

reservation_id=$1; shift
experiment_id=$1; shift
env_dir=$1; shift
task_dir=$1; shift
task_name=$1; shift
wandb_project=$1; shift
num_nodes=$1; shift

# This directory
script_root=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

qsub \
  -P gcg51557 \
  -q ${reservation_id} \
  -N ${experiment_id}_pretrain \
  -l select=${num_nodes},walltime=10000:00:00 \
  -v RTYPE=rt_HF,ENV_DIR=${env_dir},TASK_ROOT_DIR=${task_dir},TASK_NAME=${task_name},WANDB_PROJECT=${wandb_project} \
  -o /dev/null \
  -e /dev/null \
  -m n \
  ${script_root}/qsub_train.sh
