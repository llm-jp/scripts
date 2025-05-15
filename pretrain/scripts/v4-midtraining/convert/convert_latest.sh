#!/bin/bash

# LLM-jp v4 model converter (PBS version)
# Usage:
#   bash convert_latest.sh \
#       /path/to/task \    ... TASK_DIR: path to the model to save
#       v3-13b \           ... PARAM_NAME: model config; corresponding file in `params/` should exist

set -eu -o pipefail

task_dir=$1; shift
param_name=$1; shift
dataset_size=$1; shift # 50B or 100B or 300B
iter=$(cat ${task_dir}/${param_name}/checkpoints/latest_checkpointed_iteration.txt)

script_root=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining

qsub \
  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},DATASET_SIZE=${dataset_size},ITER=${iter},RTYPE=rt_HF \
  -m n \
  -o /dev/null \
  -e /dev/null \
  ${script_root}/convert/qsub_convert.sh
