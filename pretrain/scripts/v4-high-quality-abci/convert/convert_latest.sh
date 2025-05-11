#!/bin/bash

# LLM-jp v4 model converter (PBS version)
# Usage:
#   bash convert_latest.sh \
#       /path/to/task \    ... TASK_DIR: path to the model to save
#       v3-13b \           ... PARAM_NAME: model config; corresponding file in `params/` should exist

set -eu -o pipefail

task_dir=$1; shift
param_name=$1; shift
iter=$(cat ${task_dir}/checkpoints/latest_checkpointed_iteration.txt)

qsub \
  -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},ITER=${iter},RTYPE=rt_HF \
  -m n \
  -o /dev/null \
  -e /dev/null \
  /groups/gcg51557/experiments/0150_corpus-v3-vs-v4/scripts/pretrain/scripts/v4-high-quality-abci/convert/qsub_convert.sh
