#!/bin/bash

# LLM-jp v4 model converter (PBS version)
# Usage:
#   bash convert_last_10ckpts.sh \
#       /path/to/task \    ... TASK_DIR: path to the model to save
#       v3-1.8b \          ... PARAM_NAME: model config; corresponding file in `params/` should exist
#       123000             ... Checkpoint step to convert

set -eu -o pipefail

task_dir=$1; shift
param_name=$1; shift
iter=$1; shift

prev_job_id=""
for i in $(seq $((iter - 9)) $iter); do
  if [[ -n "$prev_job_id" ]]; then
    prev_job_id=$(qsub \
      -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},ITER=${i},RTYPE=rt_HF \
      -m n \
      -o /dev/null \
      -e /dev/null \
      -W depend=afterok:$prev_job_id \
      scripts/pretrain/scripts/v3-instruct-pretrain-abci/converter/qsub_convert.sh)
  else
    prev_job_id=$(qsub \
      -v TASK_DIR=${task_dir},PARAM_NAME=${param_name},ITER=${i},RTYPE=rt_HF \
      -m n \
      -o /dev/null \
      -e /dev/null \
      scripts/pretrain/scripts/v3-instruct-pretrain-abci/converter/qsub_convert.sh)
  fi
done
