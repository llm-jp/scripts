#!/bin/bash

set -eu -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 <MEGATRON_CHECKPOINT_DIR> <HF_CHECKPOINT_DIR>"
    >&2 echo "Example: $0 /path/to/checkpoints/iter_0001000 /path/to/checkpoints_hf/iter_0001000"
    exit 1
fi

megatron_checkpoint_dir=$1; shift
hf_checkpoint_dir=$1; shift

qsub -l select=1 \
  -v MEGATRON_CHECKPOINT_DIR=${megatron_checkpoint_dir},HF_CHECKPOINT_DIR=${hf_checkpoint_dir},RTYPE=rt_HF \
  -m n \
  -o /dev/null \
  -e /dev/null \
  scripts/pretrain/scripts/v3-instruct-pretrain-abci/converter/qsub_convert.sh
