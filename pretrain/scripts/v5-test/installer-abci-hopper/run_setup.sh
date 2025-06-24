#!/bin/bash

set -eu -o pipefail

if [ $# -ne 1 ]; then
    >&2 echo "Usage: $0 <target-dir>"
    >&2 echo "Example: $0 /path/to/target_dir"
    exit 1
fi

target_dir=$1; shift

qsub \
  -v TARGET_DIR=${target_dir},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  qsub_setup.sh

