#!/bin/bash

set -eu -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 <target-dir> <reservation-id>"
    >&2 echo "Example: $0 /path/to/target_dir R01234567890"
    exit 1
fi

target_dir=$1; shift
reservation_id=$1; shift

qsub \
  -P gcg51557 \
  -q ${reservation_id} \
  -v TARGET_DIR=${target_dir},RTYPE=rt_HF \
  -m n \
  -o /dev/null \
  -e /dev/null \
  -l select=1,walltime=01:00:00 \
  qsub_setup.sh

