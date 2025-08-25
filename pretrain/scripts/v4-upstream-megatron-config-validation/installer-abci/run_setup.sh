#!/bin/bash

set -eu -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 <target-dir> <env_name>"
    >&2 echo "Example: $0 /path/to/target_dir environment_flash_3"
    exit 1
fi

target_dir=$1; shift
env_name=$1; shift

qsub \
  -v TARGET_DIR=${target_dir},ENV_NAME=${env_name},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  qsub_setup.sh

