#!/bin/bash
#
# This script generates a formatted list of dataset paths with token counts, optionally filtering in specified patterns.
#
# Usage: Run `bash megatron_data_formatter.sh INFO_FILE REPEAT_SIZE (INCLUDE_CHARS)`
# - INFO_FILE: CSV file that contains the dataset information (token count, file path, etc.).
# - REPEAT_SIZE: Multiplier to adjust the token count (e.g., 0.25 or 1.0).
# - INCLUDE_CHARS: (Optional) Space-separated list of strings to filter in the dataset paths. 
#   The script will include only the lines in the dataset path output that contain any of these strings.
#
# The script reads the dataset information from INFO_FILE, multiplies the token counts by REPEAT_SIZE, and formats each dataset path with its adjusted token count.
# If INCLUDE_CHARS is provided, it will include only paths that contain one or more of the specified patterns.
#
# Example:
# `bash megatron_data_formatter.sh my_data.csv 0.25 "include_pattern1 include_pattern2"`
# This command will read `my_data.csv`, adjust token counts by multiplying by 0.25, and include only lines that contain "include_pattern1" or "include_pattern2" in the output.
#
set -euxo pipefail

if [ $# -lt 2 ]; then
  set +x
  echo >&2 "Usage: bash megatron_data_formatter.sh \
  INFO_FILE REPEAT_SIZE (INCLUDE_CHARS)"
  exit 1
fi

INFO_FILE=$1
REPEAT=$2
INCLUDE_CHARS=${3:-}

# format information
DATA_PATH_SET=$(awk -v repeat="$REPEAT" -F, '{printf "TRAIN_DATA_PATH=\"${TRAIN_DATA_PATH} %d %s\"\n", int(($3 * repeat) + 0.5), $2}' "$INFO_FILE")


if [ -z "${TRAIN_DATA_PATH:-}" ]; then
  TRAIN_DATA_PATH=""
fi

if [ -n "$INCLUDE_CHARS" ]; then
  IFS=$' '
  for pattern in $INCLUDE_CHARS; do
    SINGLE_DATA_PATH=$(echo "$DATA_PATH_SET" | grep "$pattern")
    IFS=$'\n'
    for line in $SINGLE_DATA_PATH; do
      eval "$line"
      export TRAIN_DATA_PATH
    done
  done
else
  IFS=$'\n'
  for line in $DATA_PATH_SET; do
      eval "$line"
      export TRAIN_DATA_PATH
    done
fi

echo "$TRAIN_DATA_PATH"
