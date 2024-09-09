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

# Function to load dataset paths ($DATA_PATH_SET) and export TRAIN_DATA_PATH
load_train_data_paths() {
  while IFS= read -r line; do
    eval "$line"
    export TRAIN_DATA_PATH
  done <<<"$1"
}

# Function to display dataset paths and their token counts
display_file_and_tokens() {
  declare -A LANG_TOTAL_TOKENS
  TOTAL_TOKEN_SIZE=0

  printf "%-5s %-20s %15s\n" "Lang" "File Name" "Token Size"

  # split $TRAIN_DATA_PATH by " " and, read token_size and file path
  set -- $TRAIN_DATA_PATH
  while [ $# -gt 1 ]; do
    token_size=$1
    lang=$(basename "$(dirname "$2")")
    filename=$(basename "${2%.jsonl_text_document}")
    printf "%-5s %-20s %'15d\n" "$lang" "$filename" "$token_size"

    if [[ -z "${LANG_TOTAL_TOKENS[$lang]:-}" ]]; then
      LANG_TOTAL_TOKENS[$lang]=$token_size
    else
      LANG_TOTAL_TOKENS[$lang]=$((LANG_TOTAL_TOKENS[$lang] + token_size))
    fi
    TOTAL_TOKEN_SIZE=$((TOTAL_TOKEN_SIZE + token_size))

    shift 2
  done

  echo "Summary"
  for lang in "${!LANG_TOTAL_TOKENS[@]}"; do
    printf "%-5s %'15d\n" "$lang" "${LANG_TOTAL_TOKENS[$lang]}"
  done

  printf "%-5s %'15d\n" "ALL" "$TOTAL_TOKEN_SIZE"
  export TOTAL_TOKEN_SIZE
}
# Format information from INFO_FILE
DATA_PATH_SET=$(awk -v repeat="$REPEAT" -F, '{printf "TRAIN_DATA_PATH=\"${TRAIN_DATA_PATH} %d %s\"\n", int(($3 * repeat) + 0.5), $2}' "$INFO_FILE")

# Ensure TRAIN_DATA_PATH is initialized if not already
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
  TRAIN_DATA_PATH=""
fi

# Process data with optional filtering
if [ -n "$INCLUDE_CHARS" ]; then
  IFS=$' '
  for pattern in $INCLUDE_CHARS; do
    SINGLE_DATA_PATH=$(echo "$DATA_PATH_SET" | grep "$pattern" || true) # Prevent grep failure if no match
    if [ -n "$SINGLE_DATA_PATH" ]; then
      load_train_data_paths "$SINGLE_DATA_PATH"
    fi
  done
else
  load_train_data_paths "$DATA_PATH_SET"
fi
