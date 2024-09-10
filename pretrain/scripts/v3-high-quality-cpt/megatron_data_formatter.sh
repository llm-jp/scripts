#!/bin/bash

# Function to process data and load into TRAIN_DATA_PATH
# Arguments:
#   $1 - INFO_FILE (CSV file with dataset info)
#   $2 - REPEAT (Multiplier for token size)
#   $3 - INCLUDE_CHARS (Optional string filter patterns, space-separated)
process_info() {
  INFO_FILE=$1
  REPEAT=$2
  INCLUDE_CHARS=${3:-}

  if [ -z "$TRAIN_DATA_PATH" ]; then
    TRAIN_DATA_PATH=""
  fi

  # Format information from INFO_FILE
  DATA_PATH_SET=$(awk -v repeat="$REPEAT" -F, '{printf "TRAIN_DATA_PATH=\"${TRAIN_DATA_PATH} %d %s\"\n", int(($3 * repeat) + 0.5), $2}' "$INFO_FILE")

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
}

# Function to load dataset paths ($DATA_PATH_SET) into TRAIN_DATA_PATH
# Arguments:
#   $1 - Single dataset string (data paths and token sizes)
load_train_data_paths() {
  while IFS= read -r line; do
    eval "$line"
    export TRAIN_DATA_PATH
  done <<<"$1"
}

# Function to display dataset paths and their token counts
# Arguments:
#   Optional:
#     -f: ignore duplicates
#     -d: Displays details
#   Positional:
#     TRAIN_DATA_PATH (The dataset paths and token sizes string)
check_load_dataset() {
  FORCE=false
  DISPLAY_DETAILS=false

  # Parse options using getopts
  while getopts "fd" opt; do
    case ${opt} in
      f)
        FORCE=true
        ;;
      d)
        DISPLAY_DETAILS=true
        ;;
      *)
        echo "Usage: check_load_dataset [-f] [-d] TRAIN_DATA_PATH"
        return 1
        ;;
    esac
  done

  shift $((OPTIND - 1))

  # Now $1 should be the TRAIN_DATA_PATH
  TRAIN_DATA_PATH=$1

  if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "Error: TRAIN_DATA_PATH is required."
    return 1
  fi

  declare -A FILE_CHECKER
  TOTAL_TOKEN_SIZE=0

  if [ "$DISPLAY_DETAILS" = true ]; then
    declare -A LANG_TOTAL_TOKENS
    printf "%-5s %-20s %15s\n" "Lang" "File Name" "Token Size"
  fi

  # split $TRAIN_DATA_PATH by " " and get token_size and file path
  set -- $TRAIN_DATA_PATH
  while [ $# -gt 1 ]; do
    token_size=$1
    lang=$(basename "$(dirname "$2")")
    filename=$(basename "${2%.jsonl_text_document}")
    combination="$lang/$filename"

    if [[ -n "${FILE_CHECKER[$combination]:-}" ]]; then
      DUPLICATE_MSG="Duplicate entry for $combination."
      if [ "$FORCE" = false ]; then
        echo "Error: $DUPLICATE_MSG Exiting."
        exit 1
      else
        echo "Warning: $DUPLICATE_MSG"
      fi
    else
      FILE_CHECKER[$combination]=1
    fi

    if [ "$DISPLAY_DETAILS" = true ]; then
      printf "%-5s %-20s %'15d\n" "$lang" "$filename" "$token_size"

      if [[ -z "${LANG_TOTAL_TOKENS[$lang]:-}" ]]; then
        LANG_TOTAL_TOKENS[$lang]=$token_size
      else
        LANG_TOTAL_TOKENS[$lang]=$((LANG_TOTAL_TOKENS[$lang] + token_size))
      fi
    fi
    TOTAL_TOKEN_SIZE=$((TOTAL_TOKEN_SIZE + token_size))

    shift 2
  done

  if [ "$DISPLAY_DETAILS" = true ]; then
    echo "Summary"
    for lang in "${!LANG_TOTAL_TOKENS[@]}"; do
      printf "%-5s %'15d\n" "$lang" "${LANG_TOTAL_TOKENS[$lang]}"
    done

    printf "%-5s %'15d\n" "ALL" "$TOTAL_TOKEN_SIZE"
  fi
  export TOTAL_TOKEN_SIZE
}
