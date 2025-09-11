#!/bin/bash

set -eux -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 MODEL_PATH OUTPUT_DIR"
    exit 1
fi

if [ -z "${HF_HOME:-}" ]; then
    >&2 echo "Error: HF_HOME environment variable is not set."
    exit 1
fi

# Arguments
MODEL_NAME_OR_PATH=$1; shift
OUTPUT_DIR=$1; shift

OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
mkdir -p "$OUTPUT_DIR/logs"

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

cd $SCRIPT_DIR/../

# Run swallow evaluation
pushd swallow_v202411/
bash run-eval.sh \
    $MODEL_NAME_OR_PATH \
    $OUTPUT_DIR/swallow \
    0.9 \
    1 \
    1 \
    > $OUTPUT_DIR/logs/swallow_eval.log 2> $OUTPUT_DIR/logs/swallow_eval.err
popd 

# Run llm-jp-eval
pushd llm-jp-eval-v1.4.1/
bash run_llm-jp-eval.sh \
    $MODEL_NAME_OR_PATH \
    $OUTPUT_DIR/llm-jp-eval > $OUTPUT_DIR/logs/llm-jp-eval.log 2> $OUTPUT_DIR/logs/llm-jp-eval.err
popd
