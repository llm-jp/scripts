#!/bin/bash

set -euxo pipefail

CUR_DIR=${1:-..}

# Common settings
source ${CUR_DIR}/megatron_data_formatter.sh
V3_0_INFO_ROOT="/home/shared/corpus/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/token_info"

# en
EN_INFO="${V3_0_INFO_ROOT}/2024_0410_en.sakura_home.csv"
EN_REPEAT=0.2211

set +eux
process_info "$EN_INFO" "$EN_REPEAT"
# get $TOTAL_TOKEN_SIZE
check_load_dataset "$TRAIN_DATA_PATH"
set -eux
