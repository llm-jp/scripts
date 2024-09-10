#!/bin/bash

set -euxo pipefail

CUR_DIR=${1:-..}

# Common settings
source ${CUR_DIR}/megatron_data_formatter.sh
V3_0_INFO_ROOT="/home/shared/corpus/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/token_info"

# ja v3.0 cc-1 
JA_V3_0_INFO_FILE="${V3_0_INFO_ROOT}/2024_0410_ja.sakura_home.csv"
JA_CC1_FILTER="train/ja/cc-1"
JA_CC1_REPEAT=0.8996

# ja v3.0 wiki
JA_WIKI_FILTER="train/ja/wiki"
JA_WIKI_REPEAT=2

set +eux
process_info "$JA_V3_0_INFO_FILE" "$JA_CC1_REPEAT" "$JA_CC1_FILTER"
process_info "$JA_V3_0_INFO_FILE" "$JA_WIKI_REPEAT" "$JA_WIKI_FILTER"
# get $TOTAL_TOKEN_SIZE
check_load_dataset "$TRAIN_DATA_PATH"
set -eux
