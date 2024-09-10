#!/bin/bash

set -euxo pipefail

CUR_DIR=${1:-..}

# Common settings
source ${CUR_DIR}/megatron_data_formatter.sh
V3_0_INFO_ROOT="/home/shared/corpus/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver3.0/token_info"
V3_1_INFO_ROOT="/home/shared/corpus/llm-jp-corpus/v3.1.0/tokenize/v3.0b1/token_info"

# code
CODE_INFO="${V3_0_INFO_ROOT}/2024_0410_code.sakura_home.csv"
CODE_REPEAT=0.1014

# en
EN_INFO="${V3_0_INFO_ROOT}/2024_0410_en.sakura_home.csv"
EN_REPEAT=0.1014

# ja v3.1 warp-pdf-e00
JA_V3_1_INFO_FILE="${V3_1_INFO_ROOT}/2024_0718_ja_train2.sakura_home.csv"
JA_PDF00_FILTER="train2/ja/warp-pdf-e00"
JA_PDF00_REPEAT=0.2028

# ja v3.1 warp-pdf-e02
JA_PDF02_FILTER="train2/ja/warp-pdf-e02"
JA_PDF02_REPEAT=0.1014

# ja other
JA_V3_0_INFO_FILE="${V3_0_INFO_ROOT}/2024_0410_ja.sakura_home.csv"
JA_OTHER_FILTER="train/ja/cc train/ja/kaken train/ja/warp-html train/ja/wiki"
JA_OTHER_REPEAT=0.2028

set +x
process_info "$CODE_INFO" "$CODE_REPEAT"
process_info "$EN_INFO" "$EN_REPEAT"
process_info "$JA_V3_1_INFO_FILE" "$JA_PDF00_REPEAT" "$JA_PDF00_FILTER"
process_info "$JA_V3_1_INFO_FILE" "$JA_PDF02_REPEAT" "$JA_PDF02_FILTER"
process_info "$JA_V3_0_INFO_FILE" "$JA_OTHER_REPEAT" "$JA_OTHER_FILTER"
# get $TOTAL_TOKEN_SIZE
check_load_dataset "$TRAIN_DATA_PATH"
set -x
