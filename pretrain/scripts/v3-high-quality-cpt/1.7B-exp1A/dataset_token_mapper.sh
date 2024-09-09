#!/bin/bash

set -euxo pipefail

# Common settings
SCRIPT_PATH="../megatron_data_formatter.sh"
V3_0_INFO_ROOT="/model/llmjp0/Megatron-LM/scripts/mdx/tokenize/GENIAC/token_info"
V3_1_INFO_ROOT="/model/llm-jp-corpus/v3.1.0/tokenize/v3.0b1/token_info"

# code
CODE_INFO="${V3_0_INFO_ROOT}/2024_0410_code.csv"
CODE_REPEAT=0.1014
source "$SCRIPT_PATH" "$CODE_INFO" "$CODE_REPEAT"

# en
EN_INFO="${V3_0_INFO_ROOT}/2024_0410_en.csv"
EN_REPEAT=0.1014
source "$SCRIPT_PATH" "$EN_INFO" "$EN_REPEAT"

# ja v3.1 warp-pdf-e00
JA_V3_1_INFO_FILE="${V3_1_INFO_ROOT}/2024_0718_ja_train2.csv"
JA_PDF00_RPEFIX="train2/ja/warp-pdf-e00"
JA_PDF00_REPEAT=0.2028
source "$SCRIPT_PATH" "$JA_V3_1_INFO_FILE" "$JA_PDF00_REPEAT" "$JA_PDF00_RPEFIX"

# ja v3.1 warp-pdf-e02
JA_PDF02_RPEFIX="train2/ja/warp-pdf-e02"
JA_PDF02_REPEAT=0.1014
source "$SCRIPT_PATH" "$JA_V3_1_INFO_FILE" "$JA_PDF02_REPEAT" "$JA_PDF02_RPEFIX"

# ja other
JA_V3_0_INFO_FILE="${V3_0_INFO_ROOT}/2024_0410_ja.csv"
JA_OTHER_PREFIX="train/ja/cc train/ja/kaken train/ja/warp-html train/ja/wiki"
JA_OTHER_REPEAT=0.2028
source "$SCRIPT_PATH" "$JA_V3_0_INFO_FILE" "$JA_OTHER_REPEAT" "$JA_OTHER_PREFIX"

# Display token size and get $TOTAL_TOKEN_SIZE
display_file_and_tokens "$TRAIN_DATA_PATH"
export TOTAL_TOKEN_SIZE