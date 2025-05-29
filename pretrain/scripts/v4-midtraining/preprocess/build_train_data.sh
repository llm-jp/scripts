#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/dolmino-mix-1124-tokenized"
OUT_DIR="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining/tasks/v4-dolmino-mix-1124"
OUT_FILE="${OUT_DIR}/train_data.all.sh"

mkdir -p "${OUT_DIR}"

{
  echo "# Auto-generated: $(date '+%F %T')"
  echo "export TRAIN_DATA_PATH=("

  find "${ROOT_DIR}" -type f -name '*_text_document.bin' | sort | while read -r BIN; do
    BYTES=$(stat -c%s "${BIN}")
    TOKENS=$(( BYTES / 4 ))
    PREFIX="${BIN%.bin}"
    printf "    %s %s\n" "${TOKENS}" "${PREFIX}"
  done

  echo ")"
} > "${OUT_FILE}"

echo "Generated ${OUT_FILE}"
