# Installs flash attention.

# Check if Flash Attention 3 should be used instead
if [ "${PRETRAIN_USE_FLASH_ATTENTION_3:-0}" = "1" ]; then
    source ${SCRIPT_DIR}/src/install_flash_attention_3.sh
    return
fi

echo "Installing Flash Attention ${PRETRAIN_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

python -m pip install \
    --no-build-isolation \
    --no-cache-dir \
    "flash-attn==${PRETRAIN_FLASH_ATTENTION_VERSION}"

deactivate
