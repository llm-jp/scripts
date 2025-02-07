# Installs flash attention.

echo "Installing Transformer Engine ${PRETRAIN_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

pip install \
    --no-build-isolation \
    --no-cache-dir \
    "flash-attn<=${PRETRAIN_FLASH_ATTENTION_VERSION}"

deactivate
