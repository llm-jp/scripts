# Installs flash attention.

echo "Installing Flash Attention ${PRETRAIN_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

python -m pip install \
    --no-build-isolation \
    --no-cache-dir \
    "flash-attn==${PRETRAIN_FLASH_ATTENTION_VERSION}"

deactivate
