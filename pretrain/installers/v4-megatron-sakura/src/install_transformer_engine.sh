# Installs Transformer Engine.

echo "Installing Transformer Engine ${PRETRAIN_TRANSFORMER_ENGINE_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

# install transformer engine
# NOTE(odashi):
# This implicitly installs flash-attn with their recommended version.
# If the auto-installed flash-attn causes some problems, we need to re-install it.
NVTE_FRAMEWORK=pytorch python -m pip install \
  -v \
  --no-cache-dir \
  --no-build-isolation \
  git+https://github.com/NVIDIA/TransformerEngine.git@v${PRETRAIN_TRANSFORMER_ENGINE_VERSION}

deactivate
