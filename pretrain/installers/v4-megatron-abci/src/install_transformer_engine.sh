# Installs Transformer Engine.

echo "Installing Transformer Engine ${PRETRAIN_TRANSFORMER_ENGINE_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

# install transformer engine
# NOTE(odashi):
# This implicitly installs flash-attn with their recommended version.
# If the auto-installed flash-attn causes some problems, we need to re-install it.
pip install --no-build-isolation --no-cache-dir transformer_engine[pytorch]==1.9.0

deactivate
