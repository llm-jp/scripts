# Installs Transformer Engine.
#
# SYNKA には cudnn モジュールがないため、pip でインストール済みの
# nvidia-cudnn-cu12 のヘッダを CPATH / LIBRARY_PATH に追加してからビルドする。

echo "Installing Transformer Engine ${PRETRAIN_TRANSFORMER_ENGINE_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

# cuDNN headers/libs from pip-installed nvidia-cudnn-cu12
CUDNN_BASE=${TARGET_DIR}/venv/lib/python3.10/site-packages/nvidia/cudnn
export CPATH=${CUDNN_BASE}/include:${CPATH:-}
export LIBRARY_PATH=${CUDNN_BASE}/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${CUDNN_BASE}/lib:${LD_LIBRARY_PATH:-}

echo "Using cuDNN headers from: ${CUDNN_BASE}/include"

# NOTE(odashi):
# This implicitly installs flash-attn with their recommended version.
# If the auto-installed flash-attn causes some problems, we need to re-install it.
pip install --no-build-isolation --no-cache-dir transformer_engine[pytorch]==${PRETRAIN_TRANSFORMER_ENGINE_VERSION}

deactivate