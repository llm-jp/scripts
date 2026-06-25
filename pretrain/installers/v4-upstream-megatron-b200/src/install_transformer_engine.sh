# Install Transformer Engine (prebuilt cu13 binary + torch frontend).
#
# Unlike ABCI (which compiles TE from source), TE 2.16 ships a prebuilt cu13
# wheel. We install the binary + sdist meta + torch frontend with
# --no-build-isolation --no-deps so it does not drag its own torch/flash-attn.

echo "Installing Transformer Engine ${PRETRAIN_TRANSFORMER_ENGINE_VERSION} (prebuilt cu13)"

# Point any TE torch-frontend compile at the wheel cuDNN + torch-bundled NCCL headers.
CUDNN_DIR=$("${PY}" -c "import nvidia.cudnn,pathlib;print(pathlib.Path(nvidia.cudnn.__path__[0]))" 2>/dev/null || true)
NCCL_DIR=$("${PY}"  -c "import nvidia.nccl,pathlib;print(pathlib.Path(nvidia.nccl.__path__[0]))"  2>/dev/null || true)
[ -n "${CUDNN_DIR}" ] && export CUDNN_PATH="${CUDNN_DIR}" CPATH="${CUDNN_DIR}/include:${CPATH}" LIBRARY_PATH="${CUDNN_DIR}/lib:${LIBRARY_PATH}"
[ -n "${NCCL_DIR}" ]  && export NCCL_HOME="${NCCL_DIR}"  CPATH="${NCCL_DIR}/include:${CPATH}"   LIBRARY_PATH="${NCCL_DIR}/lib:${LIBRARY_PATH}"
echo "CUDNN_PATH=${CUDNN_PATH:-<unset>}  NCCL_HOME=${NCCL_HOME:-<unset>}"

TE_VER="${PRETRAIN_TRANSFORMER_ENGINE_VERSION}"
"${PY}" -m pip install --no-deps --no-build-isolation --extra-index-url https://pypi.nvidia.com \
  "transformer-engine==${TE_VER}" "transformer-engine-cu13==${TE_VER}"
"${PY}" -m pip install --no-deps --no-build-isolation "transformer-engine-torch==${TE_VER}"
