# Install torch (+ torchvision) for native Blackwell, plus build tooling.
#
# pyproject in many NVIDIA stacks marks torch as container-supplied; on bare
# metal we install it explicitly from the cu130 index.

echo "Installing torch ${PRETRAIN_TORCH_VERSION}+${PRETRAIN_CUDA_INDEX} and torchvision"

uv pip install --no-config --python "${PY}" --index-strategy unsafe-best-match \
  --extra-index-url "https://download.pytorch.org/whl/${PRETRAIN_CUDA_INDEX}" \
  "torch==${PRETRAIN_TORCH_VERSION}" torchvision \
  setuptools wheel "numpy<2" packaging ninja pybind11 cmake "Cython>=3.0.0" psutil
