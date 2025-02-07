# Install pytorch and torchvision

echo "Installing torch ${PRETRAIN_TORCH_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT} and torchvision ${PRETRAIN_TORCHVISION_VERSION}+cu${PRETRAIN_CUDA_VERSION_SHORT}"

source ${TARGET_DIR}/venv/bin/activate

python -m pip install \
  --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu${PRETRAIN_CUDA_VERSION_SHORT} \
  --find-links https://download.pytorch.org/whl/torch_stable.html \
  torch==${PRETRAIN_TORCH_VERSION} \
  torchvision==${PRETRAIN_TORCHVISION_VERSION}

deactivate
