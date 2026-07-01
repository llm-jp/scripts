# Install cu13 cuDNN / cuBLAS / nvrtc wheels.
#
# No distro cuDNN/NCCL on this cluster, so the cu13 wheels supply them at
# runtime. cuBLAS is pinned: TE 2.16 needs cublasLtGroupedMatrixLayoutInit,
# which older cuBLAS lacks.

echo "Installing cu13 cuDNN + cuBLAS ${PRETRAIN_CUBLAS_VERSION} + nvrtc"

uv pip install --no-config --python "${PY}" --index-strategy unsafe-best-match --upgrade \
  nvidia-cudnn-cu13 "nvidia-cublas==${PRETRAIN_CUBLAS_VERSION}" nvidia-cuda-nvrtc
