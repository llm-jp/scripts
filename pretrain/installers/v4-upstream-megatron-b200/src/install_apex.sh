# Install APEX from source (fused_weight_gradient_mlp_cuda, FusedAdam, etc.).
#
# Must build with the CUDA that matches torch -> torch 2.12+cu130 is CUDA 13.0.
# Built for Blackwell (TORCH_CUDA_ARCH_LIST=10.0, set in environment.sh).

echo "Installing APEX (commit=${PRETRAIN_APEX_COMMIT:-main})"

SRC="${TARGET_DIR}/src/apex"
if [ -n "${PRETRAIN_APEX_COMMIT}" ]; then
  git clone --recurse-submodules "https://github.com/NVIDIA/apex" "${SRC}"
  ( cd "${SRC}" && git checkout "${PRETRAIN_APEX_COMMIT}" && git submodule update --init --recursive )
else
  git clone --depth 1 "https://github.com/NVIDIA/apex" "${SRC}"
fi

export APEX_CPP_EXT=1 APEX_CUDA_EXT=1
uv pip install --no-config --python "${PY}" -v --no-build-isolation --no-cache-dir "${SRC}/"
