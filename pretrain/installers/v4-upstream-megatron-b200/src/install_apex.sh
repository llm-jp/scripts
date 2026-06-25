# Install APEX from source (fused_weight_gradient_mlp_cuda, FusedAdam, etc.).
#
# Must build with the CUDA that matches torch -> torch 2.12+cu130 is CUDA 13.0.
# Built for Blackwell (TORCH_CUDA_ARCH_LIST=10.0, set in environment.sh).

echo "Installing APEX (commit=${PRETRAIN_APEX_COMMIT})"

: "${PRETRAIN_APEX_COMMIT:?PRETRAIN_APEX_COMMIT must be set}"
SRC="${TARGET_DIR}/src/apex"
git clone --recurse-submodules "https://github.com/NVIDIA/apex" "${SRC}"
( cd "${SRC}" && git checkout "${PRETRAIN_APEX_COMMIT}" && git submodule update --init --recursive )

export APEX_CPP_EXT=1 APEX_CUDA_EXT=1
uv pip install --no-config --python "${PY}" -v --no-build-isolation --no-cache-dir "${SRC}/"
