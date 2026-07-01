# Install FlashAttention-4 (Blackwell / sm_100).
#
# CAUTION: this replaces FlashAttention-3 (Hopper). FA4 is Python/CuTeDSL-based
# (JIT at runtime, no nvcc build). Transformer Engine 2.16 can select FA4
# directly when this package is installed and training uses --attention-backend flash.
#
# Pins that matter:
#   flash-attn-4 == 4.0.0b17
#   nvidia-cutlass-dsl == 4.5.2   (quack 0.5.0 needs ThrMma, REMOVED in 4.6.0.dev0)
#   quack-kernels == 0.5.0

echo "Installing FlashAttention-4 ${PRETRAIN_FLASH_ATTENTION_4_VERSION} + CuTeDSL ${PRETRAIN_CUTLASS_DSL_VERSION}"

uv pip install --no-config --python "${PY}" --index-strategy unsafe-best-match \
  --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple \
  "flash-attn-4==${PRETRAIN_FLASH_ATTENTION_4_VERSION}" \
  "nvidia-cutlass-dsl[${PRETRAIN_CUDA_INDEX}]==${PRETRAIN_CUTLASS_DSL_VERSION}" \
  "quack-kernels==${PRETRAIN_QUACK_VERSION}"

# Relax the FA4 sm-arch guard so sm_100/sm_103 both pass (no-op on sm_100).
"${PY}" - <<PY
from pathlib import Path
sp = Path("${TARGET_DIR}/venv/lib/python${PRETRAIN_PYTHON_VERSION}/site-packages")
p = sp / "flash_attn/cute/flash_fwd_sm100.py"
if p.exists():
    s = p.read_text()
    old = 'assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"'
    new = 'assert self.arch.major in (10, 11), "Only SM 10.x and 11.x are supported"'
    if old in s:
        p.write_text(s.replace(old, new, 1)); print("patched FA4 sm guard")
    else:
        print("FA4 guard anchor not present (ok for sm_100)")
else:
    print("FA4 flash_fwd_sm100.py not found (skipped)")
PY
