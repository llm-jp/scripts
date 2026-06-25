#!/usr/bin/env bash
# Validate the native Megatron-LM B200 env on a real GPU: imports + CUDA + a tiny
# FA4 kernel. Run on a gpu node:
#   TARGET_DIR=<dir> srun -p gpu --gres=gpu:1 -c 16 bash verify/verify_env.sh
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-${1:-}}"
[ -n "${TARGET_DIR}" ] || { >&2 echo "Usage: TARGET_DIR=<dir> bash verify_env.sh"; exit 1; }
PY="${TARGET_DIR}/venv/bin/python"

export NVTE_FRAMEWORK=pytorch
export NVTE_FUSED_ATTN=0   # we use FA4, not TE's cuDNN fused attn

# torch-bundled cu13 libs + the wheel nvidia/ dir on the loader path.
LD_EXTRA=$("${PY}" - <<'PY'
import os, glob, torch
sp = os.path.dirname(os.path.dirname(torch.__file__))
d = glob.glob(os.path.join(sp, "nvidia", "*", "lib")); d.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
print(":".join(d))
PY
)
export LD_LIBRARY_PATH="${LD_EXTRA}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${TARGET_DIR}/venv/lib/python3.12/site-packages/nvidia"
export CUDA_PATH="${CUDA_HOME}"

echo "=== host=$(hostname) $(date) ==="
nvidia-smi -L
"${PY}" - <<'PY'
import torch
print("torch     ", torch.__version__, "| cuda", torch.version.cuda, "| arch", torch.cuda.get_arch_list())
print("cuda avail", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
import transformer_engine as te
print("TE        ", te.__version__)
import transformer_engine.pytorch as tep  # noqa: F401
print("TE.pytorch import OK")
import apex, fused_weight_gradient_mlp_cuda  # noqa: F401
print("APEX      + fused_weight_gradient_mlp_cuda OK")
import megatron.core as mcore
print("megatron-core", mcore.__version__)
from megatron.core.datasets import helpers_cpp  # noqa: F401
print("megatron dataset helpers_cpp OK")
from flash_attn.cute import flash_attn_func
import cutlass
print("FA4       flash_attn.cute.flash_attn_func OK | cutlass", cutlass.__version__)
q = torch.randn(2, 1024, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 1024,  8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 1024,  8, 128, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, k, v, causal=True)
o = o[0] if isinstance(o, tuple) else o
torch.cuda.synchronize()
print("FA4 kernel ran on GPU -> out", tuple(o.shape), o.dtype)
print("=== ALL IMPORTS + CUDA + FA4 KERNEL OK ===")
PY
