# Finalize: re-pin cuBLAS LAST and print the installed stack.
#
# Earlier dependency resolutions can drag in an older cuBLAS that breaks TE 2.16,
# so we reinstall the pinned version after everything else.

echo "Re-pinning cuBLAS ${PRETRAIN_CUBLAS_VERSION}"
uv pip install --no-config --python "${PY}" \
  --reinstall-package nvidia-cublas "nvidia-cublas==${PRETRAIN_CUBLAS_VERSION}"

echo "=== installed versions ==="
"${PY}" - <<'PY'
import importlib.metadata as m
for p in ["torch","transformer_engine","transformer_engine_cu13","transformer_engine_torch",
          "flash-attn-4","nvidia-cutlass-dsl","quack-kernels","apex","megatron-core",
          "nvidia-cublas","transformers"]:
    try: print(f"  {p:26s} {m.version(p)}")
    except Exception: print(f"  {p:26s} NOT FOUND")
PY
