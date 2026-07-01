# Install prerequisite Python packages.

echo "Installing requirements"

uv pip install --no-config --python "${PY}" -U -r "${SCRIPT_DIR}/src/requirements.txt"

# Optional: NVIDIA ModelOpt (quantization). Best-effort — not fatal if unavailable.
uv pip install --no-config --python "${PY}" --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com nvidia-modelopt || true
