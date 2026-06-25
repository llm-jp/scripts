# Install Megatron-LM (editable) and build the dataset C++ helpers.

echo "Installing Megatron-LM ${PRETRAIN_MEGATRON_TAG} from ${PRETRAIN_MEGATRON_REPO}"

MLM_SRC="${TARGET_DIR}/src/Megatron-LM"
git clone "${PRETRAIN_MEGATRON_REPO}" -b "${PRETRAIN_MEGATRON_TAG}" "${MLM_SRC}"

# Editable install of megatron-core. --no-deps/--no-build-isolation so it does
# not drag its own torch / TE / flash-attn over the versions we pinned above.
uv pip install --no-config --python "${PY}" --no-deps --no-build-isolation -e "${MLM_SRC}"

# Build the Megatron-core dataset C++ helpers with this venv's python on PATH.
# (New Megatron renamed the extension 'helpers' -> 'helpers_cpp'; the Makefile
# handles the current name.)
PATH="${TARGET_DIR}/venv/bin:${PATH}" make -C "${MLM_SRC}/megatron/core/datasets"
