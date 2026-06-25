# Create the Python virtual environment with uv.
#
# NOTE: ABCI builds CPython from source (install_python.sh) because the host has
# no suitable interpreter. On this cluster `uv` manages Python 3.12 directly, so
# we create the venv with uv instead of a from-source CPython build.

echo "Setup venv (python ${PRETRAIN_PYTHON_VERSION}) at ${TARGET_DIR}/venv"

uv venv --allow-existing --python "${PRETRAIN_PYTHON_VERSION}" "${TARGET_DIR}/venv"

# uv venv ships no pip; TE's no-build-isolation installs below need real pip.
uv pip install --no-config --python "${PY}" -U pip setuptools wheel
