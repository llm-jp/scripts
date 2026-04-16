# Create venv environment

echo "Creating venv..."
pushd ${TARGET_DIR}
${PRETRAIN_SYSTEM_PYTHON} -m venv venv

source venv/bin/activate
python -m pip install --no-cache-dir -U pip setuptools wheel
deactivate

popd  # ${TARGET_DIR}
