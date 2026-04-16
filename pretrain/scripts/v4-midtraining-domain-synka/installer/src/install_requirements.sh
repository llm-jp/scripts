# Installs prerequisite packages

echo "Installing requirements"

source ${TARGET_DIR}/venv/bin/activate

python -m pip install --no-cache-dir -U -r ${SCRIPT_DIR}/src/requirements.txt

deactivate
