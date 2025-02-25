# Script to install Python to TARGET_DIR
#
# This script will make the following directories:
#   * ${TARGET_DIR}/venv ... venv directory inherited from the above Python binary

echo "Setup venv"
pushd ${TARGET_DIR}

python/bin/python3 -m venv venv

source venv/bin/activate
python -m pip install --no-cache-dir -U pip
deactivate

popd  # ${TARGET_DIR}
