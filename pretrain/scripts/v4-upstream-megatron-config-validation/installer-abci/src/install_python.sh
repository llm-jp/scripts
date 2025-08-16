# Script to install Python to TARGET_DIR
#
# This script will make the following directories:
#   * ${TARGET_DIR}/src/cpython ... Source of Python
#   * ${TARGET_DIR}/python ... installed Python binary

echo "Installing Python ${PRETRAIN_PYTHON_VERSION}"
pushd ${TARGET_DIR}/src

git clone https://github.com/python/cpython -b v${PRETRAIN_PYTHON_VERSION}
pushd cpython
./configure --prefix="${TARGET_DIR}/python" --enable-optimizations
make -j 64
make install
popd  # cpython

popd  # ${TARGET_DIR}/src
