# Installs Megatron-LM.

echo "Installing Megatron-LM ${PRETRAIN_MEGATRON_TAG}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

# download our Megatron and build helper library
git clone https://github.com/llm-jp/Megatron-LM -b ${PRETRAIN_MEGATRON_TAG}
pushd Megatron-LM
pushd megatron/core/datasets

# NOTE(odashi):
# Original makefile in the above directory uses the system's (or pyenv's) python3-config.
# But we need to invoke python3-config installed on our target directory.
MEGATRON_HELPER_CPPFLAGS=(
  -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
  $(python -m pybind11 --includes)
)
MEGATRON_HELPER_EXT=$(${TARGET_DIR}/python/bin/python3-config --extension-suffix)

# NOTE(odashi):
# New version of Megatron-LM changed the extension name 'helpers' to 'helpers_cpp'
#g++ ${MEGATRON_HELPER_CPPFLAGS[@]} helpers.cpp -o helpers_cpp${MEGATRON_HELPER_EXT}
g++ ${MEGATRON_HELPER_CPPFLAGS[@]} helpers.cpp -o helpers${MEGATRON_HELPER_EXT}

popd  # megatron/core/datasets
popd  # Megatron-LM

popd  # ${TARGET_DIR}/src
deactivate
