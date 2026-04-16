# Install Megatron-LM

echo "Installing Megatron-LM ${PRETRAIN_MEGATRON_TAG}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

git clone https://github.com/llm-jp/Megatron-LM -b ${PRETRAIN_MEGATRON_TAG}
pushd Megatron-LM
pushd megatron/core/datasets

MEGATRON_HELPER_CPPFLAGS=(
  -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
  $(python -m pybind11 --includes)
)
# SYNKA 使用 conda+venv，python3-config 在 venv/bin 下，直接用即可
MEGATRON_HELPER_EXT=$(python3-config --extension-suffix)

g++ ${MEGATRON_HELPER_CPPFLAGS[@]} helpers.cpp -o helpers_cpp${MEGATRON_HELPER_EXT}

popd  # megatron/core/datasets
popd  # Megatron-LM

popd  # ${TARGET_DIR}/src
deactivate
