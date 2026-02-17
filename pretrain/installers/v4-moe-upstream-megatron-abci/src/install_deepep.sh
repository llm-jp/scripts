# Installs DeepEP.

echo "Installing DeepEP ${PRETRAIN_DEEPEP_VERSION}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

# download DeepEP and build helper library
git clone https://github.com/deepseek-ai/DeepEP
pushd DeepEP
git checkout ${PRETRAIN_DEEPEP_VERSION}
sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh
python -m pip install --no-build-isolation .

popd  # DeepEP

popd  # ${TARGET_DIR}/src
deactivate
