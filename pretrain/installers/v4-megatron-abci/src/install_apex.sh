# Install 

echo "Installing apex with commit ${PRETRAIN_APEX_COMMIT}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

git clone --recurse-submodules https://github.com/NVIDIA/apex
pushd apex

# Checkout the specific commit
git checkout ${PRETRAIN_APEX_COMMIT}
git submodule update --init --recursive


python -m pip install \
  -v \
  --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  ./
popd

popd  # ${TARGET_DIR}/src
deactivate
