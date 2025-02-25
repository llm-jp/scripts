# Install 

echo "Installing apex ${PRETRAIN_APEX_VERSION}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

git clone --recurse-submodules https://github.com/NVIDIA/apex -b ${PRETRAIN_APEX_VERSION}
pushd apex
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
