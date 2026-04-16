# Install NVIDIA Apex

echo "Installing apex with commit ${PRETRAIN_APEX_COMMIT}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

if [ -d "apex" ] && git -C apex rev-parse --git-dir > /dev/null 2>&1; then
    echo "apex directory already exists and is a valid git repo, skipping clone"
else
    rm -rf apex
    git clone --recurse-submodules https://github.com/NVIDIA/apex
fi
pushd apex

git checkout ${PRETRAIN_APEX_COMMIT}
git submodule update --init --recursive

# Apex の setup.py は nvcc バージョン（12.8）と PyTorch ビルド CUDA（12.6）を比較し、
# マイナーバージョン差でエラーになる場合がある。CUDA 12.x 内は ABI 互換のためスキップ。
# 参考: https://github.com/NVIDIA/apex/pull/323#discussion_r287021798
sed -i 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/# check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)  # patched: skip CUDA version check/' setup.py

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
