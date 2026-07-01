# Installs flash attention 3 (flash attention for NVIDIA Hopper architecture).

echo "Installing Flash Attention ${PRETRAIN_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

pushd ${TARGET_DIR}/src

git clone https://github.com/Dao-AILab/flash-attention.git -b v${PRETRAIN_FLASH_ATTENTION_VERSION} --recursive

pushd flash-attention
# install v2
python setup.py install
pushd hopper
# install v3
python setup.py install
python_path=$(python -c "import site; print(site.getsitepackages()[0])")
cp ./flash_attn_interface.py ${python_path}/flash_attn_3
popd
popd

popd # ${TARGET_DIR}/src

deactivate
