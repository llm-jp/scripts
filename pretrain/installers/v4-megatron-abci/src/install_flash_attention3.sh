# Ref: https://github.com/llm-jp/scripts/blob/exp/tokenizer_test/experiments/v4-hq_tokenizer_test/installer/install_megatron.sh

echo "Installing FlashAttention ${PRETRAIN_FLASH_ATTENTION_VERSION} (commit ${PRETRAIN_FLASH_ATTENTION_COMMIT})"
source "${TARGET_DIR}/venv/bin/activate"
pushd "${TARGET_DIR}/src"

git clone https://github.com/Dao-AILab/flash-attention.git
pushd flash-attention
git checkout "${PRETRAIN_FLASH_ATTENTION_COMMIT}"
pushd hopper # cd hopper/
export TORCH_CUDA_ARCH_LIST="90"
python setup.py install

python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flash_attn_3
wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/${PRETRAIN_FLASH_ATTENTION_COMMIT}/hopper/flash_attn_interface.py

popd  # flash-attention/hopper
popd  # flash-attention
popd  # ${TARGET_DIR}/src
deactivate
