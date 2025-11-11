# Installs flash attention 3 (flash attention for NVIDIA Hopper architecture).

# CAUTION(sosuke):
# Installing flash attention v2 and v3 in the same environment may cause problems when used with Megatron-LM.
# We highly recommend only to use flash attention v3 for Hopper architecture.

echo "Installing Flash Attention ${PRETRAIN_FLASH_ATTENTION_VERSION}"
source ${TARGET_DIR}/venv/bin/activate

pushd ${TARGET_DIR}/src

git clone https://github.com/Dao-AILab/flash-attention.git
pushd flash-attention/
git checkout ${PRETRAIN_FLASH_ATTENTION_VERSION}

# Use flash-attention 3
pushd hopper/

python setup.py install

python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flash_attn_3
wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/${PRETRAIN_FLASH_ATTENTION_VERSION}/hopper/flash_attn_interface.py

popd # hopper/
popd # flash-attention/
popd # ${TARGET_DIR}/src

deactivate
