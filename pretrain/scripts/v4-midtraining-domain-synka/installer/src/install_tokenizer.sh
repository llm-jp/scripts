# Install LLM-jp Tokenizer.

echo "Installing LLM-jp Tokenizer ${PRETRAIN_TOKENIZER_COMMIT}"
pushd ${TARGET_DIR}/src

git clone https://github.com/llm-jp/llm-jp-tokenizer
pushd llm-jp-tokenizer
git checkout ${PRETRAIN_TOKENIZER_COMMIT}

popd
popd  # ${TARGET_DIR}/src