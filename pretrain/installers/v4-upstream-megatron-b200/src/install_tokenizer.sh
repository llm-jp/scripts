# Install the LLM-jp tokenizer (cloned into the env for training scripts to use).

echo "Installing LLM-jp Tokenizer ${PRETRAIN_TOKENIZER_TAG}"

git clone https://github.com/llm-jp/llm-jp-tokenizer -b "${PRETRAIN_TOKENIZER_TAG}" \
  "${TARGET_DIR}/src/llm-jp-tokenizer"
