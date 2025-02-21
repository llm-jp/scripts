#!/bin/bash

pushd ./environments/train/src
git clone git@github.com:llm-jp/Megatron-LM Megatron-LM_tokenize -b llmjp0-mdx
popd