#!/bin/bash

# code
#cat /data/llm-jp-corpus/v3.1.0/gitlab/code/code_stack/train_*.jsonl.gz > corpus/json/code/code_stack_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/olmo-mix-1124/data/starcoder/v1-decon-100_to_20k-2star-top_token_030/documents/*json.gz > corpus/json/code/code_olmo-starcoder_0000.jsonl.gz

# en
#cat /data/llm-jp-corpus/v3.1.0/gitlab/en/en_wiki/train_*.jsonl.gz > corpus/json/en/en_wiki_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/dolma-v1_7/books-*.json.gz > corpus/json/en/en_dolma-books_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/dolma-v1_7/wiki-*.json.gz > corpus/json/en/en_dolma-wiki_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/dolma-v1_7/pes2o-*.json.gz > corpus/json/en/en_dolma-pes2o_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/dolma-v1_7/reddit-*.json.gz > corpus/json/en/en_dolma-reddit_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/olmo-mix-1124/data/arxiv/train/*.json.gz > corpus/json/en/en_olmo-arxiv_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/olmo-mix-1124/data/open-web-math/train/*.json.gz > corpus/json/en/en_olmo-openwebmath_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/olmo-mix-1124/data/algebraic-stack/train/*.json.gz > corpus/json/en/en_olmo-algebraicstack_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/en_math_pile/*_train.jsonl.gz > corpus/json/en/en_mathpile_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/en_gsm8k/*.jsonl.gz > corpus/json/en/en_gsm8k_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/en_math/*.jsonl.gz > corpus/json/en/en_math_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/en_finemath/finemath-4plus/train.jsonl.gz > corpus/json/en/en_finemath-4plus_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/download/dolmino-mix-1124/data/stackexchange/*.json.gz > corpus/json/en/en_dolmino-stackexchange_0000.jsonl.gz

#for d in $(ls /data/llm-jp-corpus/v4.0.0/sample/en_fineweb); do
#    cat /data/llm-jp-corpus/v4.0.0/sample/en_fineweb/$d/*.jsonl.gz > corpus/json/en/en_fineweb-rest_$d.jsonl.gz
#done

#for d in $(ls /data/llm-jp-corpus/v4.0.0/sample/en_fineweb-edu-score-2/original); do
#    cat /data/llm-jp-corpus/v4.0.0/sample/en_fineweb-edu-score-2/original/$d/*.jsonl.gz > corpus/json/en/en_fineweb-eduscore2_$d.jsonl.gz
#done

# ja
#cat /data/llm-jp-corpus/v3.1.0/gitlab/ja/ja_wiki/train_*.jsonl.gz > corpus/json/ja/ja_wiki_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/ja_fineweb-2/000_*.jsonl.gz > corpus/json/ja/ja_fineweb2_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/ja_fineweb-2/001_*.jsonl.gz > corpus/json/ja/ja_fineweb2_0001.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/ja_fineweb-2/002_*.jsonl.gz > corpus/json/ja/ja_fineweb2_0002.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/ja_fineweb-2/003_*.jsonl.gz > corpus/json/ja/ja_fineweb2_0003.jsonl.gz

# zh
#cat /data/llm-jp-corpus/v3.1.0/gitlab/zh/zh_wiki/train_*.jsonl.gz > corpus/json/zh/zh_wiki_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/zh_fineweb-2/000_*.jsonl.gz > corpus/json/zh/zh_fineweb2_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/zh_fineweb-2/001_*.jsonl.gz > corpus/json/zh/zh_fineweb2_0001.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/zh_fineweb-2/002_*.jsonl.gz > corpus/json/zh/zh_fineweb2_0002.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/zh_fineweb-2/003_*.jsonl.gz > corpus/json/zh/zh_fineweb2_0003.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/zh_fineweb-2/004_*.jsonl.gz > corpus/json/zh/zh_fineweb2_0004.jsonl.gz

# ko
#cat /data/llm-jp-corpus/v3.1.0/gitlab/ko/ko_wiki/train_*.jsonl.gz > corpus/json/ko/ko_wiki_0000.jsonl.gz
#cat /data/llm-jp-corpus/v4.0.0/sample/ko_fineweb-2/*.jsonl.gz > corpus/json/ko/ko_fineweb2_0000.jsonl.gz
