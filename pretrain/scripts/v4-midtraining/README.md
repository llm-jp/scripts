# LLMjp-v4 Midtraining

## Overview

OLMo2におけるMidtrainingをLL-jp-4-enのモデルで再現する実験を行う

## 環境構築

ref: [scripts/pretrain/installers/v4-megatron-abci at 0130-instruct-pretrain · llm-jp/scripts](https://github.com/llm-jp/scripts/tree/0130-instruct-pretrain/pretrain/installers/v4-megatron-abci)

```sh
bash run_setup.sh /path/to/target_dir
```

## tokenize

```sh
```

## job実行

```sh
cd /path/to/v4-midtraining

# 1.3b-count1
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 1.3b-count1 16

# 7.7b-count1
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 7.7b-count1 16
```

## Checkpoint変換

```sh
cd /path/to/v4-midtraining

bash convert/convert_latest.sh $(realpath tasks/v4-dolmino-mix-1124) {PARAM_NAME} {ITER}
```
