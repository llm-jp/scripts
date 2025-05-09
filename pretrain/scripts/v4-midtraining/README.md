# LLMjp-v4 Midtraining

## Overview

OLMo2におけるMidtrainingをLL-jp-4-enのモデルで再現する実験を行う

## データセットの割合

合計Token数: 55,797,411,281 tokens

| Datasets      | Tokens         | Source(%) | Mix(%) | Original OLMo2 Mix (%) |
|---------------|----------------|-----------|--------|------------------------|
| DCLM          | 26,540,912,669 | 3.23%     | 47.57% | 47.20%                 |
| FLAN          | 9,242,742,021  | 50.00%    | 16.56% | 16.60%                 |
| peS2o         | 3,236,969,300  | 5.15%     | 5.80%  | 5.85%                  |
| Wikipedia     | 3,896,965,449  | 100.00%   | 6.98%  | 7.11%                  |
| Stackexchange | 1,464,772,187  | 100.00%   | 2.63%  | 2.45%                  |
| Math          | 11,415,049,655 | 100.00%   | 20.46% | 20.80%                 |





## 環境構築

ref: [scripts/pretrain/installers/v4-megatron-abci at 0130-instruct-pretrain · llm-jp/scripts](https://github.com/llm-jp/scripts/tree/0130-instruct-pretrain/pretrain/installers/v4-megatron-abci)

```sh
bash run_setup.sh /path/to/target_dir
```

## tokenize

TODO

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
