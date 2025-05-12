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

### tokenize

```bash
export EXP_DIR="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/"
export EXP_SCRIPT_DIR="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining"
cd $EXP_DIR

# 1. Huggingfaceからdolmino-mix-1124をダウンロード
huggingface-cli download allenai/dolmino-mix-1124 --local-dir "$EXP_DIR/dolmino-mix-1124"

cd $EXP_SCRIPT_DIR
# 2. データセットの展開 (`$EXP_DIR/dolmino-mix-1124-extracted` に展開される)
bash ./preprocess/extract.sh

# 3. データセットファイルのmerge (`$EXP_DIR/dolmino-mix-1124-extracted-merged` に結合ファイルが作成される)
qsub ./preprocess/merge_files.sh

# (3が完了したら)
# 4. データセットのtokenize (`$EXP_DIR/dolmino-mix-1124-tokenized` にtokenizeされたファイルが作成される)
qsub ./preprocess/tokenize.sh

# (optional) 中間ファイルの削除
rm -rf $EXP_DIR/dolmino-mix-1124-extracted $EXP_DIR/dolmino-mix-1124-extracted-merged
```

### データセットの作成

データセットの作成前に事前にtokenizeが完了している必要がある。

```sh
# ./tasks/v4-dolmino-mix-1124/train_data.all.shを作成する
# 自動的にtoken数を計算し、"token数 PATH"をtrain_data.all.shに書き込む
./preprocess/build_train_data.sh

# ./tasks/v4-dolmino-mix-1124/train_data.all.shから./tasks/v4-dolmino-mix-1124/train_data.shを作成する
# dolminoのmidtrainingと同じ配合の50Bのデータセットサイズになるようにtoken数を更新する
./preprocess/update_train_data_to_50B.sh
```

## 環境構築

ref: [scripts/pretrain/installers/v4-megatron-abci at 0130-instruct-pretrain · llm-jp/scripts](https://github.com/llm-jp/scripts/tree/0130-instruct-pretrain/pretrain/installers/v4-megatron-abci)

```sh
bash run_setup.sh /path/to/target_dir
```

> [!NOTE]
> environment2を今回は利用している（Transformer engineのversionを1.9にdowngradeした。）
> ref: https://docs.nvidia.com/nemo-framework/user-guide/24.07/knownissues.html

## tokenize

FILL LATER

## job実行

```sh
cd /path/to/v4-midtraining

# example:
# 1.3b-llama3-ecjk
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 1.3b-llama3-ecjk 16

# 7.7b-llama3-ecjk
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 7.7b-llama3-ecjk 16
```

## Checkpoint変換

```sh
cd /path/to/v4-midtraining

bash convert/convert_latest.sh $(realpath tasks/v4-dolmino-mix-1124) {PARAM_NAME} {ITER}

# example:
bash convert/convert_latest.sh $(realpath tasks/v4-dolmino-mix-1124) 1.3b-llama3-ecjk 1
```
