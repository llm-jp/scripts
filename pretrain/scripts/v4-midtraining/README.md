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
# ./tasks/v4-dolmino-mix-1124/train_data.all.shを作成
# 自動的にtoken数を計算し、"token数 PATH"をtrain_data.all.shに書き込む
./preprocess/build_train_data.sh

# ./tasks/v4-dolmino-mix-1124/train_data.all.shから./tasks/v4-dolmino-mix-1124/train_data_50B.shを作成
# dolminoのmidtrainingと同じ配合の50Bのデータセットサイズになるようにtoken数を更新する
./preprocess/update_train_data_to_50B.sh
# 100B, 300Bも同様
```

## 環境構築

ref: [scripts/pretrain/installers/v4-megatron-abci at 0130-instruct-pretrain · llm-jp/scripts](https://github.com/llm-jp/scripts/tree/0130-instruct-pretrain/pretrain/installers/v4-megatron-abci)

```sh
bash run_setup.sh /path/to/target_dir
```

> [!CAUTION]
> Transformer engineのv1.10以上を使うとエラーが出るため、environment2を今回利用している（Transformer engineのversionを1.9にdowngradeした。）
> ref: https://docs.nvidia.com/nemo-framework/user-guide/24.07/knownissues.html

> [!CAUTION]
> `environment/src/Megatron-LM/megatron/core/dist_checkpointing/strategies/common.py`の72行目に"weights_only=False"を加えた
> ref: https://github.com/huggingface/accelerate/issues/3539


## job実行

```sh
cd /path/to/v4-midtraining

# example:
# 1.3b-llama3-ecjk
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 1.3b-llama3-ecjk 50B 16

# 7.7b-llama3-ecjk
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 7.7b-llama3-ecjk 50B 16
```

### [Option] 依存関係付きのjob実行

qsub の `-W depend=...` の機能を利用して、ジョブ間に依存関係をつけて実行するためのスクリプトを用意している。
`run_train.sh` ではなく `run_train_with_deps.sh` を利用して実行する。

```sh
# 最後の引数に `-W depend=` に渡す値を書く
bash midtrain/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 7.7b-llama3-ecjk 50B 16 afterok:xxxx.pbs1:yyyy.pbs1
```

依存関係の詳しい記法は ABCI 3.0 上で `man qsub` を参照すること

## Checkpoint変換

> [!CAUTION]
> 下のスクリプトを実行する前に、`scripts/pretrain/scripts/v4-midtraining/midtrain/params`の`--no-load-optim`を外してください。

```sh
cd /path/to/v4-midtraining

bash convert/convert_latest.sh {TASK_DIR} {PARAM_NAME} {DATASET_SIZE}

# example:
bash convert/convert_latest.sh $(realpath tasks/v4-dolmino-mix-1124) 1.3b-llama3-ecjk 50B
```

> [!CAUTION]
> `/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/environment2/src/Megatron-LM/tools/checkpoint/loader_mcore.py`の先頭に以下のコードを加えた
> ```
> import json, os, sys, torch, functools
> torch.load = functools.partial(torch.load, weights_only=False)
> ```

## Model soup

[arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) を利用して、モデルのマージを行う

モデルマージ用の環境は `$EXP_DIR/venv-mergekit` に用意した

```sh
source $EXP_DIR/venv-mergekit/bin/activate

# 初回にmergekitをインストール
pip install mergekit
```

`./merge/` 配下にマージの設定ファイルを配置している

merge実行コマンド

```sh
mergekit-yaml merge/your_config.yaml model/output/path/
```
