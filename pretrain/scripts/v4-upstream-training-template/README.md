# v5 Training Template

ABCI 3.0 上で Megatron-LM を利用した LLM-jp v5 用の学習スクリプトのテンプレート

## Usage

以下では、ABCI 3.0 上での環境構築と学習の実行方法を説明する。
`$EXP_DIR` は、事前に作成した実験ディレクトリを指すものとする (e.g. `/home/ach17726fj/experiments/1234_pretraining_test/`)。

### Environment Setup

まず、`$EXP_DIR` 内に `llm-jp/scripts` リポジトリをクローンする

```bash
cd $EXP_DIR
git clone git@github.com:llm-jp/scripts.git
```

次に、 [pretrain/installers/v5-megatron-abci](../../installers/v5-megatron-abci/README.md) を利用し、`$EXP_DIR/env` に環境を構築する。

```bash
cd $EXP_DIR/scripts/pretrain/installers/v5-megatron-abci/
bash run_setup.sh $EXP_DIR/env
```

実行するとジョブスケジューラにジョブが投入される。
投入されたジョブが完了すると、`$EXP_DIR/env` に環境が構築される。

### Run Training

#### 1. タスクディレクトリの作成

まず、 `task_template/` ディレクトリを `$EXP_DIR/tasks/$TASK_NAME` にコピーする。
`$TASK_NAME` は、任意の実験のタスク名を指定する。

```bash
mkdir -p $EXP_DIR/tasks/
cp -r scripts/pretrain/task_template/ $EXP_DIR/tasks/$TASK_NAME
```

タスクディレクトリは学習の設定を定義する。
詳しい設定方法は "Training Configuration" セクションを参照。

#### 2. 学習の実行

次に、以下のコマンドで学習を実行する。

```bash
cd $EXP_DIR/scripts/pretrain/$TRAINING_SCRIPT_DIR/
bash run_train.sh <RESERVATION_ID> <EXPERIMENT_ID> <EXPERIMENT_DIR> <TASK_NAME> <WANDB_PROJECT> <NUM_NODES>

# Example:
bash run_train.sh R0123456789 0123 /path/to/0123_experiment task_name 0123_experiment 32
```

CLIからは以下の引数を指定する

- `<RESERVATION_ID>`: ABCI の予約キュー ID
- `<EXPERIMENT_ID>`: 実験の識別子 (e.g. `0123`)
- `<EXPERIMENT_DIR>`: 実験ディレクトリのパス (e.g. `/home/ach17726fj/experiments/0123_experiment`)
- `<TASK_NAME>`: タスクディレクトリ名 (e.g. `task_name`)
- `<WANDB_PROJECT>`: WandB に記録するプロジェクト名 (e.g. `0123_experiment`)
- `<NUM_NODES>`: 使用するノード数 (e.g. `32`)

### Training Configuration

タスクディレクトリ内には以下のようなファイルが含まれる。

- `params.sh`: モデルハイパーパラメータ、optimizer設定、学習器の各種設定などを定義するスクリプト
  - Megatron-LM の `pretrain_gpt.py` に渡す引数をこのファイル内の変数に定義する
- `train_data.sh`: 学習データのパス及び利用するトークン数などを定義するスクリプト
  - Megatron-LM の `--train-data` 引数に渡す値をこのファイル内の `$TRAIN_DATA_PATH` 変数に定義する
- `train_iters.txt`: 学習イテレーション数を定義するファイル
  - 学習するイテレーション数を記載し、他には何も記載しない
