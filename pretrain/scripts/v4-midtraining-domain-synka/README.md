# v4 Training Template on SYNKA

SYNKA 上で Megatron-LM を利用した LLM-jp v4 用の学習スクリプトのテンプレート

## Usage

以下では、SYNKA 上での環境構築と学習の実行方法を説明する。
`$EXP_DIR` は、事前に作成した実験ディレクトリを指すものとする (e.g. `/data/llmjp-pj/experiments/9999_template`)。

### Environment Setup

まず、`$EXP_DIR` 内に `llm-jp/scripts` リポジトリをクローンする

```bash
cd $EXP_DIR
git clone https://github.com/llm-jp/scripts.git
```

次に、 [pretrain/scripts/v4-midtraining-domain-synka/installer](installer/README.md) を利用し、`$EXP_DIR/env` に環境を構築する。

```bash
cd $EXP_DIR/scripts/pretrain/scripts/v4-midtraining-domain-synka/installer
sbatch sbatch_setup.sh $EXP_DIR/env
```

実行するとジョブスケジューラにジョブが投入される。
投入されたジョブが完了すると、`$EXP_DIR/env` に環境が構築される。

### Run Training

#### 1. タスクディレクトリの作成

まず、 `task_template/` ディレクトリを `$EXP_DIR/tasks/$TASK_NAME` にコピーする。
`$TASK_NAME` は、任意の実験のタスク名を指定する。

```bash
mkdir -p $EXP_DIR/tasks
cd $EXP_DIR/scripts
cp -r pretrain/scripts/v4-midtraining-domain-synka/task_template $EXP_DIR/tasks/$TASK_NAME
```

タスクディレクトリは学習の設定を定義する。
詳しい設定方法は "Training Configuration" セクションを参照。

#### 2. 学習の実行

次に、以下のコマンドで学習を実行する。

```bash
cd $EXP_DIR/scripts
sbatch --nodes <NUM_NODES> pretrain/scripts/v4-midtraining-domain-synka/pretrain/sbatch_train.sh <EXPERIMENT_DIR> <TASK_NAME> <WANDB_PROJECT> 
# Example:
sbatch --nodes 2 pretrain/scripts/v4-midtraining-domain-synka/pretrain/sbatch_train.sh /data/llmjp-pj/experiments/0340_v4_domain_common v4-8b-mid-phase3 0340_domain_common
```

CLIからは以下の引数を指定する

- `<NUM_NODES>`: 使用するノード数 (e.g. `32`)
- `<EXPERIMENT_DIR>`: 実験ディレクトリのパス (e.g. `/path/to/0123_experiment`)
- `<TASK_NAME>`: タスクディレクトリ名 (e.g. `task_name`)
- `<WANDB_PROJECT>`: WandB に記録するプロジェクト名 (e.g. `0123_experiment`)

### Training Configuration

タスクディレクトリ内には以下のようなファイルが含まれる。

- `configure_corpus.py`: 学習データのパス及び利用するトークン数などを定義するスクリプト
  - Megatron-LM の `--train-data` 引数に渡す値をこのファイル内の `$TRAIN_DATA_PATH` 変数に定義する
- `params.sh`: モデルハイパーパラメータ、optimizer設定、学習器の各種設定などを定義するスクリプト
  - Megatron-LM の `pretrain_gpt.py` に渡す引数をこのファイル内の変数に定義する
- `train_data.sh`: `configure_corpus.py` を実行して学習データの bash 配列を作成するスクリプト．基本的に触る必要なし
