# llm-jp-eval v1.3.1 installation and execution script

llm-jp-eval の v1.3.1 で評価するためスクリプト
環境構築のためのスクリプト・評価実行のためのスクリプトを含みます

## Usage

### Build

インストール処理のためにCPUノードを1個使用します。

1. リポジトリのクローン
  ```shell
  git clone https://github.com/llm-jp/scripts
  cd scripts/evaluation/installers/llm-jp-eval-v1.3.1
  ```

2. インストール
指定したディレクトリ（`~/myspace`）下に環境構築用ディレクトリ (`~/myspace/environment`) が作成されます
- `<env-name>`には環境名(llm-jp, llm-jp-nvlink, sakura, etc)を入力してください。
  - scripts/envs以下にあるフォルダ名が選択可能です。
```shell
# For a cluster with SLURM
sbatch --partition {partition} install.sh <env-name> ~/myspace
# For a cluster without SLURM
bash install.sh <env-name> ~/myspace > logs/install.out 2> logs/install.err
```

3. (Optional) wandb, huggingface の設定
```shell
# Additional process for a cluster with SLURM
srun --partition {partition} --nodes 1 --pty bash
```
```shell
cd ~/myspace
source environment/venv/bin/activate
wandb login
huggingface-cli login
```
```shell
# Additional process for a cluster with SLURM
exit
```

### Contents in installed directory (~/myspace)

インストール終了後、下記のディレクトリ構造が構築されます。

```
~/myspace/
    run_llm-jp-eval.sh    評価を実行するスクリプト
    logs/                 SLURM用ログ保存ディレクトリ
    resources/
        config_base.yaml  評価実行時に読み込む設定ファイルのテンプレート
    environment/
        installer_envvar.log  インストール開始後に記録した環境変数の一覧
        install.sh            使用したインストールスクリプト
        dataset/llm-jp-eval   llm-jp-eval評価用データセット
        python/               Python実行環境
        scripts/              各種の環境設定用スクリプト
        src/                  個別ダウンロードされたライブラリ
        venv/                 Python仮想環境 (python/ にリンク)
```

### Evaluation
必要に応じて`run_llm-jp-eval.sh`・`resources/config_base.yaml`内の変数を書き換えてください
 - tokenizer・wandb entity・wandb projectを変更する場合`run_llm-jp-eval.sh`のみの変更で対応可能
 - その他の変更を行う場合、`resources/config_base.yaml`を変更した上で、`run_llm-jp-eval.sh`内でファイルを指定
```shell
cd ~/myspace
# (Optional) If you need to change variables
cp resources/config_base.yaml resources/config_custom.yaml
cp run_llm-jp-eval.sh run_llm-jp-eval_custom.sh
# Set `resources/config_custom.yaml` in run_llm-jp-eval_custom.sh

# For a cluster with SLURM
sbatch --partition {partition} run_llm-jp-eval.sh {path/to/model} {wandb.run_name}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES={num} bash run_llm-jp-eval.sh {path/to/model} {wandb.run_name}
```

#### Sample code
```shell
# For a cluster with SLURM
sbatch --partition {partition} run_llm-jp-eval.sh llm-jp/llm-jp-13b-v2.0 test-$(whoami)
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES=0 bash run_llm-jp-eval.sh llm-jp/llm-jp-13b-v2.0 test-$(whoami)
```

## resources/sha256sums.csv の作成コマンド
```shell
TARGET_DIR={path/to/dataset/directory/containing/json/files}
find $TARGET_DIR -type f | xargs -I{} sh -c 'echo -e "$(basename {})\t$(sha256sum {} | awk "{print \$1}")"'
```
