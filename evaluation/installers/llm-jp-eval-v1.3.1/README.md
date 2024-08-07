# Megatron installation script for Sakura/LLM-jp v3 models

llm-jp-eval の v1.3.1 で評価するための環境インストールします

## Usage

### Build

インストール処理のためにCPUノードを1個使用します。

```shell
git clone https://github.com/llm-jp/scripts
cd scripts/evaluation/installers/llm-jp-eval-v1.3.1

# ~/myspace に環境をインストールします。
sbatch install.sh ~/myspace
```

### Check

インストール終了後、下記のようなディレクトリ構造が構築されています。

```
~/myspace/
    example/              サンプルスクリプト
    installer_envvar.log  インストール開始後に記録した環境変数の一覧
    install.sh            使用したインストールスクリプト
    python/               Python実行環境
    scripts/              各種の環境設定用スクリプト
    src/                  個別ダウンロードされたライブラリ
    venv/                 Python仮想環境 (python/ にリンク)
```

インストールした環境で正常に事前学習ジョブを起動できるかどうかを確認します。

```shell
cd ~/myspace

# デフォルトでは1ノードを専有し、GPUを8枚全て使うジョブが起動します。
sbatch example/sbatch.sh

# W&Bにtrain lossが記録されるのを確認したらジョブを止めてください。
```

## hashs.tsv の作成コマンド
```shell
TARGET_DIR=<path/to/directory/name/of/json>
find $TARGET_DIR -type f | xargs -I{} sh -c 'echo -e "$(basename {})\t$(sha256sum {} | awk "{print \$1}")"'
```