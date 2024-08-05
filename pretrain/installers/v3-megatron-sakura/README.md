# Megatron installation script for Sakura/LLM-jp v3 models

LLM-jp v3モデルを作成するためのMegatron環境をSakuraクラスタにインストールするためのスクリプトです。
System Pythonやpyenvに依存しない閉じた環境を指定したディレクトリ上に構築します。

## Usage

### Build

インストール処理のためにSakuraクラスタのCPUノードを1個使用します。
時間がかかるので気長に待って下さい。

```shell
git clone https://github.com/llm-jp/scripts
cd scripts/pretrain/installers/v3-megatron-sakura

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
    requirements.txt      venvに事前インストールされたライブラリ一覧
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
