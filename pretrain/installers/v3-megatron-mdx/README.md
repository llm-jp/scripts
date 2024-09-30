# Megatron installation script for mdx/LLM-jp v3 models

LLM-jp v3モデルを作成するためのMegatron環境をmdxクラスタにインストールするためのスクリプトです。
System Pythonやpyenvに依存しない閉じた環境を指定したディレクトリ上に構築します。

## Usage

### Build

インストール処理のためにCPUリソースを使用します。
時間がかかるので気長に待って下さい。
SLRUMの入っていないクラスタでは`sbatch`を`bash`で読み替えてください。

```shell
git clone https://github.com/llm-jp/scripts
cd scripts/pretrain/installers/v3-megatron-mdx
cp ../v3-megatron-sakura/{install.sh,requirements.txt} .

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

# デフォルトではGPUを8枚全て使うジョブが起動します。
sbatch example/mpi_wrapper.sh

# W&Bにtrain lossが記録されるのを確認したらジョブを止めてください。
```
