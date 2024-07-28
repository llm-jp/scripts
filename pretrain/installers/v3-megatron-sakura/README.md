# Megatron installation script for Sakura/LLM-jp v3 models

LLM-jp v3モデルを作成するためのMegatron環境をSakuraクラスタにインストールするためのスクリプトです。

## Usage

### Prerequisites

pyenv上にPython 3.10.14をインストールしてください。ここから派生させたvenv上に環境を構築します。

```shell
curl https://pyenv.run | bash

echo 'export PATH="/home/$(whoami)/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

exec bash

pyenv install 3.10.14
```

### Build

インストール処理のためにSakuraクラスタのCPUノードを1個使用します。
時間がかかるので気長に待って下さい。

```shell
git clone https://github.com/llm-jp/llm-jp-scripts
cd llm-jp-scripts/pretrain/installers/v3-megatron-sakura

# ~/myspace に環境をインストールします。
sbatch install.sh ~/myspace
```

