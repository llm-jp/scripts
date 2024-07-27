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

# Optional
# Python 3.10.14 がインストールされていない場合、 install.sh が自動でインストールします。
pyenv install 3.10.14
```

### Build

```shell
git clone https://github.com/llm-jp/llm-jp-pretrain-scripts

# ~/myspace に環境をインストールします。
# 時間がかかるので気長に待って下さい。
bash llm-jp-pretrain-scripts/pretrain/installers/v3-megatron-sakura/install.sh ~/myspace 2>&1 | tee myspace_install.log
```

