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

```shell
git clone https://github.com/llm-jp/llm-jp-scripts
cp -a llm-jp-scripts/installers/v3-megatron-sakura ~/myspace
cd ~/myspace # ワーキングディレクトリ

pyenv local 3.10.14
which python # Pythonのパスを確認
python --version # Pythonのバージョンを確認

python -m venv venv
source venv/bin/activate

bash install.sh
# ワーキングディレクトリにツール一式がインストールされます。時間がかかるので気長に待って下さい。
```

