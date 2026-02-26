# Megatron-LM Installer for LLM-jp v4 MoE on ABCI

## Usage

ABCI 3.0上で以下のコマンドを実行し、`<env_install_path>`に環境を構築できる

```bash
cd pretrain/installers/v4-moe-upstream-megatron-abci/
bash run_setup.sh <env_install_path>
```

環境構築後、以下のコマンドで環境を有効化できる

```bash
source <env_install_path>/scripts/environment.sh  # Load environment variables and modules
source <env_install_path>/venv/bin/activate       # Activate Python virtual environment
```

各種ライブラリのバージョンに関しては `scripts/environment.sh` を参照
