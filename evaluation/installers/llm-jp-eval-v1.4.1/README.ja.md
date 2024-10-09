[English version](README.md)
# llm-jp-eval v1.4.1 インストールおよび実行スクリプト

llm-jp-eval v1.4.1で評価を行うためのスクリプトです。<br>
環境構築用スクリプトと評価実行用スクリプトを含みます。<br>

注意：このバージョンではDockerに未対応のため、Code Generation タスク (mbpp) はスキップされます。

## 使用方法

### ビルド

インストール処理ではCPUを使用します。

1. リポジトリのクローン
  ```shell
  git clone https://github.com/llm-jp/scripts
  cd scripts/evaluation/installers/llm-jp-eval-v1.4.1
  ```

2. インストール
指定したディレクトリ（`~/myspace`）に、環境構築用ディレクトリ (`~/myspace/environment`) が作成されます。
通信速度によりますが、少なくとも20分ほどかかります。
```shell
# For a cluster with SLURM
sbatch --partition {FIX_ME} install.sh ~/myspace
# For a cluster without SLURM
bash install.sh ~/myspace > logs/install.out 2> logs/install.err
```

3. (オプション) wandb, huggingface の設定
```shell
cd ~/myspace
source environment/venv/bin/activate
wandb login
huggingface-cli login
```

### インストール後のディレクトリ構成 (~/myspace)

インストール完了後、以下のディレクトリ構造が作成されます。

```
~/myspace/
    run_llm-jp-eval.sh         評価を実行するスクリプト
    logs/                      SLURM用ログ保存ディレクトリ
    resources/
        config_base.yaml       評価実行時に読み込む設定ファイルのテンプレート
    vllm_outputs/              vllm用の出力ディレクトリ
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
必要に応じて`run_llm-jp-eval.sh`・`resources/config_*.yaml`の変数を必要に応じて変更してください。
 - tokenizer・wandb entity・wandb projectを変更する場合、`run_llm-jp-eval.sh`のみの変更で対応可能です。
 - その他の変更を行う場合、`resources/config_*.yaml`を編集し、`run_llm-jp-eval.sh`内でファイルを指定してください。

VRAMはモデルサイズの2.5-3.5倍必要（例: 13B model -> 33GB-45GB）<br>
2GPU以上で評価を行う場合、`resources/config_offline_inference_vllm.yaml`の`tensor_parallel_size`の変更してください<br>
SLURM環境で実行する場合、デフォルトでは`--gpus 1`のため、クラスタの要件に合わせて`--mem`を適切に設定してください。
```shell
cd ~/myspace
# (Optional) If you need to change variables
cp resources/config_base.yaml resources/config_custom.yaml
cp run_llm-jp-eval.sh run_llm-jp-eval_custom.sh
# Set `resources/config_custom.yaml` in run_llm-jp-eval_custom.sh

# For a cluster with SLURM
sbatch --partition {FIX_ME} run_llm-jp-eval.sh {path/to/model} {wandb.run_name}
# For a cluster without SLURM
CUDA_VISIBLE_DEVICES={FIX_ME} bash run_llm-jp-eval.sh {path/to/model} {wandb.run_name}
```

#### サンプルコード
```shell
# Evaluate 70B model on a cluster with SLURM using H100 (VRAM: 80GB)
sbatch --partition {FIX_ME} --gpus 4 --mem 8G run_llm-jp-eval.sh sbintuitions/sarashina2-70b test-$(whoami)
# Evaluate 13B model on a cluster without SLURM using A100 (VRAM: 40GB)
CUDA_VISIBLE_DEVICES=0,1 bash run_llm-jp-eval.sh llm-jp/llm-jp-13b-v2.0 test-$(whoami)
```

## 開発者向け: resources/sha256sums.csv の作成コマンド
```shell
TARGET_DIR={path/to/dataset/directory/containing/json/files}
find $TARGET_DIR -type f | xargs -I{} sh -c 'echo -e "$(basename {})\t$(sha256sum {} | awk "{print \$1}")"'
```

