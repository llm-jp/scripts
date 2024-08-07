# llm-jp-eval v1.3.1 installation and execution script for any environment

llm-jp-eval の v1.3.1 で評価するためスクリプト
環境構築のためのスクリプト・評価実行のためのスクリプトを含みます

## Usage

### Build

インストール処理のためにCPUノードを1個使用します。

1. リポジトリのクローン
  ```shell
  git clone https://github.com/llm-jp/scripts
  cd scripts/evaluation/installers/llm-jp-eval-v1.3.1
  mkdir logs
  ```

2. インストール（例：~/myspace）
  -  SLURMが入っているクラスタ
    ```shell
    sbatch --partition {partition} install.sh ~/myspace
    ```
  - そのほかのクラスタ
    ```shell
    bash install.sh ~/myspace > logs/install.out 2> logs/install.err
    ```

3. (Optional) 設定：wandb, huggingface
  - SLURMが入っているクラスタでの追加処理
    ```shell
    srun --partition {partition} --nodes 1 --pty bash
    ```
  ```shell
  cd ~/myspace
  source venv/bin/activate
  wandb login
  huggingface-cli login
  ```
- SLURMが入っているクラスタでの追加処理
    ```shell
    exit
    ```

### Evaluation
```shell
cd ~/myspace
```
-  SLURMが入っているクラスタ
  ```shell
  sbatch --partition {partition} run_llm-jp-eval.sh {path/to/model} {wandb.project} {wandb.run_name} 
  ```
- そのほかのクラスタ
  ```shell
  CUDA_VISIBLE_DEVICES={num} bash run_llm-jp-eval.sh {path/to/model} {wandb.project} {wandb.run_name}
  ```


### Check

インストール終了後、下記のようなディレクトリ構造が構築されています。

```
~/myspace/
    installer_envvar.log  インストール開始後に記録した環境変数の一覧
    install.sh            使用したインストールスクリプト
    run_llm-jp-eval.sh    評価を実行するスクリプト
    dataset/llm-jp-eval   llm-jp-eval評価用データセット
    python/               Python実行環境
    scripts/              各種の環境設定用スクリプト
    src/                  個別ダウンロードされたライブラリ
    venv/                 Python仮想環境 (python/ にリンク)
```


## hashs.tsv の作成コマンド
```shell
TARGET_DIR={path/to/directory/containing/json/files}
find $TARGET_DIR -type f | xargs -I{} sh -c 'echo -e "$(basename {})\t$(sha256sum {} | awk "{print \$1}")"'
```
