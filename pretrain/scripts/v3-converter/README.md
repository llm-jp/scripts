# チェックポイント変換スクリプト

このスクリプトは、Megatron形式のチェックポイントをHugging Face形式に変換します。

## スペック
- 必要リソース: gpu 1ノード
  - VRAMは使用せず、pytorch上でのCUDAチェックにのみ利用

## 実行方法

### 注意事項
このスクリプトを実行する前に、環境に適したインストーラを実験ディレクトリにインストールしてください (例: /data/experiments/{exp-id}/enviroment)。
以前のチェックポイントが保存されていることを確認してください (例: /data/experiments/{exp-id}/checkpoints/)。

### 実行手順

1. 実験ディレクトリに移動します：
    ```shell
    cd /data/experiments/{exp-id}
    ```

2. スクリプトを実行環境と同じディレクトリにコピーし、ログ出力フォルダを作成します：
    ```shell
    cp {this directory}/convert.sh .
    mkdir logs
    ```

3. スクリプトを実行します：
    ```shell
    # For a cluster with SLURM
    sbatch --partition {partition} convert.sh SOURCE_DIR TARGET_DIR
    # For a cluster without SLURM
    bash convert.sh SOURCE_DIR TARGET_DIR > logs/convert.out 2> logs/convert.err
    ```


### パラメータ
- `SOURCE_DIR`: `iter_NNNNNNN`を含むMegatronチェックポイントディレクトリ
- `TARGET_DIR`: Hugging Face形式の出力ディレクトリ

### サンプルコード
```shell
sbatch convert.sh /data/experiments/{exp-id}/checkpoints/iter_0001000 /data/experiments/{exp-id}/hf_checkpoints/iter_0001000
```

### 作業ディレクトリについて
実行中、$HOME上作業用ディレクトリ(`ckpt_convert_YYYYMMDDHHSSMM`)が作成されます。
実行エラーが起きてもデバッグのために残る私用のため各自で削除してください。
