# チェックポイント変換スクリプト

このスクリプトは、Megatron形式のチェックポイントをHugging Face形式に変換します。

## スペック
- 必要リソース: cpu 1ノード

## 実行方法

### 注意事項
このスクリプトを実行する前に、環境に適したインストーラを実験ディレクトリにインストールしてください (例: /data/experiments/{exp-id}/venv)。
以前のチェックポイントが保存されていることを確認してください (例: /data/experiments/{exp-id}/checkpoints/)。

### 実行手順

1. 実験ディレクトリに移動します：
    ```shell
    cd /data/experiments/{exp-id}
    ```

2. スクリプトを実行環境と同じディレクトリにコピーします：
    ```shell
    cp {this directory} .
    ```

3. スクリプトを実行します：
  - SLURMが入っているクラスタ && cpu partionがある
    ```shell
    sbatch convert.sh SOURCE_DIR TARGET_DIR TEMPORAL_DIR
    ```
  - そのほか
    ```shell
    bash convert.sh SOURCE_DIR TARGET_DIR TEMPORAL_DIR
    ```


### パラメータ

- `SOURCE_DIR`: `iter_NNNNNNN`を含むMegatronチェックポイントディレクトリ
- `TARGET_DIR`: Hugging Face形式の出力ディレクトリ
- `TEMPORAL_DIR`: 中間ファイル用の一時ディレクトリ（オプション）
  - デフォルト値： `$HOME/tmp`

### 例

```shell
sbatch convert.sh /data/experiments/{exp-id}/checkpoints/iter_0001000 /data/experiments/{exp-id}/hf_checkpoints/iter_0001000 ~/temp
```

