# チェックポイント変換スクリプト

このスクリプト群は、Megatron形式のチェックポイントとHugging Face形式のチェックポイントを相互に変換します。

## スペック

- 必要リソース: GPU 1ノード
  - VRAMは使用せず、CUDAチェックにのみ使用します。

## 実行方法

### 注意事項

- 環境に適したインストーラにより事前に実験ディレクトリに事前学習用Pythonの環境構築を行なってください (例: /data/experiments/{exp-id}/environment)。
以前のチェックポイントが保存されていることを確認してください (例: /data/experiments/{exp-id}/checkpoints/)。

### 準備

1. 実験ディレクトリに移動します：
    ```shell
    cd /data/experiments/{exp-id}
    ```

2. スクリプトの準備と出力フォルダの作成：
    ```shell
    cp {this directory}/convert.sh .
    mkdir -p outputs
    ```

### チェックポイント変換スクリプトの実行
SLURMで実行する場合、--mem オプションをモデルサイズの2倍以上に設定してください。

#### Megatron → Hugging Face

##### SLURMでの実行例
```shell
sbatch --partition {partition} convert.sh SOURCE_DIR TARGET_DIR
```
##### SLURMなしの環境での実行例
```shell
bash convert.sh SOURCE_DIR TARGET_DIR > outputs/convert.out 2> outputs/convert.err
```
##### パラメータ
- SOURCE_DIR: iter_NNNNNNN を含むMegatron形式のチェックポイントディレクトリ
- TARGET_DIR: Hugging Face形式の出力ディレクトリ

#### Hugging Face → Megatron

##### SLURMでの実行例
```shell
sbatch --partition {partition} hf2megatron.sh SOURCE_DIR TARGET_DIR
```
##### SLURMなしの環境での実行例
```shell
bash hf2megatron.sh SOURCE_DIR TARGET_DIR > outputs/convert.out 2> outputs/convert.err
```

##### パラメータ
- SOURCE_DIR: Hugging Face形式のチェックポイントディレクトリ
- TARGET_DIR: Megatron形式の出力ディレクトリ
  - ディレクトリ名には、tp{N:int} と pp{M:int} の形式でテンソル並列とパイプライン並列サイズを含めてください。

#### サンプルコード

##### Megatron → Hugging Face:
```shell
sbatch convert.sh /data/experiments/{exp-id}/checkpoints/iter_0001000 /data/experiments/{exp-id}/hf_checkpoints/iter_0001000
```
##### Hugging Face → Megatron:
```shell
sbatch hf2megatron.sh /data/experiments/{exp-id}/hf_checkpoints /data/experiments/{exp-id}/checkpoints/tp2-pp2-cp1
```

### 注意事項： 作業ディレクトリについて
- 実行中、$HOME 上に ckpt_convert_YYYYMMDDHHSSMM という形式の一時作業用ディレクトリが作成されます。
- 実行エラーが発生しても、デバッグのためにこのディレクトリが残ります。必要に応じて手動で削除してください。
