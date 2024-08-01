# v3 13B exp4 (Sakura)

LLM-jp v3 13B exp4 の学習をSakuraクラスタ上で行うスクリプトです。

## スペック

* 必要リソース: gpu-small (H100 x8) 32ノード
* 学習速度: 1日あたり約20k steps

## 実行方法

事前に v3-megatron-sakura インストーラで `/data/experiments/12345/environment` に環境をインストールしたものとします。
また `/data/experiments/12345/checkpoints` に以前のチェックポイントが保存されているものとします。

```shell
cd /data/experiments/12345

# 実行環境と同じ階層にスクリプトをコピー
cp {this directory} .

# ログ保存用ディレクトリ
mkdir outputs

# 実行
sbatch sbatch.sh
```
