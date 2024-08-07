# v3 172B exp2 (Sakura)

LLM-jp v3 172B exp2 の学習をSakuraクラスタ上で行うスクリプトです。

Experiment: https://github.com/llm-jp/experiments/issues/9

## スペック

* 必要リソース: gpu (H100 x8) 64ノード
* 学習速度: 1日あたり約2k steps

## 実行方法

事前に v3-megatron-sakura インストーラで `/data/experiments/{exp-id}/environment` に環境をインストールしたものとします。
`{exp-id}` は登録時のIDを指定しますが、実験結果保全のため本実験のIDは指定しないでください。
また `/data/experiments/{exp-id}/checkpoints` に以前のチェックポイントが保存されているものとします。

```shell
cd /data/experiments/{exp-id}

# 実行環境と同じ階層にスクリプトをコピー
cp {this directory} .

# ログ保存用ディレクトリ
mkdir outputs

# 実行
sbatch sbatch.sh
```
