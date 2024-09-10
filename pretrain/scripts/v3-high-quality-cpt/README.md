# v3 model hight quality continual pretrain 

LLM-jp v3 の学習をSakura・llm-jp-nvlinkクラスタ上で行うスクリプトです。

Experiment: https://github.com/llm-jp/experiments/issues/22

## スペック

* 必要リソース: 
  * llm-jp-nvlink: gpu (A100 x8) 16ノード
  * sakura: gpu-small (H100 x8) 16ノード
* 学習速度: 
  * 1.7B: 1日あたり約 90k steps
  * 3.8B: 1日あたり約 40k steps
  * 13B: 1日あたり約 11k steps

## 実行方法

事前に v3-megatron-sakura インストーラで `/data/experiments/{exp-id}/environment` に環境をインストールしたものとします。
`{exp-id}` は登録時のIDを指定しますが、実験結果保全のため本実験のIDは指定しないでください。
また `/data/experiments/{exp-id}/checkpoints` に以前のチェックポイントが保存されているものとします。

```shell
cd /data/experiments/{exp-id}

# 実行環境と同じ階層にスクリプトをコピー
cp megatron_data_formatter.sh . 
cp -r {target directory} .

# ログ保存用ディレクトリ
mkdir outputs

# 実行
sbatch sbatch.sh
```
