# v3 172B exp2 (Sakura)

LLM-jp v3 172B exp2 の学習をSakuraクラスタ上で行うスクリプトです。

Experiment: https://github.com/llm-jp/experiments/issues/14

## スペック

* 必要リソース: gpu (H100 x8) 64ノード
* 学習速度: 1日あたり約2k steps

## 実行方法

事前に v3-megatron-sakura インストーラで `/home/shared/experiments/{exp-id}/environment` に環境をインストールしたものとします。
`{exp-id}` は登録時のIDを指定しますが、実験結果保全のため本実験のIDは指定しないでください。
また `/home/shared/experiments/{exp-id}/checkpoints` に以前のチェックポイントが保存されているものとします。

```shell
cd /data/experiments/{exp-id}

# ログ保存用ディレクトリ
mkdir outputs

# 実行
sbatch scripts/pretrain/scripts/v3-172b-exp2-sakura/sbatch.sh
```

## 保存する必要のないチェックポイントの移動

172Bでは1000iter刻み以外のチェックポイントは保存する予定がないので以下のスクリプトで`trash`に移動します。  
例外として最新のチェックポイントと1000iter未満のチェックポイントは残します。

`--dryrun`オプションがついているため、実際の移動は行われず行われる予定のコマンド操作だけ表示されます。
実際に移動する場合は`--dryrun`オプションを消してください。

```shell
python3 ./scripts/pretrain/scripts/v3-172b-exp2-sakura/mv_unnecessaary_ckpt.py \
    --src_dir /home/shared/experiments/9/checkpoints/tp4-pp16-cp1 \
    --dst_dir /home/shared/experiments/9/trash/tp4-pp16-cp1 \
    --dryrun
```