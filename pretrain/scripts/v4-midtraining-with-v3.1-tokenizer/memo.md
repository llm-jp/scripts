# 中間学習memo

- 学習にあたり、READMEに書くほど整理されていないが書き残しておきたいことを書く場所
- 内容が固まってきたら適宜README.mdに移行する

## 学習の手順

1. latest_checkpointed_iteration.txtの追加
2. train_data.all.shに学習data pathを記入
  - `ls -al /path/to/data.bin`の値を4で割れば良い
3. data pathの値を合計して、midtrain/paramsに学習iter値を書き込む
4. train_iters.txtに全体の学習量を書く
5. checkpointのシンボリックリンクを貼る
```sh
export TARGET_DIR=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/v4-megamath-pro-max/7.7b_v4_3.5t_tokenizer_v3.1/80B/checkpoints/iter_0432581
export SRC_DIR=/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/checkpoints_bak/7.7b_v4_3.5t_tokenizer_v3.1/iter_0432581
ln -s $SRC_DIR $TARGET_DIR
```

## Tokenizerのコピー

- `/groups/gcg51557/experiments/0138_corpus_v4_pretrain/src/llm-jp-tokenizer/hf/ver3.1`
- `/groups/gcg51557/experiments/0138_corpus_v4_pretrain/src/llm-jp-tokenizer/models/ver3.1`
の内容をコピーする

## checkpoint読み込めねえ

`/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/environment3/src/Megatron-LM/read_metadata.py`を作成
```py
# read_ckpt_metadata.py
import os
from megatron.core import dist_checkpointing

def read_parallel_sizes(ckpt_dir: str):
    cs = dist_checkpointing.load_common_state_dict(ckpt_dir)
    # まずどのキーに入っているか確認
    if "metadata" in cs:
        meta = cs["metadata"]
    elif "args" in cs:
        meta = cs["args"]
    elif "megatron_args" in cs:
        meta = cs["megatron_args"]
    else:
        raise KeyError(f"No metadata-like key in common_state: {list(cs.keys())}")

    print(f"Checkpoint dir        : {ckpt_dir}")
    print(f"Tensor parallel size  : {meta.tensor_model_parallel_size}")
    print(f"Pipeline parallel size: {meta.pipeline_model_parallel_size}")
    print(f"Data parallel size    : {meta.data_parallel_size}")

if __name__ == "__main__":
    # チェックポイントの親ディレクトリを指定
    # 例: /…/iter_0432581
    ckpt_dir = "/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/checkpoints_bak/7.7b_v4_3.5t_tokenizer_v3.1/iter_0432581/"
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"{ckpt_dir} が見つかりません")
    read_parallel_sizes(ckpt_dir)
```

```stdout
Tensor parallel size  : 1
Pipeline parallel size: 1
Data parallel size    : 512
```


## GSM8Kのファイルに空行が混じっていたので、アドほっくはスクリプトを作って修正

- script: `/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/preprocess/gsm8k_tokenize.sh`
- directory: `/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/dolmino-tmp/math/gsm8k-all_clean.jsonl`

```sh
qsub ./scripts/gsm8k_tokenize.sh
```
