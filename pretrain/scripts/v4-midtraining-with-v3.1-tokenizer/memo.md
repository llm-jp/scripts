# 中間学習memo

- 学習にあたり、READMEに書くほど整理されていないが書き残しておきたいことを書く場所
- 内容が固まってきたら適宜README.mdに移行する

## 学習の手順

1. latest_checkpointed_iteration.txtの追加
2. train_data.all.shに学習data pathを記入
  - `ls -al /path/to/data.bin`の値を4で割れば良い
3. data pathの値を合計して、midtrain/paramsに学習iter値を書き込む
4. train_iters.txtに全体の学習量を書く

## Tokenizerのコピー

- `/groups/gcg51557/experiments/0138_corpus_v4_pretrain/src/llm-jp-tokenizer/hf/ver3.1`
- `/groups/gcg51557/experiments/0138_corpus_v4_pretrain/src/llm-jp-tokenizer/models/ver3.1`
の内容をコピーする
