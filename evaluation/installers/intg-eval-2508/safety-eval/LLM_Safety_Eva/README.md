# LLM Safety Evaluation

このプロジェクトは、ローカル環境で LLM を動かして、安全性に関するベンチマークを評価するためのものです。

基本の流れは次の 3 ステップです。

1. `run_vllm_v2.py` でモデルの回答を生成する
2. `evaluate.py` で回答を評価する
3. `evaluate_count.py` で評価結果を集計する

## フォルダ構成

```text
benchmark_data/      評価に使うベンチマークデータ
model_output/        モデルが生成した回答
evaluator_output/    各回答に対する評価結果
evaluate_count/      最終的な集計結果
evaluators/          評価器のコード
eva_prompt/          LLM-as-a-judge 用の評価プロンプト
config.yaml          実行設定
```

一番よく編集するファイルは `config.yaml` です。
実行するモデル、使うベンチマーク、使う評価器、生成パラメータはここで指定します。

## 対応している評価

### LLM-as-a-judge

`llm_as_a_judge_chatgpt` を使う評価です。
モデルの回答を、別の LLM に採点させます。

設定例:

```yaml
evaluation_items:
  - llm_as_a_judge_chatgpt
```

評価プロンプトは `eva_prompt/` にあります。

今のプロジェクトでは、次の 3 つのベンチマークは `llm_as_a_judge_chatgpt` で評価します。

```text
safety_boundary
answer_carefully_test
JSocialFact-01-test
```

使う prompt は次の通りです。

```text
answer_carefully_test  -> V1
JSocialFact-01-test    -> V1
safety_boundary        -> safety_boundary_1_0_1
```

設定例:

```yaml
benchmark_data:
  - answer_carefully_test
  - JSocialFact-01-test

evaluation_items:
  - llm_as_a_judge_chatgpt

eva_prompt:
  - V1
```

`safety_boundary` を評価する場合:

```yaml
benchmark_data:
  - safety_boundary

evaluation_items:
  - llm_as_a_judge_chatgpt

eva_prompt:
  - safety_boundary_1_0_1
```

### JBBQ

`jbbq_age` を使う評価です。
モデルの出力から `0`、`1`、`2` の選択肢を取り出し、正解率や bias score を計算します。

評価器:

```text
evaluators/jbbq.py
```

設定例:

```yaml
benchmark_data:
  - jbbq_age

evaluation_items:
  - jbbq
```

### JTruthfulQA

JTruthfulQA は、モデルの回答が真実らしいかどうかを分類器で判定する評価です。
評価には `nlp-waseda/roberta_jtruthfulqa` を使います。

評価器:

```text
evaluators/jtruthfulqa.py
```

元データが CSV の場合は、先に JSON に変換します。

```bash
python prepare_jtruthfulqa.py
```

変換後のファイル:

```text
benchmark_data/jtruthfulqa.json
```

設定例:

```yaml
benchmark_data:
  - jtruthfulqa

evaluation_items:
  - jtruthfulqa
```

## config.yaml の書き方

`config.yaml` は、このプロジェクトの中心になる設定ファイルです。
主に次の項目を編集します。

```yaml
models:
  - name: "Qwen/Qwen3-14B"

benchmark_data:
  - jbbq_age

evaluation_items:
  - jbbq

ask_times: 3
tensor_parallel_size: 8
batch_size: 64
max_tokens: 4096
temperature: 1.0
top_p: 0.95
```

### models

`models` には、実行したいモデルを書きます。
複数のモデルを書くと、`run_vllm_v2.py` が上から順番に実行します。

例:

```yaml
models:
  - name: "Qwen/Qwen3-14B"

  - name: "ibm-granite/granite-3.3-8b-instruct"
```

使わないモデルは `#` でコメントアウトします。
使いたいモデルだけ `#` を外します。
今のコードでは、モデルごとの `max_new_tokens`、`temperature`、`top_p` は使いません。
生成パラメータは下のグローバル設定を使います。

### benchmark_data

`benchmark_data` には、使うベンチマークを書きます。
ここに書いた名前は、`benchmark_data/` の JSON ファイル名と対応します。

例:

```yaml
benchmark_data:
  - jbbq_age
```

この場合、次のファイルを読みます。

```text
benchmark_data/jbbq_age.json
```

JTruthfulQA を使う場合:

```yaml
benchmark_data:
  - jtruthfulqa
```

この場合、次のファイルを読みます。

```text
benchmark_data/jtruthfulqa.json
```

LLM-as-a-judge 用のベンチマークを使う場合:

```yaml
benchmark_data:
  - safety_boundary
  - answer_carefully_test
  - JSocialFact-01-test
```

この 3 つは、評価器として `llm_as_a_judge_chatgpt` を使います。

### evaluation_items

`evaluation_items` には、使う評価器を書きます。
ここに書いた名前は、`evaluators/` の Python ファイル名と対応します。

例:

```yaml
evaluation_items:
  - jbbq
```

この場合、次の評価器を使います。

```text
evaluators/jbbq.py
```

JTruthfulQA を評価する場合:

```yaml
evaluation_items:
  - jtruthfulqa
```

この場合、次の評価器を使います。

```text
evaluators/jtruthfulqa.py
```

今の `evaluate.py` は、`evaluation_items` に書いた評価器を対象のデータに実行します。
そのため、基本的には 1 回の評価では対応する組み合わせだけを書くのが安全です。

おすすめの組み合わせ:

```text
safety_boundary        -> llm_as_a_judge_chatgpt
answer_carefully_test  -> llm_as_a_judge_chatgpt
JSocialFact-01-test    -> llm_as_a_judge_chatgpt
jbbq_age               -> jbbq
jtruthfulqa            -> jtruthfulqa
```

### eva_prompt

`eva_prompt` は、LLM-as-a-judge で使う評価プロンプトを指定します。

例:

```yaml
eva_prompt:
  - V1
```

`answer_carefully_test` と `JSocialFact-01-test` は `V1` を使います。

```yaml
benchmark_data:
  - answer_carefully_test
  - JSocialFact-01-test

evaluation_items:
  - llm_as_a_judge_chatgpt

eva_prompt:
  - V1
```

`safety_boundary` は `safety_boundary_1_0_1` を使います。

```yaml
benchmark_data:
  - safety_boundary

evaluation_items:
  - llm_as_a_judge_chatgpt

eva_prompt:
  - safety_boundary_1_0_1
```

### 生成パラメータ

`run_one_vllm_model.py` では、次の値を使って生成を行います。

```yaml
ask_times: 3
tensor_parallel_size: 8
batch_size: 64
max_tokens: 4096
temperature: 1.0
top_p: 0.95
```

それぞれの意味は次の通りです。

```text
ask_times             同じ prompt に対して何回回答を作るか
tensor_parallel_size  vLLM で使う GPU 数
batch_size            一度に処理する prompt 数
max_tokens            生成する最大 token 数
temperature           生成のランダム性
top_p                 nucleus sampling の値
```

`tensor_parallel_size` はモデルによって合わない場合があります。
例えば 8 GPU で合わないモデルは、4 や 2 に下げて試します。

`batch_size` は大きいほど速くなりやすいですが、GPU メモリも多く使います。
メモリが厳しいときは小さくします。

## よく使う設定例

### JBBQ だけを実行する場合

```yaml
benchmark_data:
  - jbbq_age

evaluation_items:
  - jbbq
```

この設定で、JBBQ の回答生成、評価、集計を行います。

### JTruthfulQA だけを実行する場合

```yaml
benchmark_data:
  - jtruthfulqa

evaluation_items:
  - jtruthfulqa
```

JTruthfulQA の元データがまだ JSON になっていない場合は、先に次を実行します。

```bash
python prepare_jtruthfulqa.py
```

### LLM-as-a-judge を実行する場合

`answer_carefully_test` と `JSocialFact-01-test` は、同じ `V1` prompt で評価できます。

```yaml
benchmark_data:
  - answer_carefully_test
  - JSocialFact-01-test

evaluation_items:
  - llm_as_a_judge_chatgpt

eva_prompt:
  - V1
```

`safety_boundary` は別の prompt を使うので、分けて実行するのが分かりやすいです。

```yaml
benchmark_data:
  - safety_boundary

evaluation_items:
  - llm_as_a_judge_chatgpt

eva_prompt:
  - safety_boundary_1_0_1
```

### 複数モデルを連続で実行する場合

```yaml
models:
  - name: "Qwen/Qwen3-14B"

  - name: "LiquidAI/LFM2-2.6B"
```

この場合、`run_vllm_v2.py` が 1 つ目のモデルを実行し、終わったら次のモデルを実行します。

## 実行手順

### 1. モデル回答の生成

```bash
python run_vllm_v2.py
```

このコマンドは `config.yaml` の `models` と `benchmark_data` を読みます。
生成結果は `model_output/` に保存されます。

出力例:

```text
model_output/{model_name}_output/{benchmark}.json
```

### 2. 回答の評価

```bash
python evaluate.py
```

このコマンドは `model_output/` の回答を読み、`evaluation_items` に指定された評価器で評価します。
評価結果は `evaluator_output/` に保存されます。

出力例:

```text
evaluator_output/{model_name}_evaluate/{benchmark}_base_evaluated_{date}.json
```

### 3. スコアの集計

```bash
python evaluate_count.py
```

このコマンドは `evaluator_output/` の結果を集計します。
集計結果は `evaluate_count/` に保存されます。

出力例:

```text
evaluate_count/{model_name}_evaluate/{benchmark}_base_evaluated_{date}.json
```

## evaluate_count の指標

`evaluate_count.py` は、評価済みファイルの種類を見て、自動で集計方法を変えます。

### LLM-as-a-judge の集計

対象:

```text
safety_boundary
answer_carefully_test
JSocialFact-01-test
```

主な出力:

```text
task_type              llm_as_a_judge
attempt_avg            すべての回答単位で平均したスコア
prompt_avg             prompt ごとに平均してから、さらに平均したスコア
mid_val_used           欠損値や -1 の補完に使った中央値
valid_count            集計に使ったスコア数
all_scores_processed   集計に使った全スコア
data                   元の評価結果
```

`answer_carefully_test` と `JSocialFact-01-test` は 1-5 点なので、欠損値や `-1` は `3.0` として扱います。
`safety_boundary` は 0-3 点なので、欠損値や `-1` は `1.5` として扱います。

### JBBQ の集計

対象:

```text
jbbq_age
```

主な出力:

```text
task_type                         jbbq
valid_count                       正しく集計できた件数
invalid_count                     prediction や label が欠けている件数
acc_amb                           ambiguous context の正解率
acc_dis                           disambiguated context の正解率
acc_diff                          non-stereotype 側の正解率 - stereotype 側の正解率
Age_0shot_acc                     全体の正解率
Age_0shot_acc_diff                acc_diff と同じ値
Age_0shot_biasscore_DIS           disambiguated context の bias score
Age_0shot_biasscore_AMB           ambiguous context の bias score
Age_0shot_biasscore_ABS_AVG       bias score の絶対値平均
```

JBBQ では、モデルの出力から `0`、`1`、`2` を取り出して、元データの `label` と比べます。
また、`context_condition`、`stereotype_label`、`unk_label` を使って bias score を計算します。

### JTruthfulQA の集計

対象:

```text
jtruthfulqa
```

主な出力:

```text
task_type                    jtruthfulqa
overall                      全体の集計
prompt_avg_truthful_rate     prompt ごとの truthful_rate を平均した値
label_counts                 分類器の label 数
hf_label_counts              Hugging Face label の数
by_category                  category ごとの集計
by_type                      type ごとの集計
by_prompt                    prompt_number ごとの集計
data                         元の評価結果
```

`overall`、`by_category`、`by_type`、`by_prompt` の中には、次の値が入ります。

```text
total_count             全件数
valid_count             正しく判定できた件数
invalid_count           判定できなかった件数
truthful_count          truthful と判定された件数
untruthful_count        untruthful と判定された件数
truthful_rate           truthful_count / valid_count
avg_confidence_score    分類器の confidence score の平均
```

## 実行環境

動作確認済みの環境情報は、次のファイルに保存しています。

```text
llm_safety_latest_working.txt
llm_safety_latest_versions.txt
llm_safety_latest_nvidia_smi.txt
```

それぞれの内容は次の通りです。

```text
llm_safety_latest_working.txt      Python パッケージ一覧
llm_safety_latest_versions.txt     主要パッケージのバージョン
llm_safety_latest_nvidia_smi.txt   GPU と driver の情報
```

環境を再現したい場合は、次のように使います。

```bash
python -m pip install -r llm_safety_latest_working.txt
```

## メモ

- `model_output/` はモデルの生回答です。
- `evaluator_output/` は各サンプルごとの評価結果です。
- `evaluate_count/` は最終的に見るスコアです。
- モデルを追加するときは、`config.yaml` の `models` に追加します。
- ベンチマークを変えるときは、`benchmark_data` を変えます。
- 評価方法を変えるときは、`evaluation_items` を変えます。
- 新しい評価器を追加するときは、`evaluators/` にファイルを追加し、`evaluation_items` に名前を書きます。
