# 安全性評価 (safety-eval / LLM_Safety_Eva)

安全性WGから受領した安全性評価コード ([LLM_Safety_Eva/](./LLM_Safety_Eva/) に
ベンダリング。ジャッジ評価器への OpenAI互換エンドポイント対応の追加を除き無改変)
を統合評価基盤に組み込むサブインストーラです。

以下のベンチマークを評価します:

| ベンチマーク | 評価器 | 評価プロンプト | 外部API |
| --- | --- | --- | --- |
| `jbbq_age` | `jbbq` (正解率 / bias score) | - | 不要 |
| `jtruthfulqa` | `jtruthfulqa` (分類器 `nlp-waseda/roberta_jtruthfulqa`) | - | 不要 |
| `answer_carefully_test` | `llm_as_a_judge_chatgpt` (1-5点) | `V1` | ジャッジAPI |
| `JSocialFact-01-test` | `llm_as_a_judge_chatgpt` (1-5点) | `V1` | ジャッジAPI |
| `safety_boundary` | `llm_as_a_judge_chatgpt` (0-3点) | `safety_boundary_1_0_1` | ジャッジAPI |

処理は 3 フェーズです: (1) ターゲットモデルの回答生成 (ローカルvLLM,
オフライン推論)、(2) 評価器による採点、(3) スコア集計。詳細は
[LLM_Safety_Eva/README.md](./LLM_Safety_Eva/README.md) を参照してください。

## ベンチマークデータについて (重要)

**ベンチマークデータ (`LLM_Safety_Eva/benchmark_data/`) はこのリポジトリに
含まれていません**。サイズが大きい (~160MB) ことに加え、gated データセット
(AnswerCarefully) 由来のデータを含むため、公開リポジトリにはコミットできません。

安全性WGからの配布 zip (`LLM_Safety_Eva_Web-main.zip`) を展開し、その中の
`LLM_Safety_Eva/benchmark_data/` を以下のいずれかで指定してください:

```bash
# 方法1: このディレクトリ配下に配置 (.gitignore 済み)
unzip LLM_Safety_Eva_Web-main.zip -d /tmp/safety_eva
cp -r /tmp/safety_eva/LLM_Safety_Eva_Web-main/LLM_Safety_Eva/benchmark_data \
      ./LLM_Safety_Eva/benchmark_data

# 方法2: install.sh の第2引数で指定
bash install.sh $INSTALL_DIR/safety-eval /path/to/benchmark_data
```

親インストーラ (`../install.sh`) からは、上記の方法1のパス
(または環境変数 `SAFETY_EVAL_BENCHMARK_DATA`) にデータがある場合のみ
このコンポーネントがインストールされ、無い場合は警告してスキップされます。

## インストール

```bash
cd evaluation/installers/intg-eval-2508/safety-eval

bash install.sh $INSTALL_DIR/safety-eval [BENCHMARK_DATA_DIR] \
  > ../logs/install-safety-eval.out 2> ../logs/install-safety-eval.err
```

- venv は安全性WGの動作確認済み環境に合わせて `vllm==0.11.2` /
  `transformers==4.57.6` (torch 2.9.0+cu128, Python 3.10) を固定します
  ([requirements.txt](./requirements.txt) /
  [LLM_Safety_Eva/llm_safety_latest_versions.txt](./LLM_Safety_Eva/llm_safety_latest_versions.txt))。
- JTruthfulQA の分類器 (`nlp-waseda/roberta_jtruthfulqa`) はインストール時に
  環境内キャッシュ (`environment/data/hf/`) にプリフェッチされ、評価時は
  `HF_HUB_OFFLINE=1` で参照します (計算ノードでのダウンロード作業を避けるため)。
  インストールはログインノード等で実行してください。
- この分類器のトークナイザは Juman++ を必要とするため (`word_tokenizer_type:
  jumanpp`)、インストーラが Juman++ v2.0.0-rc4 を環境内
  (`environment/jumanpp/`) にソースビルドします (cmake と C++ コンパイラが
  必要)。評価時は `run_safety-eval.sh` が PATH に追加します。

## 実行

通常は `qsub.py` / `sbatch.py` の `--safety-eval` から起動します
([../README.md](../README.md) 参照)。単体実行:

```bash
# ジャッジAPI (judge系ベンチマークを使う場合、いずれか):
export AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=...   # Azure OpenAI
# export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-2024-11-20     #   (デフォルト値)
export OPENAI_API_KEY=... OPENAI_BASE_URL=...               # OpenAI互換サーバー

bash $INSTALL_DIR/safety-eval/run_safety-eval.sh \
  <model_name_or_absolute_path> \
  <output_dir> \
  [--benchmarks jbbq_age,jtruthfulqa,answer_carefully_test,JSocialFact-01-test,safety_boundary] \
  [--tensor-parallel-size N] \
  [--ask-times 3] [--batch-size 64] [--max-tokens 4096] \
  [--temperature 1.0] [--top-p 0.95] \
  [--judge-model NAME] \
  [--benchmark-size N] \
  [--generation-only | --eval-only]
```

- judge系ベンチマーク (`answer_carefully_test` / `JSocialFact-01-test` /
  `safety_boundary`) を含む場合、ジャッジAPIのクレデンシャル (Azure:
  `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`、または OpenAI互換:
  `OPENAI_API_KEY` + `OPENAI_BASE_URL`。両方あればAzure優先) が未設定なら
  生成前に即エラーで停止します。ローカル評価のみで良い場合は
  `--benchmarks jbbq_age,jtruthfulqa` を指定してください。
- `--judge-model` はジャッジのモデル名です (Azure: デプロイメント名、既定
  `gpt-4o-2024-11-20`; OpenAI互換: そのサーバーが提供するモデル名。OpenAI互換
  サーバーでは通常指定が必要です)。
- `--judge-max-tokens N` はジャッジ1リクエストの max_tokens です (既定 512)。
  **thinking系のジャッジモデルは reasoning で予算を使い切って本文が空になる**
  ため、2048 程度を指定してください (例: mdx 上の OpenAI 互換サーバーの `gemma-4-31B-it`)。
- `--benchmark-size N` は各ベンチマークの先頭Nサンプルだけで実行します
  (スモークテスト用。生成済み `model_output/` はそのまま再利用される点に注意)。
- 生成済みの `model_output/` があるベンチマークはスキップされるため、中断後の
  再実行や、`--generation-only` → (APIキー設定後) `--eval-only` の分割実行が可能です。
- インストール時にデータが無かったベンチマークは警告してスキップされます。

### 出力

```text
<output_dir>/
  model_output/<model>_output/<benchmark>.json     モデルの生回答
  evaluator_output/<model>_evaluate/*.json         サンプル毎の評価結果
  evaluate_count/<model>_evaluate/*.json           最終的な集計スコア
  work/                                            実行時の作業コピー (config含む)
  logs/                                            フェーズ毎のログ
```

## 実装メモ

- 受領コードは原則無改変で使う方針です。変更は以下の3点のみで、変更箇所には
  「intg-eval modification」コメントを付与しています:
  1. `evaluators/llm_as_a_judge_chatgpt.py`: OpenAI互換エンドポイント対応
     (`AZURE_OPENAI_ENDPOINT` 未設定かつ `OPENAI_BASE_URL` 設定時に標準の
     `chat/completions` + Bearer認証で呼び、モデル名は環境変数
     `SAFETY_EVAL_JUDGE_MODEL` から取る)
  2. `evaluators/llm_as_a_judge_chatgpt.py`: ジャッジの max_tokens
     (元は512固定) を環境変数 `SAFETY_EVAL_JUDGE_MAX_TOKENS` で上書き可能に
     (thinking系ジャッジ対応)
  3. `evaluators/jtruthfulqa.py`: 分類器入力を先頭1000文字に制限 (トークナイザ
     は128トークンで切るためスコア不変。無制限だと長大な生成に対する Juman++
     前処理が「empty result」で失敗し、該当サンプルが invalid になる)

  CWD相対でconfig/データ/出力を解決するため、
  `run_safety-eval.sh` は `<output_dir>/work/` にコードの作業コピーを作り
  (benchmark_data はインストール先への symlink、出力3ディレクトリは
  `<output_dir>` 直下への symlink)、そこから実行します。共有インストール
  ディレクトリへの書き込みはありません。
- 生成は `run_one_vllm_model.py` を直接起動します (モデルは1ジョブ1個のため。
  受領コードの `run_vllm_v2.py` は `pkill -u $USER` によるプロセス掃除を行い、
  共有ノードで同一ユーザーの無関係なプロセスを殺しうるため使いません)。
- `run_one_vllm_model.py` は `gpu_memory_utilization=0.85` /
  `max_model_len=4096` をハードコードしています (受領時のまま)。
- 生成パラメータのデフォルト (`ask_times=3`, `temperature=1.0`, `top_p=0.95`,
  `max_tokens=4096`) は受領時の `config.yaml` に合わせています。
- 評価器/プロンプトの組み合わせが異なるため、`evaluate.py` は組み合わせ毎に
  config.yaml を書き換えて最大4回実行されます (`work/config_<pass>.yaml` に
  各パスの設定を保存)。
- `evaluators/toxicity.py` / `bias.py` / `privacy.py` / `jailbreak.py` は
  受領コード中のスタブ (乱数) であり、本パイプラインでは使用しません。
