# vllm-serve モード (EXPERIMENTAL)

巨大モデル評価の効率化のため、**vLLM サーバーを 1 回だけ起動し、swallow と
llm-jp-eval の評価クライアントがそのエンドポイントを共有する**実行モードです。

## 背景

従来のオフライン実行では、モデルロードが繰り返し発生します:

- swallow (`run-eval.sh`) はタスクグループごとに `lm_eval` を起動するため、
  1 回の評価で **6 回**モデルをロードする
- llm-jp-eval は 1 バージョンあたり 1 回ロードする (複数バージョンなら回数分)

巨大モデル (TP 並列) では 1 ロードに数十分かかるため、これが支配的なコストに
なります。本モードではロードは **ジョブ全体で 1 回**です。

## 仕組み

```
vllm serve MODEL (GPU, 常駐)  ←  http://127.0.0.1:PORT/v1
    ├── swallow:     lm_eval --model local-completions (6回起動、ロードなし)
    └── llm-jp-eval: inference_openai.py (dump/eval は従来どおり)
```

- swallow: 既存フォークの `local-completions` バックエンドを使用。プロンプトは
  HF tokenizer による**トークン ID 列**で送信され、loglikelihood は
  `echo=True + logprobs` で取得されるため、計算内容はオフライン版と同一
- llm-jp-eval: `inference_openai.py` (GeneratorBase 継承) が dump 済みプロンプト
  (chat template 適用込みのトークン ID) を `/v1/completions` に投げる。
  dump / eval フェーズは各バージョンの環境をそのまま使用

## 使い方

```bash
ENV_DIR=/data/experiments/<exp>/environment

# インストール (既存環境への追加コピー + swallow への互換パッチ)
bash install.sh $ENV_DIR

# 実行例: サーバー1本で swallow + llm-jp-eval v2.1.5 を評価
bash $ENV_DIR/vllm-serve/run_eval_serve.sh \
  <model_name_or_path> <output_dir> \
  --swallow --swallow-env swallow_v202411-tf5 \
  --llm-jp-eval-versions v2.1.5 \
  --tensor-parallel-size 8
```

サーバーの venv は `--serve-venv` で指定可能 (デフォルトは自動検出:
`vllm-serve/serve-venv-*` → llm-jp-eval-v2.1.5 の vllm venv →
swallow_v202411-tf5 → swallow_v202411)。

### vllm 0.11.2 venv は `vllm serve` が起動できない場合がある

llm-jp-eval-v2.1.3 の vllm venv (vllm 0.11.2 + openai 1.99.5) では
`vllm serve` が import エラーで即死します。vllm 0.11.2 は
`openai.types.chat.chat_completion_message_tool_call_param` の `Function` 型に
依存していますが、この型は openai 1.99.2 の型再構成で削除されており、
vllm の依存指定 (`openai>=1.99.1`) が上限を欠くためです。オフライン評価は
サーバーを使わないため影響ありません。

対処: 既存 venv は変更せず、サーバー専用 venv を作成します
(スコア互換性のため vllm / torch などは既存 venv と同一バージョンに揃える):

```bash
uv venv --python 3.11.13 $ENV_DIR/vllm-serve/serve-venv-vllm0.11.2
uv pip install --python $ENV_DIR/vllm-serve/serve-venv-vllm0.11.2/bin/python \
  vllm==0.11.2 openai==1.99.1 flashinfer-python==0.5.2 \
  xformers==0.0.33.post1 numpy==1.26.4 transformers==4.57.6
```

`vllm-serve/serve-venv-*` は自動検出の最優先候補なので、作成後は
`--serve-venv` の指定は不要です。

## スコア互換性に関する注意

- **推論エンジンはサーバー側の vLLM バージョンになります。** オフライン実行と
  同一バージョンの vLLM を使う場合のみスコアは直接比較可能です
  (例: サーバーを v2.1.5 の venv (vllm 0.19.1) で立てれば llm-jp-eval v2.1.5 の
  オフライン結果と同等; v2.1.3 (vllm 0.11.2) のオフライン結果とは比較不可)。
- swallow の生成系タスクは temperature=0 (greedy)、loglikelihood は数学的に
  オフラインと同一。バッチ境界起因の数値ゆらぎは通常の実行間差と同程度
- llm-jp-eval の `--reasoning_parser` は本モードでは**未対応** (オフライン版を
  使用してください)
- `server.max_model_len` (デフォルト 4096) は vLLM の `truncate_prompt_tokens`
  でオフライン版の `model.max_model_len` を模倣します

## 適用されるパッチ

`patches/openai_completions-serve-compat.patch` (lm-evaluation-harness-en):

1. `batch_size` が CLI から文字列で渡ると chunking の比較が壊れる → int キャスト
2. `generate_until` が `until` を `completions.create()` にそのまま渡し、
   openai クライアント 2.x が未知引数として拒否 → 除外 (`stop` として送信済み)

`openai_completions.py` は従来のオフライン評価フロー (`--model vllm`) では
使用されないため、**既存の評価挙動には影響しません**。

## 検証状況 (2026-07-20)

- mock OpenAI サーバー (CPU) に対して:
  - `lm_eval local-completions`: hellaswag (loglikelihood) + gsm8k (生成) 完走
  - `inference_openai.py`: jamp 100 サンプル (v2.1.5 の実 venv 使用) 完走
- **実 GPU + 実 vLLM サーバーでの end-to-end とスコア一致性検証は未実施**
  (GPU キュー空き待ち)。検証手順は下記

## GPU 検証手順 (TODO)

```bash
# 1. スモークテスト (150m, サーバー共有で swallow+llm-jp-eval)
sbatch --partition=gpu --gpus=1 --mem=180G --time=6:00:00 --wrap "\
  bash $ENV_DIR/vllm-serve/run_eval_serve.sh llm-jp/llm-jp-3-150m \
    /data/experiments/<exp>/results/serve-smoke-<date> \
    --swallow --llm-jp-eval-versions v2.1.5"

# 2. スコア一致性: 同一モデルのオフライン実行結果 (sbatch.py 経由) と
#    swallow result.json / llm-jp-eval result.json を比較
```
