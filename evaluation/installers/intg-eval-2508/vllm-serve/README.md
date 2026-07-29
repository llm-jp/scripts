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
- llm-jp-eval v2.x: `inference_openai.py` (GeneratorBase 継承) が dump 済み
  プロンプト (chat template 適用込みのトークン ID) を `/v1/completions` に
  投げる。dump / eval フェーズは各バージョンの環境をそのまま使用
- llm-jp-eval v1.4.x: `inference_openai_v1.py` (スタンドアロン) が
  `offline_inference_vllm.py` の代替として同じ入出力形式で動く
  (greedy / chat template 非対応は offline 版と同じ)。dump / eval は
  v1.4.1 環境の venv-eval をそのまま使用
- llm-jp-eval は 2 パス実行: dump + inference を全バージョン分サーバー稼働中に
  行い、eval (BERTScore / COMET が GPU メモリを使う) は**サーバー停止後**に
  まとめて実行する (`run_llm-jp-eval[-v1]-serve.sh --phase inference|eval`)。
  vLLM 0.19.1 は `--gpu-memory-utilization 0.9` でも GPU をほぼ全量
  (実測 98%) 確保するため、サーバー常駐のまま eval を走らせると CUDA OOM
  になる (gpt-oss-120b TP4 で実測)。llm-jp-judge の generation/judging 分割と
  同じ構成

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

### vllm 0.11.2 venv は openai を 1.99.1 にピンしないと `vllm serve` が起動できない

llm-jp-eval-v2.1.3 の vllm venv は lock が openai 1.99.5 を固定しており、
そのままでは `vllm serve` が import エラーで即死します。vllm 0.11.2 は
`openai.types.chat.chat_completion_message_tool_call_param` の `Function` 型に
依存していますが、この型は openai 1.99.2 の型再構成で削除されており、
vllm の依存指定 (`openai>=1.99.1`) が上限を欠くためです。オフライン評価は
openai クライアントを使わないため影響ありません。

llm-jp-eval-v2.1.3 のインストーラーは venv 同期後に `openai==1.99.1` を
ピンし直すようになっています (2026-07-24 以降)。それ以前にインストールした
環境では次の一行で修正できます (オフライン評価の挙動は変わりません):

```bash
uv pip install --python \
  $ENV_DIR/llm-jp-eval-v2.1.3/environment/src/llm-jp-eval/llm-jp-eval-inference/inference-modules/vllm/.venv/bin/python \
  "openai==1.99.1"
```

(以前この節にあった「サーバー専用 venv (`vllm-serve/serve-venv-*`) を作る」
回避策は不要になりました。専用 venv は自動検出の最優先候補となり他の
利用者のサーバー venv 選択まで変えてしまうため、残っている場合は削除を
推奨します。)

### サーバープロセスの PATH には serve venv の bin が乗る

サーバーは venv を activate せず `<venv>/bin/vllm` を直接起動するため、
serve_common.sh が activate 相当 (`PATH` 先頭に `<venv>/bin`、`VIRTUAL_ENV`)
を設定した上で起動します。これが無いと MoE モデル等のカーネル JIT が
`ninja` を PATH から見つけられず `FileNotFoundError: 'ninja'` でエンジン
初期化に失敗します (オフライン実行は venv を activate するため顕在化しない)。

## スコア互換性に関する注意

- **推論エンジンはサーバー側の vLLM バージョンになります。** オフライン実行と
  同一バージョンの vLLM を使う場合のみスコアは直接比較可能です
  (例: サーバーを v2.1.5 の venv (vllm 0.19.1) で立てれば llm-jp-eval v2.1.5 の
  オフライン結果と同等; v2.1.3 (vllm 0.11.2) のオフライン結果とは比較不可)。
- swallow の生成系タスクは temperature=0 (greedy)、loglikelihood は数学的に
  オフラインと同一。バッチ境界起因の数値ゆらぎは通常の実行間差と同程度
- llm-jp-eval の `--reasoning_parser` は本モードでは**未対応** (オフライン版を
  使用してください)
- `server.max_model_len` (デフォルト 4096) はオフライン版の
  `model.max_model_len` を模倣します: オフライン vLLM と同様プロンプトは
  保持したまま、生成長をリクエスト毎に残りコンテキストへクランプします
  (プロンプト単体がコンテキストを超える場合のみ `truncate_prompt_tokens`
  で切り詰め)
- swallow クライアントの `max_length` はデフォルトでサーバーの実際の
  max_model_len (= モデル config、`--max-model-len` 指定時はその値) に
  追従します。オフライン版ハーネスがエンジンのコンテキスト長に従うのと
  同じ挙動です (`--swallow-max-length` で上書き可)

## 適用されるパッチ

`patches/openai_completions-serve-compat.patch` (lm-evaluation-harness-en):

1. `batch_size` が CLI から文字列で渡ると chunking の比較が壊れる → int キャスト
2. `generate_until` が `until` を `completions.create()` にそのまま渡し、
   openai クライアント 2.x が未知引数として拒否 → 除外 (`stop` として送信済み)
3. `num_concurrent` (model_args) を追加: バッチを ThreadPoolExecutor で複数
   同時にサーバーへ送る。素の実装はバッチ 1 本ずつの直列送信で、サーバーの
   continuous batching に仕事が渡らず GPU が遊ぶ (150m の実測でオフライン比
   2.4 倍遅かった)。リクエスト内容・結果順序は不変で、変わるのはバッチ構成
   のみ (既に許容済みの数値ゆらぎと同クラス)

## クライアント並列度 (`--client-concurrency`)

サーバーを飽和させるための in-flight プロンプト数はフレームワークに依らない
共通概念なので、`run_eval_serve.sh` の `--client-concurrency N` (デフォルト
256 ≈ vLLM の max_num_seqs 相当) に一本化しています。各クライアントが自分の
リクエスト形に変換します:

- swallow: `num_concurrent = ceil(N / batch_size)` (バッチ 16 プロンプト単位)
- llm-jp-eval: `server.num_concurrent = N` (1 プロンプト/リクエストなので 1:1)
- 新しい評価フレームワークを追加する場合も、そのアダプタで同様に変換する

過剰な値は速度向上には繋がらないが害も小さい (サーバー内キューで待つだけ)。
不足すると GPU が遊ぶため、大きめが安全側。

`openai_completions.py` は従来のオフライン評価フロー (`--model vllm`) では
使用されないため、**既存の評価挙動には影響しません**。

## 検証状況

実クラスタ (ABCI / さくら) での検証記録は [../VALIDATION.md](../VALIDATION.md)
に集約している。要点:

- **スコア一致性は ABCI (H100) とさくら (B200) の両方で確認済み**
  (swallow / llm-jp-eval v1.4.1 / v2.1.3 / v2.1.5)
- 所要時間はモデルロード削減 + `--client-concurrency` により、
  gpt-oss-120b (TP4, swallow + v2.1.3) 実測でオフライン比 **2.4 倍**
- gpt-oss 等の MXFP4 MoE は vllm 0.11.2 だと生成品質が劣化する
  (serve 経由では `!!!!` への出力退化)。サーバーもオフラインも
  v2.1.5 環境 (vllm 0.19.1) を使うこと
