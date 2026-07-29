# CHANGELOG (intg-eval-2508)

利用者に影響する変更のまとめ (新しい順)。実験・検証の詳細は
[VALIDATION.md](./VALIDATION.md) を、vllm-serve モードの設計は
[vllm-serve/README.md](./vllm-serve/README.md) を参照。

## 2026-07-29

- **変更 (vllm-serve)**: llm-jp-eval の eval フェーズ (BERTScore / COMET) を
  **サーバー停止後**に実行するよう再構成。vLLM 0.19.1 サーバーは
  `--gpu-memory-utilization 0.9` でも GPU をほぼ全量確保するため、従来構成では
  eval が CUDA OOM になっていた。単体実行用に
  `run_llm-jp-eval[-v1]-serve.sh --phase inference|eval` を追加 (デフォルト
  `all` で従来どおり)

## 2026-07-24

- **追加**: `--basemodel` (qsub.py / sbatch.py / serve モード)。事前学習
  チェックポイント向けの評価モード: 固定プロンプトテンプレート・
  `add_special_tokens=False`・temperature 0.0・4-shot 系データセットのみ。
  結果の `lang_scores` に JA/EN 別集計。llm-jp-eval v2.1.5 以降のみ、
  `--apply-chat-template` と排他
- **追加**: llm-jp-judge (LLM-as-a-Judge) サブインストーラーと
  `--llm-jp-judge` 一式 (`--judge-client openai|azure|bedrock|vllm`、
  `--judge-model` など)。serve モードでは生成を共有サーバーで行い、
  judging はサーバー停止後に実行
- **追加**: qsub.py / sbatch.py の dry-run 回帰テスト
  (`tests/run_regression.sh`)。オプション組み合わせごとの生成ジョブ
  スクリプトを golden 比較
- **追加 (vllm-serve)**: llm-jp-eval **v1.4.1 の serve 対応**
  (`inference_openai_v1.py`; chat template 非対応はオフライン版と同じ)
- **変更**: 評価対象が重複しない限り、複数ジョブで同一 output_dir を共有可能に
  (ジョブアーティファクトは `sbatch_<eval-set>.out` 等にサフィックス付与)
- **修正 (重要)**: llm-jp-eval v2.1.3 の vllm venv に `openai==1.99.1` をピン。
  lock の openai 1.99.5 では vllm 0.11.2 の `vllm serve` が import エラーで
  起動不可 (オフライン評価には影響なし)。既存環境の修正方法は
  vllm-serve/README.md 参照
- **修正 (重要)**: serve モードのサーバーを venv の PATH を通して起動するよう修正。
  従来は MoE モデル等のカーネル JIT が `ninja` を見つけられず
  `FileNotFoundError` でエンジン初期化に失敗していた
- **修正 (vllm-serve)**: loglikelihood リクエストの logprobs を 10→1 に削減
  (スコア不変で高速化)、v1.4.1 クライアントの repetition_penalty を
  extra_body 経由で送信、共有出力ディレクトリでのサーバーログ名衝突回避、
  swallow クライアントの evaluate モジュール解決ガード
- **運用ノート (重要)**: **vllm 0.11.2 は gpt-oss (MXFP4 MoE) の生成品質が
  劣化する** (serve 経由では `!!!!...` への出力退化、オフラインでもスコア低下)。
  gpt-oss 系はオフライン・serve とも v2.1.5 環境 (vllm 0.19.1) を使うこと

## 2026-07-23

- **追加**: `--vllm-serve` (qsub.py / sbatch.py)。vLLM サーバーを 1 本立てて
  swallow / llm-jp-eval がエンドポイントを共有する実行モード。モデルロードが
  ジョブ全体で 1 回になり、巨大モデルで大幅に高速化 (gpt-oss-120b TP4 実測で
  従来比 2.4 倍)。`--serve-venv` / `--max-model-len` も追加。スコアは
  サーバー venv の vLLM バージョンに紐づく点に注意
- **追加**: `--client-concurrency` (serve モード)。サーバーに対する in-flight
  プロンプト数 (デフォルト 256)。導入前の直列送信比で swallow が約 3 倍高速化
- **追加**: `vllm-serve/compare_results.py`。2 つの実行結果ディレクトリの
  スコア差分と所要時間を比較表示
- **変更**: sbatch.py の生成ジョブが `[intg-eval] job start/end:` の時刻マーカーを
  出力 (さくらは Slurm accounting が無効なため、事後の所要時間計測に使用)
- **修正 (vllm-serve)**: リクエストの max_tokens がサーバーの max_model_len を
  超える場合のリクエスト毎クランプ

## 2026-07-20

- **追加**: vllm-serve モード本体 (`vllm-serve/` インストーラーとスクリプト一式)
- **追加**: llm-jp-eval v2.1.5 インストーラー (vllm 0.19.1 / torch 2.10)
- **追加**: swallow の transformers-v5 変種 `swallow_v202411-tf5` (オプトイン、
  `--swallow-version v202411-tf5`)
- **追加**: さくらクラスタ (Slurm) 対応: `scripts/sbatch.py` (qsub.py の Slurm 版、
  `--cuda-module` 付き)。インストーラーの uv 管理 Python 化、chabsa データセットの
  Kaggle ミラー化、B200 (sm_100) 対応の torch 2.8 上書き、
  swallow / v1.4.1 の vllm 0.10 互換パッチ、sandbox なし環境での
  mbpp / jhumaneval 自動スキップ
- **制限**: llm-jp-eval v2.1.0 はさくら (B200) 非対応 (locked vllm 0.9.0.1 /
  torch 2.7.0+cu126 が sm_100 で動作しないため sbatch.py の選択肢から除外)

## 2026-05 以前 (ABCI 運用分)

- 2026-05: デフォルト PBS キューを R9920261000 に変更
- 2026-03: dify-sandbox の singularity イメージを digest 固定
- 2026-02: llm-jp-eval v2.1.3 インストーラー、`--apply-chat-template` /
  `--reasoning-parser` / `--chat-template-args` / `--legacy-output`、
  `--llm-jp-eval-max-num-samples`
- 2026-01: PBS キュー / プロジェクトグループ指定 (`--pbs-queue` 等)、
  swallow の vllm 0.10.2 化
