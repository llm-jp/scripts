# CHANGELOG (intg-eval-2508)

利用者に影響する変更のまとめ (新しい順)。実験・検証の詳細は
[VALIDATION.md](./VALIDATION.md) を、vllm-serve モードの設計は
[vllm-serve/README.md](./vllm-serve/README.md) を参照。

## 2026-08-20

- **追加**: 安全性評価コンポーネント `safety-eval` と `--safety-eval` /
  `--safety-eval-benchmarks` / `--safety-eval-judge-model` /
  `--safety-eval-benchmark-size` (qsub.py / sbatch.py)。安全性WG受領の評価
  コード (LLM_Safety_Eva) で JBBQ / JTruthfulQA / AnswerCarefully /
  JSocialFact / safety_boundary を評価する。生成はローカル vLLM (venv は
  動作確認済みの vllm 0.11.2 / transformers 4.57.6 を固定)、採点は JBBQ /
  JTruthfulQA がローカル (JTruthfulQA 分類器はインストール時プリフェッチ +
  オフライン参照)、他3ベンチマークがジャッジAPI (Azure OpenAI または
  OpenAI互換エンドポイント; クレデンシャルはジョブへ転送)。受領コードは
  原則無改変だが、ジャッジ評価器にのみ OpenAI互換エンドポイント対応
  (`OPENAI_BASE_URL` / `OPENAI_API_KEY` / モデル名指定) を追加している。
  集計結果は `<output_dir>/safety-eval/evaluate_count/`。`--vllm-serve` とは
  非対応。**ベンチマークデータは配布 zip から手動配置が必要** (gated データ
  由来のためリポジトリ非同梱; 未配置時はインストールを警告スキップ)。詳細は
  [safety-eval/README.md](./safety-eval/README.md)
- **訂正 (運用ノート)**: 07-31 の「ABCI 計算ノードは外部ネットワーク不可」は
  誤り (当時 DNS が不安定だったための誤認)。計算ノードから外部へは到達できる。
  ただし計算ノードでダウンロード等を行わない方針は維持し、依存物の
  インストール時プリフェッチは今後も必須とする

## 2026-08-12

- **修正 (swallow_v202411-tf5)**: `vllm_causallms-vllm010-compat.patch` が
  vllm 0.19 の venv で import に失敗し、**vllm バックエンドの全タスクが
  `NameError: name 'LLM' is not defined` で即死する問題を解消** (スコアは
  全列 -1.0 になる)。原因は 2 件: (1) ray ベース DP の置換後も import ガード
  に残っていた死にコードの `import ray` (vllm 0.19 は ray を依存に持たないため
  ModuleNotFoundError → ガードが握りつぶし)、(2) vllm 0.19 で
  `vllm.utils.network_utils` に移動した `get_open_port`。パッチから
  `import ray` を削除し、`get_open_port` はフォールバック付き import に変更
  (vllm 0.10 を使う base swallow_v202411 とパッチ共有のまま両対応)。
  ABCI の配備済み tf5 環境には適用済みで、GPU (vllm バックエンド) の実機検証も
  今回が初 (llm-jp-3-150m で EN 全 15 列取得、VALIDATION.md 08-12)
- **修正 (swallow_v202411-tf5, data parallel)**: 同パッチの multiprocessing DP
  経路が vllm 0.19 で `Offline data parallel mode is not supported/useful for
  dense models` により全滅する問題を解消。vllm の `VLLM_DP_*` 環境変数方式を
  やめ、**各ワーカーが `CUDA_VISIBLE_DEVICES` の自ランク分スライスで独立
  エンジンを立てる方式**に変更 (dense / MoE を問わず動く)。あわせてエンジン
  再起動時の GPU メモリ解放待ち競合で WorkerProc init が死ぬ問題に初期化
  リトライ (15s ×3) を追加。DP=8 で完走し DP=1 とスコア一致
  (|diff| ≤ 0.001、VALIDATION.md 08-12)

## 2026-08-07

- **運用ノート (重要, vllm-serve)**: serve の `run_llm-jp-eval-serve.sh` は
  バージョン非依存で、`HF_HOME` / `NLTK_DATA` を駆動先の llm-jp-eval 環境から
  導出し `HF_HUB_OFFLINE=1` で eval する。**07-29 以降のインストーラで
  再インストールした (= prefetch キャッシュ `hf/`・`nltk/` を持つ) バージョン
  でないと、serve 経由の eval が COMET / BERTScore のロードで失敗する**。
  新しい vllm-serve をデプロイしたら、serve で回したい v2.x は再インストール
  しておくこと (offline 経路は各バージョンの env 内スクリプトを使うため無関係。
  v1.4.1 は別スクリプトで COMET / BERTScore を使わないため無関係)。
  ABCI で v2.1.5 の serve 経路を検証済み (書き込み 0 件、VALIDATION.md 08-07)

## 2026-08-05

- **修正 (重要, llm-jp-eval v2.x = v2.1.0 / v2.1.3 / v2.1.5)**: 07-29 の
  「eval の出力先を `OUTPUT_DIR` に変更」で取り切れていなかった、**eval 時
  プロンプト dump による共有インストールディレクトリへの書き込みを解消**。
  `evaluate()` は dump サブコマンドと同じ
  `load_dataset_and_construct_prompt_template()` を呼ぶため eval も毎回
  プロンプトを dump するが、EVAL_OPTS に `--inference_input_dir` が無いと
  dump 先が `output_dir/datasets/<ver>/evaluation/<split>/prompts_<hash>`
  (= 共有 install への symlink) にフォールバックしていた。対応として
  EVAL_OPTS に **`--inference_input_dir=${PROMPT_OUTPUT_DIR}` と
  `--max_num_samples=${MAX_NUM_SAMPLES}`** を追加 (後者は prompt ハッシュの
  構成要素なので dump と揃える必要がある)。これで eval は dump 済みの
  プロンプトを再利用し、共有 install への書き込みが完全にゼロになる
  (無駄な再 dump も無くなる)。v2.1.0 / v2.1.3 / v2.1.5 の
  `run_llm-jp-eval.sh` と serve の `run_llm-jp-eval-serve.sh` に適用。
  **反映には該当バージョンの再インストールが必要** (env 内の run スクリプトを
  差し替えるため)。ABCI で v2.1.5 を再インストールし、eval のみの A/B
  (スコア 163 個完全一致・書き込み 63→0 件) と e2e (書き込み 0 件) を検証済み。
  詳細は VALIDATION.md 2026-08-05 の項
- **運用ノート (スコア比較)**: 同一環境・同一設定でも生成が run-to-run で
  変動する (vLLM の prefix caching / バッチ依存の数値差、`seed=None`)。
  8b-base `--basemodel` を 4 回実行した AVG は 0.58230〜0.58487 で**幅 0.0026**。
  スコアの厳密比較が必要な場合は、既存の推論結果に対して eval のみを
  再実行して比べること

## 2026-07-31

- **追加**: `--judge-gen-reasoning-effort {low,medium,high}`。llm-jp-judge の
  生成リクエストに reasoning_effort を明示的に載せる。**gpt-oss 系 thinking
  モデルを vLLM 0.15.x でサーブする場合は必須** (サーバーが request の
  reasoning_effort を無条件にチャットテンプレートへ注入するため、未指定だと
  None 連結の TypeError で全リクエストが 400 になる)
- **追加**: `--judge-gen-extract-final`。llm-jp-judge の生成応答から Harmony の
  reasoning をクライアント側で除去 (最後の 'assistant final' マーカー以降のみ
  残す)。**thinking モデルの final のみをジャッジに読ませる推奨手段**。
  vLLM の openai_gptoss reasoning parser は非ストリーミング chat を全バージョン
  (0.11.2 / 0.15.1 / 0.19.1) で拒否するため、サーバー側パースは使えない
- **修正**: llm-jp-judge インストーラに upstream クライアントへのパッチを追加:
  None 値の sampling params を送信前に除去 (JSON null が上記 vLLM バグを誘発)、
  choices を含まない 200 応答で全体をクラッシュさせず該当サンプルのみ None 扱い
- **修正 (tf5)**: swallow-tf5 インストーラのデータセットプリフェッチが、隣に
  ベース swallow 環境があればそのキャッシュを流用するように (tf5 venv の新しい
  huggingface_hub は `gsm8k` 等の名前空間なしデータセット ID を拒否するため、
  自前ダウンロードが失敗する)
- **運用ノート (重要・thinking モデルの llm-jp-judge 推奨設定)**:
  `--judge-gen-max-tokens 8192 --judge-gen-reasoning-effort medium
  --judge-gen-extract-final --max-model-len 16384`。
  この構成で offline / serve ともスコアが成立・整合することを ABCI で確認済み
  (VALIDATION.md 07-31)。生の Harmony テキストを読ませた場合とはスコアが
  変わるため、応答範囲を揃えずに比較しないこと
- **運用ノート (ABCI)**: 計算ノードは外部ネットワーク不可 (github / HF に
  届かない)。インストールとプリフェッチは必ずログインノードで行うこと

## 2026-07-29

- **追加**: 生成長・reasoning の制御フラグ (デフォルトはすべて従来挙動):
  `--llm-jp-eval-max-tokens` (生成トークン数の全体上書き; 既定はデータセット毎の
  output_length)、`--llm-jp-eval-reasoning-content-length` (thinking モデル用の
  加算予算、`--reasoning-parser` 必須)、`--judge-gen-max-tokens` (llm-jp-judge
  全ベンチマークの生成上限上書き; 既定 1024)、`--judge-gen-reasoning-parser`
  (生成サーバーの reasoning parser; serve モードでは共有サーバーへの
  `--server-reasoning-parser` になり、completions API を使う llm-jp-eval /
  swallow には影響しない)
- **運用ノート (重要)**: **thinking モデル (llm-jp-4 系等) を llm-jp-judge に
  かける場合は上記フラグが必須**。既定の max_tokens=1024 では reasoning の途中で
  打ち切られ、vLLM 0.15+ (Harmony 自動パース) では応答が空になり全スコア ≈1 の
  無効な評価になる (vLLM 0.11.2 は analysis 込み生テキストを読む)。目安:
  `--judge-gen-max-tokens 8192 --judge-gen-reasoning-parser openai_gptoss
  --max-model-len 16384`。詳細は VALIDATION.md の 07-29 ABCI ラウンド参照
- **変更**: `--max-model-len` が **オフラインモードでも有効に** (従来は
  `--vllm-serve` 専用)。オフラインでは llm-jp-eval v2.1.5 の
  `model.max_model_len` (既定 4096 のまま) と llm-jp-judge のローカルサーバーに
  適用。serve モードでは llm-jp-eval クライアント側の切り詰めエミュレーション
  (従来 4096 固定) とローカルジャッジサーバーにも追従
- **追加**: 評価依存物の**インストール時プリフェッチ + ランタイムオフライン化**。
  swallow は全タスクデータセット + evaluate モジュールを環境内キャッシュ
  (`environment/data/hf/`) に取得し、評価時は `HF_DATASETS_OFFLINE=1` /
  `HF_EVALUATE_OFFLINE=1` で Hub 非接続 (プリフェッチ前の既存環境は従来挙動。
  後付けは `swallow_v202411/scripts/prefetch_en_eval_deps.py` を venv-harness で
  実行)。llm-jp-eval v2.x の COMET / BERTScore / NLTK は次項「修正 (重要)」の
  とおりインストール時に共有キャッシュへ取得する。GPQA / AnswerCarefully は
  gated のためインストール時に承認済み HF_TOKEN が必要
- **修正 (重要, llm-jp-eval v2.x = v2.1.0 / v2.1.3 / v2.1.5)**: eval の出力先を
  共有インストール先からユーザーの `OUTPUT_DIR` に変更し、**結果とメトリクス
  キャッシュを共有インストールディレクトリへ書き込まない**ようにした
  (この時点では eval 時 dump による書き込みが残っていた。2026-08-05 の項で解消)。従来は
  `evaluate_llm.py` の `--output_dir` がデータセット読み込み元・メトリクス
  キャッシュ・結果出力先を兼ねる仕様のため、結果 (`result.json`) と COMET
  チェックポイントを共有 install の `data/llm-jp-eval/` 配下に書き出しており、
  インストールした本人以外 (別ユーザー・`llm-jp` グループ非所属) が実行すると
  パーミッションで失敗し得た。対応として (1) run スクリプトは
  `--output_dir=${OUTPUT_DIR}` とし、読み込み専用の `datasets` / `cache` だけを
  `OUTPUT_DIR` に symlink、(2) **インストーラーが eval 時ダウンロード物 (COMET
  `wmt22-comet-da`、BERTScore の `roberta-large` / `bert-base-multilingual-cased`、
  nltk `punkt`/`punkt_tab`) を共有キャッシュに事前取得**し、eval フェーズは
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` で共有キャッシュを read-only
  参照する。これによりオフラインノード・グループ非所属ユーザーでも動作する。
  prefetch ロジックは全 v2.x 共通のため `installers/_common/prefetch_eval_caches.py`
  に一本化 (バージョン追加時のコピー削減)。offline の `run_llm-jp-eval.sh` と
  serve の `run_llm-jp-eval-serve.sh` の両方に適用。**反映には該当バージョンの
  再インストールが必要** (既存環境には prefetch 済みキャッシュ `hf/`・`nltk/` が
  無いため)。さくら B200 で v2.1.5 / v2.1.3 を offline・serve とも検証済み
  (v2.1.0 はコード修正のみ; B200 非対応で CLI 除外のため未実行)。
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
