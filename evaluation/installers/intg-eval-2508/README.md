# 統合評価基盤インストーラ (v2508)

llm-jp-eval、swallow英語評価、llm-jp-judge (LLM-as-a-Judge)、安全性評価 (safety-eval) を実行可能な統合基盤をインストールするスクリプトです。

- 利用者向けの変更履歴: [CHANGELOG.md](./CHANGELOG.md)
- 実クラスタ (ABCI / さくら) での検証記録 — どのクラスタ・環境でどのコマンドを実行し
  何を確認したか: [VALIDATION.md](./VALIDATION.md)

## セットアップ

### インストール手順 (ローカル)

```bash
# インストール先を指定
INSTALL_DIR="Path/to/install_dir" # FIX_ME

# インストーラーを取得
git clone https://github.com/llm-jp/scripts.git -b feat/intg-eval-installer

# ディレクトリ移動
cd scripts/evaluation/installers/intg-eval-2508

# インストール
bash install.sh $INSTALL_DIR > logs/install-eval.out 2> logs/install-eval.err
```

### インストール手順 (Slurmクラスタ)

2026年度から利用しているさくらインターネットのクラスタ (Slurm, B200 GPU) を想定した手順です。
`scripts/sbatch.py` や各種スクリプト中の "this cluster" はこのクラスタを指します。

```bash
# インストール先を指定 (実験ディレクトリ直下の environment/ を推奨)
INSTALL_DIR="Path/to/experiment_dir/environment" # FIX_ME

# インストーラーを取得
git clone https://github.com/llm-jp/scripts.git -b feat/intg-eval-installer

# ディレクトリ移動
cd scripts/evaluation/installers/intg-eval-2508

# インストール (CPUパーティションへのジョブ投入を推奨)
sbatch --partition=cpu install.sh $INSTALL_DIR
# もしくはログインノードで直接実行
# bash install.sh $INSTALL_DIR > logs/install-eval.out 2> logs/install-eval.err
```

> [!NOTE]
> `uv` がPATH上にある場合、CPythonのソースビルドの代わりにuv管理のスタンドアロンPythonを使用します。
> ノードに `liblzma-dev` / `libbz2-dev` / `libsqlite3-dev` などの開発ヘッダがない環境では
> ソースビルドされたPythonに `_lzma` 等のモジュールが欠落するため、uvの使用を推奨します。

### インストール手順 (on ABCI)

```bash
# インストール先を指定
INSTALL_DIR="Path/to/install_dir" # FIX_ME

# インストーラーを取得
git clone https://github.com/llm-jp/scripts.git -b feat/intg-eval-installer

# GPUノード確保
qsub -N 0206_llm_jp_judge -P gcg51557 -q R9920251000 -v RTYPE=rt_HG -l select=1 -l walltime=3:00:00 -I

# ディレクトリ移動
cd $PBS_O_WORKDIR/scripts/evaluation/installers/intg-eval-2508

# インストール
bash install.sh $INSTALL_DIR > logs/install-eval.out 2> logs/install-eval.err
```

## 環境変数

### HF_HOMEの設定

モデルやデータセットのキャッシュが保存される`HF_HOME`を指定する必要があります。

> [!NOTE]
> HF_HOMEを指定していない場合は動きません。
> また、`/groups/gcg51557/experiments/<experiment_dir>`以下で指定する必要があります。(ABCI上での暫定対応)
> `<experiment_dir>`は適宜、実験ディレクトリ名で置き換えてください。

```
export HF_HOME=/groups/gcg51557/experiments/<experiment_dir>/.cache/huggingface
```

### HF_TOKENの設定

データセットの取得にHF_TOKENが必要な場合があります。

> [!NOTE]
> HF_TOKENを指定していない場合は動きません。

```
export HF_TOKEN=<HuggingFaceのアクセストークン>
```

> [!NOTE]
> llm-jp-judgeの安全性評価用データセット (AnswerCarefully) はgatedデータセットのため、
> インストール時に利用申請の承認済みアカウントの `HF_TOKEN` が必要です。
> 取得できない場合は該当ベンチマークをスキップしてインストールを継続します (詳細は `llm-jp-judge/README.md`)。

> [!NOTE]
> 安全性評価 (safety-eval) のベンチマークデータはリポジトリに含まれません
> (gatedデータセット由来のデータを含むため)。安全性WGからの配布zip
> (`LLM_Safety_Eva_Web-main.zip`) の `benchmark_data/` を
> `safety-eval/LLM_Safety_Eva/benchmark_data/` に配置してから `install.sh` を
> 実行してください。未配置の場合、safety-evalコンポーネントは警告してスキップ
> されます (詳細は `safety-eval/README.md`)。

> [!NOTE]
> 評価用データセット類は**インストール時に**取得します:
> - swallow: 全タスクのデータセットと evaluate メトリクスモジュールを環境内キャッシュ (`environment/data/hf/`) にプリフェッチし、評価時は `HF_DATASETS_OFFLINE=1` / `HF_EVALUATE_OFFLINE=1` でHubに接続しません (プリフェッチ済み環境のみ。GPQAはgatedのためインストール時に承認済み `HF_TOKEN` が必要)。
> - llm-jp-eval v2.1.5: 評価フェーズで使うCOMETチェックポイントを環境内 (`data/llm-jp-eval/cache`) に、BERTScoreモデル等をHFキャッシュにプリフェッチします。**インストール時と評価時で同じ `HF_HOME` を使ってください。**
> - 評価対象モデル自体のダウンロードは対象外です (未キャッシュのHubモデルを評価する場合は事前に取得しておくこと)。

## 評価実行

### 実行

```bash
bash $INSTALL_DIR/scripts/run_eval.sh \
  <model_name_or_absolute_path> \
  <output_dir>
```

### ジョブ形式での実行 (Slurm)

```bash
python3 $INSTALL_DIR/scripts/sbatch.py \
  <model_name_or_absolute_path> \
  <output_dir_absolute_path> \
  --experiment-dir <実験ディレクトリ> \
  [--llm-jp-eval-versions v1.4.1 v2.1.3 v2.1.5] \
  [--basemodel] \
  [--llm-jp-judge] \
  [--partition gpu] \
  [--gpus 1] \
  [--dry-run]
```

- `--experiment-dir` は評価環境をインストールした実験ディレクトリ (`environment/` の親) を指定します。環境変数 `INTG_EVAL_EXPERIMENT_DIR` でも指定可能です。
- その他のオプションは `qsub.py` と共通です (`--help` を参照)。

### 事前学習モデル (ベースモデル) のチェックポイント評価

llm-jp-eval v2.1.5のベースモデル評価用設定 (upstreamの`config_basemodel.yaml`/`vllm_inference_basemodel.yaml`相当) でチェックポイント評価を行う場合は `--basemodel` を指定します。

```bash
python3 $INSTALL_DIR/scripts/sbatch.py \
  <model_name_or_absolute_path> \
  <output_dir_absolute_path> \
  --experiment-dir <実験ディレクトリ> \
  --disable-swallow \
  --llm-jp-eval-versions v2.1.5 \
  --basemodel
```

- `--basemodel` では以下が一括で切り替わります:
  - プロンプトテンプレート: ベースモデル用のAlpaca風テンプレート (`config_basemodel.yaml`)
  - データセット: 4-shotデータセットのみ (`eval_configs/only_4shots.yaml`。英語系データセットを含み、コード実行系は含まないためサンドボックス不要)
  - 推論パラメータ: `add_special_tokens: False`, `temperature: 0.0` を明示的に固定
- 日本語・英語のスコアは `result.json` の `lang_scores` フィールドに言語別に出力されます (英語スコアはllm-jp-evalの英語ベンチマークで算出されるため、チェックポイント評価ではswallow評価を `--disable-swallow` で省略できます)。
- `--vllm-serve` とも併用可能です。

### llm-jp-judge (LLM-as-a-Judge) の実行

```bash
export OPENAI_API_KEY=<APIキー>  # ジャッジにOpenAI APIを使う場合

python3 $INSTALL_DIR/scripts/sbatch.py \
  <model_name_or_absolute_path> \
  <output_dir_absolute_path> \
  --experiment-dir <実験ディレクトリ> \
  --llm-jp-judge \
  [--judge-client {openai,azure,bedrock,vllm}] \
  [--judge-model gpt-4o-2024-08-06] \
  [--judge-benchmark-size N] \
  [--disable-mt-bench]
```

- 生成はターゲットモデルをローカルvLLMサーバで起動して行い、採点はジャッジクライアント経由で行います。結果は `<output_dir>/llm-jp-judge/evaluation/score_table.json` に出力されます。
- ジャッジのAPIクレデンシャル (`OPENAI_API_KEY` / `AZURE_OPENAI_*` / `AWS_*`) は投入時の環境変数からジョブスクリプトへ埋め込まれます。
- 計算ノードから外部APIへ疎通できないクラスタでは `--judge-client vllm --judge-model <ジャッジ用モデル>` を指定するとジャッジもローカルvLLMサーバで実行します。
- `--vllm-serve` と併用した場合、生成は共有vLLMサーバを利用し、ジャッジは共有サーバ停止後に実行されます (ローカルジャッジは解放されたGPUを使用)。詳細は `llm-jp-judge/README.md` を参照してください。

> [!NOTE]
> - singularity等のコンテナランタイムがないノードでは、llm-jp-eval v2系のコード実行系データセット (`mbpp`, `jhumaneval`) とCGカテゴリは自動的にスキップされます (`DISABLE_CODE_EXEC=1` で明示的な無効化も可能)。そのためAVGスコアはコード実行を含む環境での結果と直接比較できません。なおllm-jp-eval v1.4.1のmbppはインプロセスの`exec()`で評価されるため、コンテナランタイムなしでも実行されます。
> - llm-jp-eval v2.1.0はvllm 0.9.0.1 / torch 2.7.0 (cu126) に依存しており、Blackwell世代 (B200等, sm_100) のGPUでは動作しない可能性があります。その場合はvllm 0.11.2 / torch 2.9.0 (cu128) を使用するv2.1.3、またはvllm 0.19.1 / torch 2.10.0を使用するv2.1.5を利用してください。
> - llm-jp-eval v2.1.5は本インストーラ作成時点 (2026-07-20) の最新リリースです。llm-jp-eval-inferenceにはリリースタグがないため、同時点の最新コミット (`c6cd0fa`) を固定しています。v2.1.3までで必要だったHarmony再エンコードパッチはupstreamに取り込み済みのため適用しません。v2.1.5ではデータセットに`jculture_mcq`・`jfinqa`・`structeval`が追加されているため、AVGスコアはv2.1.3以前と直接比較できません。

### 安全性評価 (safety-eval) の実行

安全性WG受領の安全性評価 (JBBQ / JTruthfulQA / AnswerCarefully /
JSocialFact / safety_boundary) を実行する場合は `--safety-eval` を指定します。

```bash
# ジャッジAPI (judge採点系ベンチマークに必要、いずれか):
export AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=...   # Azure OpenAI
export OPENAI_API_KEY=... OPENAI_BASE_URL=...               # OpenAI互換サーバー

python3 $INSTALL_DIR/scripts/sbatch.py \
  <model_name_or_absolute_path> \
  <output_dir_absolute_path> \
  --experiment-dir <実験ディレクトリ> \
  --safety-eval \
  [--safety-eval-benchmarks jbbq_age jtruthfulqa answer_carefully_test JSocialFact-01-test safety_boundary] \
  [--safety-eval-judge-model <ジャッジモデル名>] \
  [--safety-eval-benchmark-size N]
```

- 生成はローカルvLLM (オフライン推論)、採点は `jbbq_age` / `jtruthfulqa` が
  ローカル評価器、その他3ベンチマークがジャッジAPI (Azure OpenAI または
  OpenAI互換エンドポイント。両方設定時はAzure優先) です。クレデンシャルは
  投入時の環境変数からジョブスクリプトへ埋め込まれます。OpenAI互換サーバーを
  使う場合は `--safety-eval-judge-model` でそのサーバーが提供するモデル名を
  指定してください。ジャッジAPIを使わない場合は
  `--safety-eval-benchmarks jbbq_age jtruthfulqa` を指定してください。
- 集計結果は `<output_dir>/safety-eval/evaluate_count/` に出力されます。
- `--vllm-serve` とは併用できません (生成がオフライン推論のため)。
- 詳細 (単体実行、出力レイアウト、実装メモ) は `safety-eval/README.md` を
  参照してください。

### 簡易SFT後の統合評価 (on ABCI)

事前学習チェックポイントに [llm-jp/simple_tuning](https://github.com/llm-jp/simple_tuning)
の簡易SFT (`sft_simple` 設定) を当ててから統合評価を行う場合は、
`qsub_sft_eval.py` を使います。qsub.py 自体には SFT 用のオプションはなく、
このスクリプトが (1) simple_tuning の `qsub_sft.py` で SFT ジョブ (HF→NeMo→SFT→HF、rt_HF)
を投入し、(2) その完了を `-W depend=afterok` で待つ評価ジョブを qsub.py で投入します。

```bash
export HF_HOME=... HF_TOKEN=...                      # qsub.py と同じ
export OPENAI_API_KEY=... OPENAI_BASE_URL=...        # llm-jp-judge のジャッジAPI

python3 $INSTALL_DIR/scripts/qsub_sft_eval.py \
  <base_model_absolute_path> \
  <output_dir_absolute_path> \
  [--param-name llmjp4_8b_4K] \
  [--sft-num-nodes 1] \
  [--reasoning-effort low] \
  [--pbs-queue R9920261000] \
  -- [qsub.py に渡す追加オプション...]
```

- 出力は `<output_dir>/sft/` (SFT 成果物、最終チェックポイントは `sft/converted/final_hf`) と
  `<output_dir>/eval/` (qsub.py の出力) に分かれます。両ジョブIDと実行コマンドは
  `<output_dir>/qsub_sft_eval.json` に記録されます。
- 評価側には simple_tuning の出力 (llm-jp-4 系、Harmony chat template) 向けの設定が固定で付きます:
  llm-jp-eval v2.1.5 + `--apply-chat-template --reasoning-parser llmjp4`、
  llm-jp-judge + `--judge-gen-extract-final`。
  reasoning parser は llm-jp-eval-inference (v2.1.5 で固定しているコミット c6cd0fa) の
  llm-jp-4 専用アダプタ `llmjp4` です。llm-jp-4 は Harmony 形式で出力しますが語彙が gpt-oss と
  異なるため、vLLM の `openai_gptoss` をそのまま指定すると v2.1.5 のオフライン推論では
  全サンプルの generated が空になります (v2.1.3 までとは挙動が異なります)。ジャッジ生成の `--judge-gen-max-tokens` は
  `--param-name` の文脈長サフィックスから導出します (`_4K` → 2048、上限 8192。プロンプト + 生成長が
  文脈長に収まる必要があるため)。文脈長 (`--max-model-len`) はチェックポイントの config.json に
  任せます (4K 設定のモデルに 16384 を指定すると vLLM が起動を拒否します)。
  `--reasoning-effort` は llm-jp-eval とジャッジ生成の両方に適用されます (デフォルト `low`。
  `sft_simple` の学習データが reasoning_low 系のため)。
- `--` 以降は qsub.py にそのまま渡されます (例: `--judge-model <名前> --pbs-queue rt_HG --rtype rt_HG --disable-swallow`)。
  評価ジョブのキュー・リソースは qsub.py のデフォルト (R9920261000 / rt_HG) で、SFT ジョブ側とは独立です。
- SFT の投入前に qsub.py を `--dry-run` で事前検証するため、評価側の引数ミスで SFT ジョブだけが残ることはありません。
  SFT ジョブが失敗した場合は PBS が評価ジョブを削除します。
- simple_tuning の checkout は `--simple-tuning-dir` または環境変数 `SIMPLE_TUNING_DIR` で指定します
  (デフォルト: `/groups/gcg51557/experiments/0366_simple_tuning/simple_tuning`。SFT 環境・学習データ・
  tokenizer はこのディレクトリ配下のものを参照します)。wandb ロガーは投入者に資格情報が無い前提で
  無効化しています (`--sft-override exp_manager.create_wandb_logger=True` で戻せます)。
- スモークテストには `--sft-override trainer.sft.max_steps=20` が使えます。8b・1ノードで
  変換込み約12分です (本番の1エポックは約40分)。SFT の中間生成物 (`sft/converted/input_nemo`, `sft/train`)
  は 8b で約70GBあるので、評価が終わったら不要なら削除してください。

### ジョブ形式での実行 (on ABCI)

```
python3 $INSTALL_DIR/scripts/qsub.py \
  <model_name_or_absolute_path> \
  <output_dir_absolute_path> \
  [--experiment_dir /groups/gcg51557/experiments/0182_intg_eval_2507] \
  [--swallow_version v202411] \
  [--llm_jp_eval_version v1.4.1] \
  [--job_name 0182_intg_eval] \
  [--rtype rt_HG] \
  [--select 1] \
  [--options ""]
```

必須引数:
- `<model_name_or_absolute_path>` モデル名またはモデルの絶対パス
- `<output_dir_absolute_path>` 結果出力先の絶対パス

オプション引数:
- `--experiment_dir` 評価環境がインストールされた実験ディレクトリのパス (デフォルト: `/groups/gcg51557/experiments/0182_intg_eval_2507`)
- `--swallow_version` swallowバージョン（デフォルト: `v202411`）現状は`v202411`のみに対応。
- `--llm_jp_eval_version` llm-jp-evalバージョン（デフォルト: `v1.4.1`）現状は`v1.4.1`のみに対応。
- `--job_name` ジョブ名（デフォルト: `0182_intg_eval`）
- `--rtype` リソースタイプ（デフォルト: `rt_HG`）
- `--select` GPU数またはノード数（デフォルト: 1）
- `--options` qsubスクリプトへの追加オプション
