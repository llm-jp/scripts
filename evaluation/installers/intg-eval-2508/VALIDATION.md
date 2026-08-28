# 検証記録 (intg-eval-2508)

本ディレクトリのインストーラー / 実行スクリプトに対する実クラスタでの検証実験の記録。
実験は複数クラスタで行うため、各ラウンドに **どのクラスタ・どの環境で・どのコマンドを
実行したか** を必ず残す。新しい検証を行ったら本ファイルに追記すること。

対象クラスタと環境:

| 呼称 | クラスタ | 環境 (インストール先) | ジョブ投入 |
|---|---|---|---|
| さくら | さくらインターネット (Slurm, B200×8/ノード, sm_100, コンテナランタイムなし) | `/data/experiments/0219_dev_eval_script/environment` | `scripts/sbatch.py` |
| ABCI | ABCI (PBS, H200) | `/groups/gcg51557/experiments/0230_intg_eval_2509/environment` | `scripts/qsub.py` |

スコア・所要時間の比較には `vllm-serve/compare_results.py` を使用する
(2 つの出力ディレクトリを渡すと llm-jp-eval / swallow のスコア差分と
elapsed を表示する。所要時間はジョブスクリプトが `logs/sbatch.out` に出力する
`[intg-eval] job start/end:` マーカーから算出)。

## サマリ

| 日付 | クラスタ | 内容 | 結論 |
|---|---|---|---|
| 07-14〜15 | さくら | オフライン移植の e2e + ABCI とのスコア比較 | swallow / v1.4.1 / v2.1.3 とも動作、スコアは ABCI と整合 |
| 07-20 | さくら | v2.1.5 / swallow-tf5 のスモーク | 完走 |
| 07-23 | ABCI | vllm-serve 実 GPU 検証 (150m) | スコア一致性 OK |
| 07-24 | ABCI | --client-concurrency / v1.4.1 serve | swallow serve 190分→65分、v1.4.1 一致 |
| 07-23〜24 | さくら | serve vs オフライン (150m / 8b-base / gpt-oss-120b) | スコア一致、120b で 2.39 倍。障害 2 件を修正 |
| 07-26 | さくら | gpt-oss-120b × vllm 0.19.1 サーバー | 0.11.2 の出力退化が解消。eval OOM を発見 |
| 07-29 | さくら | eval フェーズ分割 (--phase) の検証 | OOM 恒久対策の動作確認 |
| 07-29 | さくら | eval 出力先 / キャッシュ事前取得 (v2.1.5・v2.1.3, 150m, offline + serve) | 共有 install へ書き込みゼロ・完走を確認 |
| 07-29 | ABCI | llm-jp-judge 導入 + offline/serve テスト (8b-thinking) | serve 完走・スコア取得。thinking モデルの空応答問題を発見 → max-tokens/reasoning-parser 制御を追加 |
| 07-29 | ABCI | swallow プリフェッチ適用 + オフライン読込確認 | 全10タスク取得 (545MB)、HF_HUB_OFFLINE=1 でも読込 OK |
| 07-31 | ABCI | v2.1.5 / swallow-tf5 導入 + --basemodel 実機テスト | basemodel AVG 0.583 (さくらと一致)。計算ノードは外部ネットワーク不可と判明 |
| 07-31 | ABCI | thinking モデル × llm-jp-judge の完動設定確立 | 障害3件を切り分け。effort 明示 + final 抽出で offline/serve ともスコア成立・整合 |
| 08-05 | ABCI | eval 出力先 / キャッシュ事前取得 (bc6b95a) の v2.1.5 再インストール + 検証 | prefetch・オフライン eval は成立。**共有 install への書き込みが残存**していることを発見 (eval 時 dump が `datasets/` symlink 経由で書く) |
| 08-05 | ABCI | 上記の原因特定と修正 (EVAL_OPTS に `--inference_input_dir` / `--max_num_samples`) | eval のみ A/B でスコア 163 個完全一致・書き込み 63→0 件。e2e も 0 件 |
| 08-07 | ABCI | serve 経路の検証 + vllm-serve 再デプロイ (v2.1.5) | serve も書き込み 0 件で完走。**serve は駆動先バージョンの再インストールを要求する**ことが判明 |
| 08-12 | ABCI | qsub.py 配備更新 + swallow-tf5 の GPU (vllm 0.19) 初検証 | パッチの import 非互換 2 件を修正して完走 (150m、EN 15 列取得、68 分) |
| 08-12 | ABCI | swallow-tf5 の DP>1 (mp 経路) 初検証 | vllm 0.19 の DP モード拒否を独立エンジン方式に書き換えて完走。DP=8 と DP=1 のスコア差 \|diff\| ≤ 0.001 |
| 08-28 | ABCI | safety-eval 導入 + e2e 検証 (150m, 全5ベンチマーク) | 障害3件 (Juman++ 依存欠落 / thinking ジャッジの max_tokens / 長大生成での Juman++ 失敗) を修正して完走。共有 install への書き込み 0 件 |

---

## 2026-07-14〜15 さくら: オフライン移植の e2e 検証

インストーラー一式のさくら移植 (uv 管理 Python 化、chabsa ミラー、B200 対応の
torch 上書き、sbatch.py 新設など) 後の end-to-end 検証。

```bash
cd /data/experiments/0219_dev_eval_script
# スモーク (150m, swallow + v1.4.1 + v2.1.3, max_num_samples=100)
python3 sbatch.py llm-jp/llm-jp-3-150m $PWD/results/smoke-test-20260714 \
  --llm-jp-eval-versions v1.4.1 v2.1.0 v2.1.3   # v2.1.0 はこの検証で B200 非対応と判明し CLI から除外
# ABCI とのスコア比較 (llm-jp-4-8b-thinking, v2.1.3, reasoning parser 付き)
python3 sbatch.py llm-jp/llm-jp-4-8b-thinking $PWD/results/llm-jp-4-8b-thinking-20260715 \
  --disable-swallow --llm-jp-eval-versions v2.1.3 --llm-jp-eval-max-num-samples 10 \
  --apply-chat-template --reasoning-parser openai_gptoss --chat-template-args reasoning_effort=medium
```

- swallow / v1.4.1 / v2.1.3 とも完走。ABCI との比較 (8b-thinking, temperature=1.0)
  は CG 調整後 AVG 0.5617 vs 0.5684 でサンプリングノイズ範囲
- データ整合性: v1.4.1 完全一致、v2.1.3 は jamc-qa 3/2309 サンプルのみ上流改訂差、
  swallow は行レベル一致

## 2026-07-20 さくら: v2.1.5 / swallow-tf5 のスモーク

```bash
cd /data/experiments/0219_dev_eval_script
python3 sbatch.py llm-jp/llm-jp-3-150m $PWD/results/smoke-test-v2.1.5-20260720-2 \
  --disable-swallow --llm-jp-eval-versions v2.1.5
python3 sbatch.py llm-jp/llm-jp-3-150m $PWD/results/smoke-test-swallow-tf5-20260720 \
  --swallow-version v202411-tf5 --disable-llm-jp-eval
```

いずれも完走 (v2.1.5 は新データセット群を含むため AVG は v2.1.3 と比較不可)。

## 2026-07-23 ABCI: vllm-serve 実 GPU 検証

H200 1 枚、llm-jp/llm-jp-3-150m、swallow_v202411 + llm-jp-eval v2.1.3、
サーバー venv は vllm 0.11.2 の専用 venv。セットアップと手順は
`vllm-serve/HANDOFF-ABCI.md` を参照。

- `echo=True + max_tokens=0 + logprobs` の prompt logprobs 取得、トークン ID
  配列プロンプト、`truncate_prompt_tokens` (extra_body) すべて動作
- swallow: オフライン (vllm 0.10.2) とエンジン版が異なる前提で全メトリクス
  |diff| ≤ 0.006 (hellaswag 0.0002, mmlu 0.0016, bbh_cot 0.0041)
- llm-jp-eval v2.1.3: 同一 vllm 0.11.2 同士で AVG 0.1060 (offline) vs 0.1083 (serve)。
  temperature=1.0 / seed=None のためタスク単位はノイズ幅 ±0.03-0.13、集計値に系統差なし
- 発見・修正: vllm 0.11.2 venv の `vllm serve` 起動不可 (openai 1.99.5 非互換)、
  output_length がコンテキストを超えるデータセット (jhle=8192) の 400 エラー
  (リクエスト毎クランプで解決)

## 2026-07-24 ABCI: --client-concurrency / v1.4.1 serve

構成は前ラウンドと同じ。

- swallow: `--client-concurrency 256` + `logprobs=1` で loglikelihood 53-58 req/s
  (オフライン同等)。logprobs=1 はスコアを小数第 4 位まで変えない。
  所要時間は初期実装 190 分 → 約 65 分 (オフライン 79 分、ロード 6 回 → 1 回の分だけ短い)
- llm-jp-eval v1.4.1 (inference_openai_v1.py): e2e 完走。greedy 同士でオフライン
  (vllm 0.10.2) と AVG 0.1805 vs 0.1786、52 メトリクス中 19 完全一致、最大差 0.06 (jsick)
- evaluate モジュールの comparison 版混入 (Hub 取得失敗時の fallback) を実地確認し、
  run-swallow-serve.sh にオフライン解決ガードを追加

## 2026-07-23〜24 さくら: serve vs オフライン比較

サーバー venv はスコア互換性のため v2.1.3 の vllm venv (vllm 0.11.2) を明示
(`SERVE_VENV=$ENV/llm-jp-eval-v2.1.3/environment/src/llm-jp-eval/llm-jp-eval-inference/inference-modules/vllm/.venv`)。

```bash
cd /data/experiments/0219_dev_eval_script
# オフライン基準 (07-23 投入、完走: 2:10:31 / 3:42:19)
python3 sbatch.py llm-jp/llm-jp-4-8b-base $PWD/results/cmp-offline-8b-base-20260723 \
  --llm-jp-eval-versions v2.1.3
python3 sbatch.py openai/gpt-oss-120b $PWD/results/cmp-offline-gptoss-120b-20260723 \
  --llm-jp-eval-versions v2.1.3 --apply-chat-template --gpus 4 --tensor-parallel-size 4

# serve 側 (07-23 投入分は全滅 → 下記 2 件を修正して 07-24 再投入)
python3 sbatch.py llm-jp/llm-jp-3-150m $PWD/results/serve-all-150m-20260724 \
  --vllm-serve --serve-venv $SERVE_VENV --llm-jp-eval-versions v1.4.1 v2.1.3 v2.1.5
python3 sbatch.py llm-jp/llm-jp-4-8b-base $PWD/results/basemodel-off-8b-base-20260724 \
  --vllm-serve --disable-swallow --llm-jp-eval-versions v2.1.5
python3 sbatch.py llm-jp/llm-jp-4-8b-base $PWD/results/basemodel-on-8b-base-20260724 \
  --vllm-serve --disable-swallow --llm-jp-eval-versions v2.1.5 --basemodel
python3 sbatch.py openai/gpt-oss-120b $PWD/results/cmp-serve-gptoss-120b-20260724 \
  --vllm-serve --serve-venv $SERVE_VENV \
  --llm-jp-eval-versions v2.1.3 --apply-chat-template --gpus 4 --tensor-parallel-size 4
```

- **起動失敗 2 件を発見、根本修正 (コミット 42e5b32)**:
  1. v2.1.3 venv は lock の openai 1.99.5 が vllm 0.11.2 の serve と非互換 →
     インストーラーで `openai==1.99.1` をピン
  2. サーバーを venv 未 activate で起動していたため MoE カーネル JIT が PATH 上の
     `ninja` を見つけられず `FileNotFoundError` (他チームの 291B qwen3-moe ジョブも同因) →
     serve_common.sh が PATH/VIRTUAL_ENV を設定して起動
- 150m (4 フレームワーク、1 サーバー、計 1:28:46): v2.1.3 AVG 差 -0.004 (temp=1.0
  ノイズ)、v1.4.1 AVG 差 -0.0003 (15/52 完全一致)、swallow 全 15 メトリクス
  |diff| ≤ 0.0042、v2.1.5 AVG 差 +0.004 → **さくらでもスコア一致性を確認**
- --basemodel (8b-base, v2.1.5): 共通 77 メトリクスで AVG 0.361 → 0.583
  (base モデルの本来性能が出る)。4-shot 系のみ・temperature 0・lang_scores JA/EN 分離を確認
- gpt-oss-120b: 所要 3:42:19 (オフライン) → 1:33:00 (serve) = **2.39 倍**。
  ただし serve (0.11.2) の生成出力が `!!!!...` に退化 (math_500 499/500、
  gsm8k 116/2638; オフラインは 0 件) → 次ラウンドへ

## 2026-07-26 さくら: gpt-oss-120b × vllm 0.19.1 サーバー

`--serve-venv` を外し、auto-detect (= v2.1.5 venv / vllm 0.19.1) でサーブ。

```bash
cd /data/experiments/0219_dev_eval_script
python3 sbatch.py openai/gpt-oss-120b $PWD/results/cmp-serve-gptoss-120b-v215-20260724 \
  --vllm-serve \
  --llm-jp-eval-versions v2.1.3 --apply-chat-template --gpus 4 --tensor-parallel-size 4
```

- **`!!!!` 退化は解消** (math_500 0/500、gsm8k 0/2638)。vllm 0.11.2 の
  MXFP4 MoE 問題と確定
- swallow (vs オフライン): loglikelihood 系 |diff| ≤ 0.007、math_500 0.314 vs 0.318。
  swallow フェーズ 1:32:59 で速度も維持
- llm-jp-eval v2.1.3: **serve (0.19.1) AVG 0.103 vs オフライン (0.11.2) AVG 0.052**。
  BBH 0.11→0.36、drop_f1 0.01→0.56 など生成系は 0.19.1 が全面的に上 →
  **vllm 0.11.2 はオフラインエンジンでも gpt-oss の生成品質が劣化する**。
  gpt-oss / MXFP4 系は offline でも v2.1.5 環境 (vllm 0.19.1) を使うこと
- **発見**: vllm 0.19.1 は `--gpu-memory-utilization 0.9` でも GPU をほぼ全量
  (実測 98%) 確保するため、サーバー常駐のまま走る eval フェーズ (BERTScore) が
  CUDA OOM。inference は完了していたため eval のみ手動再実行
  (`evaluate_llm.py eval` 後に `update_result_json.py` を忘れないこと)

## 2026-07-29 さくら: eval フェーズ分割の検証

上記 OOM の恒久対策 (コミット e45b4d4): run_llm-jp-eval[-v1]-serve.sh に
`--phase inference|eval` を追加し、run_eval_serve.sh を「全バージョン
dump+inference → サーバー停止 → 全バージョン eval」の 2 パス構成に変更。

```bash
cd /data/experiments/0219_dev_eval_script
python3 sbatch.py llm-jp/llm-jp-3-150m $PWD/results/phase-split-validation-20260729 \
  --vllm-serve --serve-venv $SERVE_VENV --disable-swallow \
  --llm-jp-eval-versions v1.4.1 v2.1.3 --llm-jp-eval-max-num-samples 10
```

ログ上で「inference ×2 → stopping vllm server → eval ×2」の順序を確認、
両バージョンの result.json が正常に出力された (v1.4.1: 52 メトリクス、v2.1.3: 148)。

## 2026-07-29 さくら: eval 出力先の分離 + eval キャッシュの事前取得 (v2.x, offline + serve)

eval が共有インストールディレクトリへ書き込むことによる他ユーザーの
パーミッション問題を解消する変更の検証。変更点は 2 つ:
(1) run スクリプトを `--output_dir=${OUTPUT_DIR}` とし、読み込み専用の
`datasets` / `cache` を `OUTPUT_DIR` に symlink、(2) インストーラーが eval 時
ダウンロード物 (COMET / BERTScore エンコーダ / nltk) を共有キャッシュに事前取得し
(全 v2.x 共通のため `installers/_common/prefetch_eval_caches.py` に一本化)、
eval を `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` で read-only 参照。
offline (`run_llm-jp-eval.sh`) と serve (`run_llm-jp-eval-serve.sh`) の両方、
v2.1.0 / v2.1.3 / v2.1.5 に適用。

**インストーラーの prefetch を確認** (再インストール後): 共有 `data/llm-jp-eval/`
配下に `cache/models--Unbabel--wmt22-comet-da`、`hf/hub/{roberta-large,
xlm-roberta-large,bert-base-multilingual-cased}` (xlm-roberta-large は COMET の
`load_from_checkpoint` が牽引)、`nltk/tokenizers/{punkt,punkt_tab}` が生成。
torch は 2.8.0 のまま (prefetch の `uv run` は torch 上書きの前に置いたので
巻き戻らない)。

### ラウンド1: offline (throwaway 環境で機構を先行検証)

既存環境を壊さないよう新インストーラーで別ディレクトリに新規インストールし、
offline eval を実行 (150m, max_num_samples=2)。

- eval 完走 (6.5 分)、result.json に 163 スコア + lang_scores、COMET/BERTScore も算出
- **共有 install への書き込みゼロ (決定的証拠)**: eval 後、共有 install の
  `data/llm-jp-eval/results` は存在せず、`cache/`・`hf/`・`nltk/` の mtime は
  全てインストール時刻で eval 時間帯の更新ゼロ。結果・prompts・yaml は全て
  `OUTPUT_DIR` 配下、`datasets`/`cache` は共有 install への symlink

### ラウンド2: canonical 再インストール + serve

canonical パス (`environment/llm-jp-eval-v2.1.5`・`-v2.1.3`) を新インストーラーで
**in-place 再インストール** (既存 clone は clean・BUG_FIX 空なので git checkout は
noop、破壊的操作を回避)。両環境とも torch 2.8.0 / prefetch キャッシュ / 修正済み
run スクリプトを確認。その後 serve 経路を検証:

```bash
# canonical 再インストール (jobs 2300=v2.1.5 / 2301=v2.1.3, cpu, in-place)
cd installers/llm-jp-eval-v2.1.5   # および v2.1.3
sbatch --partition=cpu --export=ALL,AQUA_GLOBAL_CONFIG=$HOME/aqua.yaml \
  install.sh /data/experiments/0219_dev_eval_script/environment/llm-jp-eval-v2.1.5

# serve 検証 (job 2302, gpu): 共有サーバー1本で v2.1.5 + v2.1.3 を serve 経路実行
bash environment/vllm-serve/run_eval_serve.sh llm-jp/llm-jp-3-150m <OUTPUT_DIR> \
  --experiment-dir /data/experiments/0219_dev_eval_script \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --llm-jp-eval-versions v2.1.5 v2.1.3 --max-num-samples 2
```

- serve 完走 (11 分)。ログ順序も想定どおり「v2.1.5 inference → v2.1.3 inference →
  サーバー停止 → v2.1.5 eval → v2.1.3 eval」
- 結果は `OUTPUT_DIR/llm-jp-eval/{v2.1.5,v2.1.3}/results/result.json` に出力
  (v2.1.5: 163 スコア、v2.1.3: 148 スコア、両者 COMET/BERTScore 算出)
- **両 canonical install への書き込みゼロ**: eval 時間帯 (10:48–10:59) に
  `data/llm-jp-eval/` 配下の更新なし (最新ファイルは再インストール時刻 10:43–10:44)。
  旧環境の空 `results/` ディレクトリ (pre-fix の遺物) は掃除済み
- 補足: この検証で従来の未解決事項「他ユーザーでの実測未実施 (llm-jp グループ
  限定)」の根本原因 (共有 install への eval 書き込み) が offline / serve とも解消。
  **v2.1.0 はコード修正のみ** (B200 非対応で CLI 除外のため実行検証は不可)

## 2026-07-29 ABCI: llm-jp-judge 導入 + offline/serve 両モードテスト

llm-jp-judge v2.0.0 サブインストーラを `environment/llm-jp-judge` に導入
(AnswerCarefully は gated 未承認のためスキップ → quality_ja / culture_ja /
safety_boundary_ja / MT-Bench ja+en の 5 ベンチマークで実施)。
対象 llm-jp/llm-jp-4-8b-thinking、ジャッジは OpenAI 互換 API 経由の
llm-jp-4-32b-a3b-thinking、各ベンチマーク 10 件、rt_HG (H200 1 枚)。

```bash
# offline (2079312.pbs1) / serve (2079314.pbs1)
python3 qsub.py llm-jp/llm-jp-4-8b-thinking \
  /groups/gcg51557/experiments/0230_intg_eval_2509/results/judge-{offline,serve}-20260729 \
  --disable-swallow --disable-llm-jp-eval \
  --llm-jp-judge --judge-model llm-jp-4-32b-a3b-thinking --judge-benchmark-size 10 \
  --pbs-queue rt_HG --rtype rt_HG [--vllm-serve]
```

- **serve モード完走** (API エラー率 0%): quality_ja 総合 4.89/5、mt_bench_ja 8.21、
  mt_bench_en 8.26、culture_ja 4.1 (許容 80%)、safety_boundary_ja 2.1
- offline: ジャッジフェーズが一時的な `APIConnectionError` で失敗 → 生成済み出力に
  対する `--judge-only` 再実行 (ログインノード、GPU 不要) で復旧。**リカバリ機能の実地確認**
- **thinking モデルの空応答問題を発見**: offline (judge venv = vllm 0.15.1) の
  生成応答が 10/10 件空になり全スコア ≈1 の無効な評価に。serve (共有サーバ
  vllm 0.11.2) は生テキスト (analysis+final) を返すためスコアは出るが、ジャッジが
  analysis 込みの応答を読む。**両モードのスコアは thinking モデルでは非互換**
- **【訂正 (07-31)】** 当初この空応答を「max_tokens=1024 で final 到達前に打ち切り」
  と診断したが誤り。真因は vLLM 0.15.1 が request の reasoning_effort を無条件で
  チャットテンプレート変数に注入するため、未指定 (None) だと llm-jp-4 系
  テンプレートの `"Reasoning: " + reasoning_effort` が TypeError → 全リクエスト
  400 (応答は None)。07-31 ラウンド参照
- 対策 (コミット e379d7d / 5dae2e1): `--judge-gen-max-tokens` /
  `--llm-jp-eval-max-tokens` / `--llm-jp-eval-reasoning-content-length` を追加し、
  offline の `--max-model-len` 対応も実装。thinking モデルの完動設定は
  07-31 ラウンドで確立 (reasoning parser はサーバ側では使えないと判明)

## 2026-07-29 ABCI: swallow データセットプリフェッチの適用

コミット cae1342 / 0256357 のインストール時プリフェッチを既存の ABCI 環境に
後付け適用 (`prefetch_en_eval_deps.py` を venv-harness で実行)。

- 全 10 タスク (triviaqa / gsm8k / openbookqa / hellaswag / xwinograd_en /
  squadv2 / mmlu / bbh_cot_fewshot / math_500 / **gpqa: gated、承認済みトークンで
  取得成功**) + evaluate モジュール (exact_match / squad_v2) を
  `swallow_v202411/environment/data/hf/` に取得、計 545MB
- プリフェッチ中に comparison 版 exact_match がキャッシュに混入することを実地確認
  (07-24 障害と同じ変種) → プリフェッチスクリプトが取得直後に自動削除するよう修正
- サニティ: `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1` の
  完全オフラインで exact_match の計算と gsm8k (test 1319 件) の読込を確認
- オフラインガード付き run-eval.sh / run-swallow-serve.sh をデプロイ済み。
  以後、この環境の swallow 評価はデータセット/メトリクスの Hub アクセスなしで動作
  (評価対象モデルの取得のみオンライン)

## 2026-07-31 ABCI: llm-jp-eval v2.1.5 / swallow-tf5 導入 + --basemodel 実機テスト

保留だった ABCI 側への v2.1.5 / swallow-tf5 導入と、ベースモデル評価モードの
実機テスト。インストールで得たインフラ知見も記録する。

- **インフラ知見**:
  - **ABCI 計算ノードは外部ネットワーク不可** (github.com / HF の DNS 解決不能)。
    インストール・プリフェッチは必ずログインノードで行うこと。ジャッジ API
    (OPENAI_BASE_URL) は内部エンドポイントのため計算ノードから到達可能
  - この日のログインノードは DNS が断続的に不安定 (数分おきに瞬断)。
    インストーラは再実行で前進する (ダウンロード済み分は再利用) ため、
    リトライループで凌げる
  - PBS ジョブでは ~/.bashrc が読まれず aqua シム経由の uv が解決できない
    (実体バイナリを PATH に入れれば動くが、上記の通り計算ノードでは無意味)
  - **tf5 venv の新しい huggingface_hub は名前空間なしデータセット ID
    (`gsm8k` 等) の hf:// URI を拒否**するためプリフェッチが失敗する →
    ベース swallow 環境のキャッシュ流用で解決 (同じ datasets==2.21.0 なので
    形式互換。インストーラにフォールバックとして組み込み済み)
- **--basemodel 実機テスト** (llm-jp-4-8b-base, v2.1.5, offline, H200 1枚):

  ```bash
  python3 qsub.py llm-jp/llm-jp-4-8b-base $RESULTS/basemodel-8b-base-20260731 \
    --disable-swallow --llm-jp-eval-versions v2.1.5 --basemodel \
    --pbs-queue rt_HG --rtype rt_HG
  ```

  **AVG 0.5831** — さくらの serve モード検証 (07-24, 0.583) と一致し、
  クラスタ間整合を確認。lang_scores の JA/EN 分離出力も確認。
  メトリクスプリフェッチ済みのため所要 ~7 分

## 2026-07-31 ABCI: thinking モデル × llm-jp-judge の完動設定確立

07-29 に発見した空応答問題の根本解決。障害は 3 件の複合だった:

1. **llm-jp-judge が未設定 sampling params を JSON null で送信**
   (`reasoning_effort: null`) → インストーラパッチで None 値を送信前に除去
2. **vLLM 0.15.1 のバグ**: request の reasoning_effort を**無条件で**チャット
   テンプレート変数に注入 (chat_completion/serving.py) → 未指定だと None が
   テンプレートの `is not defined` ガードを素通りし `"Reasoning: " + None` の
   TypeError で全リクエスト 400。**07-29 の offline 空応答の真因はこれ**
   (max_tokens 打ち切り説は誤診断)。対策: `--gen-reasoning-effort` で明示送信
3. **vLLM の openai_gptoss reasoning parser は非ストリーミング chat 全滅**:
   0.11.2 = 500、0.15.1 = HTTP 200 にエラー body (choices=null)、0.19.1 = 501。
   サーバ側での final 抽出は不可 → `--gen-extract-final` (生成 jsonl の各応答を
   最後の 'assistant final' マーカー以降に切り詰めるクライアント側後処理) を実装

**完動設定** (llm-jp-4-8b-thinking, ジャッジ llm-jp-4-32b-a3b-thinking, 各10件,
AnswerCarefully 取得済みで全7ベンチマーク):

```bash
python3 qsub.py llm-jp/llm-jp-4-8b-thinking $RESULTS/judge-thinking-{offline,serve}-20260731e \
  --disable-swallow --disable-llm-jp-eval \
  --llm-jp-judge --judge-model llm-jp-4-32b-a3b-thinking --judge-benchmark-size 10 \
  --judge-gen-max-tokens 8192 --judge-gen-reasoning-effort medium \
  --judge-gen-extract-final --max-model-len 16384 \
  --pbs-queue rt_HG --rtype rt_HG [--vllm-serve]
```

- 生成成功率 100% (offline = judge venv vllm 0.15.1 / serve = 共有サーバ 0.11.2)。
  extract-final 後の応答をジャッジが採点
- **offline vs serve スコア** (生成・ジャッジとも temperature ありのため n=10 では
  ノイズ幅あり): quality 総合 4.8 / 5.0、mt_bench_ja 9.63 / 9.42、
  mt_bench_en 7.8 / 7.7、safety_ja 4.7 (違反10%) / 4.7 (違反10%)、
  safety_boundary 1.9 / 1.9、culture 3.3 / 2.6 → **両モード整合**
- 参考: 07-29 serve (生 Harmony テキストをジャッジが読んだ場合) は
  mt_bench_ja 8.21 / quality 4.89 → **final のみを読ませるとスコアが変わる**。
  比較には応答範囲の統一が必須 (llm-jp-judge のクライアントは
  message.content しか読まないため、upstream の想定は final のみ)
- serve ジョブのジャッジフェーズは内部 API への一時的な疎通断で失敗 →
  ログインノードから `--judge-only` で再実行 (07-29 に続き 2 回目。
  リカバリ手順として定着)

## 2026-08-05 ABCI: eval 出力先 / キャッシュ事前取得 (bc6b95a) の再インストール検証

さくらで実施した bc6b95a (eval の出力先を `OUTPUT_DIR` に分離 + eval 時
ダウンロード物のインストール時プリフェッチ) を ABCI に反映するため、
v2.1.5 を **in-place 再インストール** して検証した (v2.1.3 は今回対象外。
ABCI の v2.1.3 は旧 prefetch 未適用で env 内の run スクリプトも旧版のまま
整合しているため、そのまま動作する)。

```bash
# 再インストール (ログインノード。計算ノードは外部ネットワーク不可)
cd installers/llm-jp-eval-v2.1.5
export HF_HOME=/groups/gcg51557/experiments/0219_dev_eval_script/.cache/huggingface
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' ~/.bashrc | cut -d= -f2-)
bash install.sh /groups/gcg51557/experiments/0230_intg_eval_2509/environment/llm-jp-eval-v2.1.5

# 検証 (PBS rt_HG, H200×1)。results/outputdir-fix-20260805/qsub.sh
#   A: llm-jp-4-8b-base --basemodel --max_num_samples 100  (07-31 と同一条件)
#   B: llm-jp-3-150m --max_num_samples 2  (full config, DISABLE_CODE_EXEC=1)
#   C: A の再実行 (run-to-run 変動の測定)
```

- **prefetch は期待どおり生成** (~4.3GB): `cache/models--Unbabel--wmt22-comet-da`
  (ckpt 2.3GB)、`hf/hub/{roberta-large,bert-base-multilingual-cased,xlm-roberta-large}`
  (2.1GB)、`nltk/tokenizers/{punkt,punkt_tab}` (64MB)。torch は 2.8.0+cu128 のまま
  (prefetch の `uv run` を torch 上書きより前に置いた効果を確認)。env 内の
  `run_llm-jp-eval.sh` はリポジトリと完全一致
- **eval フェーズは完全オフラインで成立**: A/B/C とも exit 0。ログ上 COMET ckpt は
  `OUTPUT_DIR/cache` symlink 経由でロード、nltk punkt_tab は
  「already up-to-date」、`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` 下で
  Hub アクセス・接続エラーはゼロ。COMET / BERTScore も算出
  (B: 163 スコア、`{alt,wikicorpus}-{e-to-j,j-to-e}_comet_wmt22`・
  `xlsum_ja_bert_score_ja_f1` など)。結果は `OUTPUT_DIR/results/result.json`
- **発見 (重要): 共有 install への書き込みはゼロになっていない** (原因特定と修正は
  次節。以下は発見時点の記録)。
  eval 時 dump が `output_dir/datasets/<ver>/evaluation/test/prompts_<hash>/` に
  書くが、`OUTPUT_DIR/datasets` は共有 install への symlink なので書き込みが
  そのまま貫通する。run B で共有 install 配下に 61 ファイル
  (`data/llm-jp-eval/datasets/2.1.5/evaluation/test/prompts_purSfVDUEf5jut0K2Qzx-A==/`)
  が新規作成された。当該ディレクトリは `drwxr-s---` でグループ書き込み不可
  なので、**インストールした本人以外は従来どおり EACCES で失敗し得る**
  (bc6b95a が解消しようとした問題そのもの)。run A で書き込みが出なかったのは
  同一 prompt ハッシュのディレクトリが 07-31 の実行で既に存在したためで、
  ハッシュが変わる (バージョン・config・データセット構成が異なる) 実行では必ず発生する。
  さくらの 07-29 検証は `results/`・`cache/`・`hf/`・`nltk/` の mtime のみ確認して
  おり `datasets/` を見ていなかったため取りこぼした
- **スコア再現性**: v2.1.5 の 8b-base --basemodel AVG は
  07-31 = 0.58310 / 08-05 run A = 0.58487 / 08-05 run C = 0.58345。
  A と C は**同一環境・同一設定・同日**でも 77 スコア中 19 個が異なり
  AVG が 0.0014 動く (`gsm8k`/`mawps`/`mgsm` などで 100 サンプル中 1 件の反転)。
  よって 07-31→08-05 の +0.0018 は再インストールによる退行ではなく、
  このパイプライン固有の run-to-run 変動 (vLLM の prefix caching / バッチ依存の
  数値差、`seed=None`) の範囲内。**AVG の比較は ±0.002 程度を同値とみなすこと**
- 補足: ABCI 計算ノードは dify-sandbox イメージを pull できないため
  run B は `DISABLE_CODE_EXEC=1` (mbpp / jhumaneval と CG カテゴリを除外)

## 2026-08-05 ABCI: eval 時 dump による共有 install 書き込みの解消

前節で見つかった書き込みの原因を llm-jp-eval 本体まで追って修正した。

**原因**: `evaluate()` は 1 行目で dump サブコマンドと同じ
`load_dataset_and_construct_prompt_template(cfg)` を呼ぶ (`evaluator.py:281`)。
つまり **eval は毎回フルのプロンプト dump を実行する** (旧コメントの
「一部のデータセットがなぜか eval 時に dump を実行する」は誤り)。dump 先は
`EvaluationConfig.inference_input_path` で決まり (`schemas.py:200-202`)、

```python
if self.inference_input_dir is not None:
    return Path(self.inference_input_dir + f"_{hash_str}")
return Path(self.target_dataset_dir / f"prompts_{hash_str}")   # ← フォールバック
```

`target_dataset_dir` は `output_dir/datasets/<ver>/evaluation/<split>`
(`schemas.py:184`)。run スクリプトは dump フェーズにだけ
`--inference_input_dir` を渡し eval フェーズには渡していなかったため、
eval の dump 先が `datasets` symlink 経由で共有 install に落ちていた。
さらにハッシュ種には `max_num_samples` が含まれる (`schemas.py:190-198`) が
EVAL_OPTS はこれも渡しておらず、config の既定値 (100) が使われていた。
`--max_num_samples 100` の run では偶然ハッシュが一致して既存ディレクトリを
再利用し書き込みが出ず、`2` を指定した run では不一致で新規作成された、という
挙動の説明もこれで付く。

**修正**: EVAL_OPTS に `--inference_input_dir=${PROMPT_OUTPUT_DIR}` と
`--max_num_samples=${MAX_NUM_SAMPLES}` を追加 (dump フェーズと同一の値)。
これで eval は dump 済みの `*.eval-prompt.json` を見つけて
`prompt_dump_path.exists()` 分岐で再生成をスキップする (`evaluator.py:47-51`)。
v2.1.0 / v2.1.3 / v2.1.5 の `run_llm-jp-eval.sh` と
`vllm-serve/scripts/run_llm-jp-eval-serve.sh` の 4 本に適用。

**検証1: eval のみの A/B (`qsub-evalonly.sh`)**。前節 run B の推論結果を流用して
eval だけを旧 EVAL_OPTS / 新 EVAL_OPTS で実行。生成を伴わないので
`result.json` を厳密比較できる (生成を含めると後述の run-to-run 変動が乗る)。
各変種の実行前に共有側の `prompts_purSfVDUEf5jut0K2Qzx-A==` を削除して前提を揃えた。

- **共有 install への書き込み: 旧 63 件 → 新 0 件**
- **スコアは完全一致**: `scores` 163 個・`lang_scores` とも一致、`records` も一致。
  差分は `time_profile` (実測時間) と `export_timestamp`、および意図した
  `config` の項目 (`inference_input_dir`, `inference_input_path`,
  `max_num_samples` 100→2) のみ。**`--max_num_samples` を eval に渡しても
  スコアは変わらない**ことの実測確認になる (オフライン評価のサンプルは推論結果
  ファイル `target_data["samples"]` 由来のため; `evaluator.py:341-343`)

**検証2: 再インストール + e2e (`qsub-e2e.sh`)**。修正版スクリプトを
canonical 環境に再インストールし (env 内スクリプトはリポジトリと一致確認)、
pre-fix の遺物 `prompts_kkD0PNH5iF_gPb3O0tj5ow==` も共有側から削除した上で、
dump→推論→eval のフルパイプラインを 2 構成で実行:

```bash
bash run_llm-jp-eval.sh llm-jp/llm-jp-3-150m     $OUT/e2e-full-150m --max_num_samples 2
bash run_llm-jp-eval.sh llm-jp/llm-jp-4-8b-base  $OUT/e2e-basemodel-8b-base --max_num_samples 100 --basemodel
```

- 両方 exit 0、**共有 install への書き込み 0 件**
- **eval が dump を再利用していることをログで確認**: dump フェーズで
  `eval-prompt.json generated` が 61 件、**eval フェーズでは 0 件**。
  eval の `inference_input_path` は `OUTPUT_DIR/prompts_0x-o3IlYSCB5GqzucObd_g==`
  (従来は共有側の `datasets/.../prompts_<別ハッシュ>`)
- 8b-base --basemodel の AVG は 0.58230。同条件 4 回の実測は
  0.58230 / 0.58310 / 0.58345 / 0.58487 で**幅 0.00257**。修正はこの範囲内で、
  かつ検証1 でスコア同一性は厳密に示せているため退行なし
- 注: 150m は `--max_num_samples 2` (1 データセット 2 サンプル) なので 1 件の
  反転が 50% 動く。生成を含む比較の分解能は低いので、スコア同一性の確認は
  検証1 の eval-only A/B で行うのが正しい

**未実施**: serve 経路 (`run_llm-jp-eval-serve.sh`) は同じ EVAL_OPTS 構造なので
同じ修正を入れたが、ABCI では未デプロイ・未実行 (ABCI の
`environment/vllm-serve` は 07-29 時点の pre-fix 版のままで自己整合している)。
v2.1.0 / v2.1.3 も同様にコード修正のみ。

## 2026-08-07 ABCI: serve 経路の検証 + vllm-serve 再デプロイ

08-05 の修正は serve の `run_llm-jp-eval-serve.sh` にも入れたが、どのクラスタでも
serve 経路は未実行だったため ABCI で検証した。ABCI の `environment/vllm-serve` は
07-29 時点の pre-bc6b95a 版だったので、**先にリポジトリのスクリプトを
`--experiment-dir` 指定で直接実行して検証し、通ってからデプロイ**した
(共有環境を壊さない順序)。

```bash
# 検証 (job 2116105)。results/serve-outputdir-fix-20260807/qsub.sh
bash <repo>/vllm-serve/scripts/run_eval_serve.sh llm-jp/llm-jp-3-150m $OUT \
  --experiment-dir /groups/gcg51557/experiments/0230_intg_eval_2509 \
  --serve-venv <v2.1.5 の vllm venv> --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 --llm-jp-eval-versions v2.1.5 --max-num-samples 2

# デプロイ
bash <repo>/vllm-serve/install.sh /groups/gcg51557/experiments/0230_intg_eval_2509/environment
```

- serve 完走 (exit 0)、**共有 install への書き込み 0 件**
- **eval は dump を再利用**: dump フェーズ 61 件生成 / eval フェーズ 0 件。
  `inference_input_path` は `OUTPUT_DIR/llm-jp-eval/v2.1.5/prompts_0x-o3IlY...`
- 結果は `OUTPUT_DIR/llm-jp-eval/v2.1.5/results/result.json` に 163 スコア、
  COMET / BERTScore も算出 (AVG 0.1192)
- デプロイ後、`environment/vllm-serve/` の 8 ファイルがリポジトリと一致することを確認

**重要な制約 (この検証で判明)**: `run_llm-jp-eval-serve.sh` はバージョン非依存で、
`HF_HOME` / `NLTK_DATA` を渡された `VERSION_ENV_DIR` から導出し
`HF_HUB_OFFLINE=1` で eval する。したがって **新しい serve スクリプトで駆動できる
のは prefetch 済み (= 07-29 以降のインストーラで再インストールした) バージョンだけ**。
ABCI 現況では v2.1.5 のみ。v2.1.0 / v2.1.3 を serve 経由で回すと eval フェーズが
COMET / BERTScore のロードで失敗する (offline 経路は各バージョンの env 内スクリプトを
使うので影響なし)。v1.4.1 は別スクリプト
(`run_llm-jp-eval-v1-serve.sh`、COMET/BERTScore を使わない) なので無関係。


## 2026-08-12 ABCI: qsub.py 配備更新 + swallow-tf5 の GPU (vllm バックエンド) 初検証

`--swallow-version v202411-tf5` を ABCI の qsub.py から使えるようにする作業。
tf5 環境自体は 07-31 に導入済みだったが、(1) 配備側
`environment/scripts/qsub.py` が旧版 (tf5 choices なし・environment3 参照) の
ままで、(2) README (tf5) が明記していた通り **GPU (vllm バックエンド) は未検証**
だった。

- **配備更新**: `environment/scripts/qsub.py` をリポジトリ版に同期 (tf5 choices・
  vllm-serve・llm-jp-judge 対応が入った現行版)。`qsub_nonbreaking.py` も予約
  キュー名のみ旧世代 (R9920251000) だったため同期した
- **1 回目の検証ジョブ (2129450, llm-jp-3-150m, EN ハーネス) は全タスク即死**
  (result.json 全列 -1.0)。原因は `vllm_causallms-vllm010-compat.patch` が
  vllm 0.19 で残していた import 非互換 2 件:
  1. **`import ray` が死にコード**: パッチで ray ベース DP は multiprocessing に
     置換済みだが import ガード先頭の `import ray` が残存。vllm 0.10 は ray を
     必須依存で連れてくるため base 環境では顕在化しなかったが、vllm 0.19 は
     ray 依存を持たないため tf5 venv では ModuleNotFoundError → ガードが
     握りつぶして実行時 `NameError: name 'LLM' is not defined` になる。
     07-20 さくらの CPU スモークは hf バックエンドのため素通りしていた
     (「import 互換性確認済み」も例外が握りつぶされるため検出できていなかった)
  2. **`vllm.utils.get_open_port` の移動**: vllm 0.19 では
     `vllm.utils.network_utils` に移動。1 を直すと今度はこれが ImportError で
     顕在化する (ModuleNotFoundError でないためガードを突き抜けてクラッシュ)
- **修正**: 共有パッチから `import ray` を削除し、`get_open_port` は旧パス →
  ImportError なら新パスのフォールバック import に変更 (vllm 0.10 / 0.19 両対応。
  base 環境のインストール済みファイルは旧世代パッチ由来のため触っていない)。
  配備済み tf5 の `vllm_causallms.py` には pristine + 新パッチの結果を反映し、
  venv-harness で import ブロックの解決を確認してから再投入
- **2 回目 (2129477) は完走** (exit 0、走行 68 分、Traceback 0 件):

  ```bash
  python3 qsub.py llm-jp/llm-jp-3-150m $RESULTS/swallow-tf5-150m-20260812b \
    --swallow-version v202411-tf5 --disable-llm-jp-eval \
    --pbs-queue rt_HG --rtype rt_HG
  ```

  EN 15 列すべて取得: mmlu 0.260 / hellaswag 0.289 / xwinograd_en 0.602 /
  bbh_cot 0.108 / gsm8k 0.0 / math_500 0.006 / gpqa 0.0 など、150m として
  妥当な値 (mmlu はチャンスレベル)。JA 列が -1.0 なのは tf5 変種の仕様
  (EN ハーネスのみ実行)。DP > 1 の multiprocessing 経路は今回未検証 (DP=1)

## 2026-08-12 ABCI: swallow-tf5 の data parallel (mp 経路) 初検証

同日の DP=1 検証に続き、パッチが ray から置換した multiprocessing data parallel
経路の初の実機検証。rt_HF (H200 ×8)、llm-jp-3-150m、TP=1 / DP=8。

```bash
python3 qsub.py llm-jp/llm-jp-3-150m $RESULTS/swallow-tf5-150m-dp8-20260812c \
  --swallow-version v202411-tf5 --disable-llm-jp-eval --data-parallel-size 8 \
  --pbs-queue rt_HF --rtype rt_HF --select 1
```

3 ジョブを要した (障害 2 件を発見・修正):

1. **vllm 0.19 は dense モデルのオフライン DP モードを拒否** (1 回目 2129845):
   mp ワーカーは vllm の `VLLM_DP_*` 環境変数方式 (公式 data_parallel.py 例由来)
   を使っていたが、0.19 は ParallelConfig の検証で
   `Offline data parallel mode is not supported/useful for dense models` を
   投げる (この協調は MoE の expert parallel 同期用)。→ **各ワーカーが自分の
   ランクに対応する `CUDA_VISIBLE_DEVICES` の TP サイズ分スライスを確保して
   独立エンジンを立てる方式に書き換え** (upstream の旧 ray 実装と同じ考え方。
   dense / MoE を問わず動き、`get_open_port` も不要になったため 08-12 に
   入れたフォールバック import ごと削除)
2. **エンジン再起動時の GPU メモリ解放待ち競合** (2 回目 2129872):
   lm_eval はリクエストグループごとにエンジンを作り直すが、直前ラウンドの
   エンジンプロセスの GPU メモリ解放が完了する前に次の初期化が走ると、
   gpu_memory_utilization=0.9 分の空き確保に失敗して WorkerProc init が死ぬ
   (最初のタスクは 3 ラウンド成功後の 4 ラウンド目で失敗 → 残骸で後続も連鎖)。
   → **ワーカーのエンジン初期化に 15 秒間隔 ×3 のリトライを追加**
3. **3 回目 (2129953) は完走** (exit 0、44 分 vs DP=1 の 68 分)。リトライは
   実際に 1 回発火して初期化競合を吸収 (ハード失敗 0 件)。**EN 全 15 列で
   DP=1 と一致**: loglikelihood 系は完全一致が多数、生成系含め最大 |diff|
   0.001 (mmlu_stem +0.0010)

## 2026-08-28 ABCI: safety-eval 導入 + e2e 検証

安全性評価コンポーネント (b18a64e で追加) の初の実機 e2e。環境は 0230
(`environment/safety-eval` に install.sh でインストール、venv は vllm 0.11.2 /
transformers 4.57.6 / Python 3.10、JTruthfulQA 分類器プリフェッチ済み)。
ターゲットは llm-jp/llm-jp-3-150m、全 5 ベンチマーク × 先頭 5 サンプル
(`--safety-eval-benchmark-size 5`、ask_times=3 → 各 15 生成)、ジャッジは
ABCI 内部 OpenAI 互換サーバーの gemma-4-31B-it。

```bash
python3 qsub.py llm-jp/llm-jp-3-150m $RESULTS/safety-eval-150m-20260828 \
  --experiment-dir /groups/gcg51557/experiments/0230_intg_eval_2509 \
  --disable-swallow --disable-llm-jp-eval \
  --safety-eval --safety-eval-judge-model gemma-4-31B-it \
  --safety-eval-judge-max-tokens 2048 --safety-eval-benchmark-size 5 \
  --pbs-queue rt_HG --rtype rt_HG
```

- **1 回目 (2181773)**: 生成 (vLLM オフライン、VLLM_USE_V1=0 設定込みの受領
  コードのまま) と jbbq / judge 系 3 ベンチマークの評価は完走したが、障害 2 件:
  1. **JTruthfulQA が ImportError (rhoknp)**: 分類器
     `nlp-waseda/roberta_jtruthfulqa` のトークナイザは
     `word_tokenizer_type: jumanpp` で、rhoknp + jumanpp バイナリが必須。
     安全性WGの動作確認 freeze (`llm_safety_latest_working.txt`) にも
     rhoknp 系は無く、この経路は先方環境でも OS 側依存だった模様。
     → インストーラに rhoknp (venv) と Juman++ v2.0.0-rc4 の環境内ソース
     ビルドを追加し、プリフェッチ時に分類器ロード + 1 件分類のスモークを実施
  2. **V1 ジャッジのスコアが全件 None** (集計は欠損補完の 3.0 になり一見
     成立するので注意): このサーバーの gemma-4-31B-it は reasoning に
     ~1900 トークン使うため、受領コード固定の max_tokens=512 では本文が空
     (finish_reason=length)。max_tokens=2048 のプローブでは
     「評価：[[3]]」まで完走、非 thinking の llm-jp-4-8b-instruct は 512 で
     完走することを確認。→ `--safety-eval-judge-max-tokens` /
     `--judge-max-tokens` (env `SAFETY_EVAL_JUDGE_MAX_TOKENS`) を追加
  - あわせて **長大生成での Juman++ 失敗** も発見: ベースモデルの暴走出力
    (~4096 トークン) への Juman++ 前処理が「returned empty result」で失敗し
    11/15 件が invalid になる。分類器入力を先頭 1000 文字に制限 (トークナイザ
    は 128 トークンで切るためスコア不変) して解消
- **2 回目 (2181850, 修正込みの再投入)**: judge 系は正常化 (V1 44/45 件
  パース、スコア 1〜3 に分布) したが **JTruthfulQA が「Juman++ exited
  unexpectedly」で全滅**。原因は検証作業側のミス: jumanpp の辞書パスは
  CMAKE_INSTALL_PREFIX からバイナリに焼き込まれるため、ログインノードの
  /tmp でビルドしたバイナリのコピーは計算ノードで辞書を解決できない
  (ログインノードでは旧ディレクトリが残存するため気づけない)。プレフィックス
  変更後の cmake 再構成でも焼き込みが更新されないことを確認し、
  **インストーラはクリーンな build ディレクトリで最終プレフィックスを指定して
  ビルドする**よう堅牢化 (インストーラ自体は元々正しい手順だった)
- **3 回目 (2181912) は完走** (exit 0、生成は 1 回目の出力を再利用、走行 7 分):
  全 5 ベンチマークの集計を取得し、全サンプルが valid。
  jbbq_age acc 0.583 (invalid 3 件は 150m の出力に 0/1/2 が無いもので正常動作)、
  jtruthfulqa 15/15 valid (truthful_rate 0.0 は 150m のゴミ出力に対して妥当)、
  judge 系 3 ベンチマークとも 15 件採点 (attempt_avg: AC 2.13 / JSF 2.67 /
  SB 0.0)。**ジョブ実行時間帯の共有 install への書き込みは 0 件**
  (`find -newermt` で確認; 検出されたのは投入前の手動デプロイのみ)
- 運用ノート: 内部サーバーのジャッジモデル選定は
  gemma-4-31B-it (`--safety-eval-judge-max-tokens 2048` 必須) または
  llm-jp-4-8b-instruct (512 で可)。gpt-4o は内部サーバーに無い
