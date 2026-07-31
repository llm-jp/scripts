# 検証記録 (intg-eval-2508)

本ディレクトリのインストーラー / 実行スクリプトに対する実クラスタでの検証実験の記録。
実験は複数クラスタで行うため、各ラウンドに **どのクラスタ・どの環境で・どのコマンドを
実行したか** を必ず残す。新しい検証を行ったら本ファイルに追記すること。

対象クラスタと環境:

| 呼称 | クラスタ | 環境 (インストール先) | ジョブ投入 |
|---|---|---|---|
| さくら | さくらインターネット (Slurm, B200×8/ノード, sm_100, コンテナランタイムなし) | `/data/experiments/0219_dev_eval_script/environment` | `scripts/sbatch.py` |
| ABCI | ABCI (PBS, H100) | `/groups/gcg51557/experiments/0230_intg_eval_2509/environment` | `scripts/qsub.py` |

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
| 07-29 | ABCI | llm-jp-judge 導入 + offline/serve テスト (8b-thinking) | serve 完走・スコア取得。thinking モデルの空応答問題を発見 → max-tokens/reasoning-parser 制御を追加 |
| 07-29 | ABCI | swallow プリフェッチ適用 + オフライン読込確認 | 全10タスク取得 (545MB)、HF_HUB_OFFLINE=1 でも読込 OK |
| 07-31 | ABCI | v2.1.5 / swallow-tf5 導入 + --basemodel 実機テスト | basemodel AVG 0.583 (さくらと一致)。計算ノードは外部ネットワーク不可と判明 |
| 07-31 | ABCI | thinking モデル × llm-jp-judge の完動設定確立 | 障害3件を切り分け。effort 明示 + final 抽出で offline/serve ともスコア成立・整合 |

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

H100 1 枚、llm-jp/llm-jp-3-150m、swallow_v202411 + llm-jp-eval v2.1.3、
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

## 2026-07-29 ABCI: llm-jp-judge 導入 + offline/serve 両モードテスト

llm-jp-judge v2.0.0 サブインストーラを `environment/llm-jp-judge` に導入
(AnswerCarefully は gated 未承認のためスキップ → quality_ja / culture_ja /
safety_boundary_ja / MT-Bench ja+en の 5 ベンチマークで実施)。
対象 llm-jp/llm-jp-4-8b-thinking、ジャッジは OpenAI 互換 API 経由の
llm-jp-4-32b-a3b-thinking、各ベンチマーク 10 件、rt_HG (H100 1 枚)。

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
- **--basemodel 実機テスト** (llm-jp-4-8b-base, v2.1.5, offline, H100 1枚):

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
