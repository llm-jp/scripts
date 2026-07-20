# ABCI での vllm-serve モード GPU 検証 — 作業引き継ぎ

さくらインターネットのクラスタ (Slurm, B200) の GPU キューが混雑しているため、
GPU を使う検証を ABCI (PBS) 側で行う。この文書は ABCI 側の Claude Code
セッションへの引き継ぎ資料である。

## ミッション

vllm-serve モード (このディレクトリ; 設計と経緯は README.md 参照) の
**実 GPU での end-to-end 検証**と**オフライン実行とのスコア一致性確認**。

CPU + mock サーバーでのプラミング検証は完了済み。未検証なのは実 vLLM サーバー
との結合部分:

1. vLLM の OpenAI サーバーが `echo=True + max_tokens=0 + logprobs` で prompt
   logprobs を返すこと (loglikelihood 経路)
2. トークン ID 配列プロンプト (`prompt: [[int, ...], ...]`) の受理
3. `truncate_prompt_tokens` (extra_body) の受理
4. スコアがオフライン実行と一致すること

## ABCI 側の前提 (要確認 — この通りでなければ調査から)

- 評価環境: `/groups/gcg51557/experiments/0230_intg_eval_2509/environment/`
  に swallow_v202411 / llm-jp-eval-v1.4.1 / v2.1.0 / v2.1.3 がインストール済み
  のはず (`scripts/qsub.py` のデフォルトから推定)
- サーバー用 venv 候補: v2.1.3 の
  `environment/src/llm-jp-eval/llm-jp-eval-inference/inference-modules/vllm/.venv`
  (vllm 0.11.2)。`bin/vllm` の存在を確認すること。
  ※ さくら側の auto-detect 順 (run_eval_serve.sh) は v2.1.5 → swallow tf5 →
  swallow v202411 だが、ABCI には v2.1.5 / tf5 は無い想定。`--serve-venv` で
  明示指定するのが確実
- ジョブ投入: PBS。`qsub -P gcg51557 -q R9920261000 -v RTYPE=rt_HG -l select=1`
  (rt_HG = H100 1枚)。インタラクティブは `-I`
- `HF_HOME` は `/groups/gcg51557/experiments/<experiment_dir>/.cache/huggingface`
  配下必須、`HF_TOKEN` 必要
- ABCI には singularity があるため llm-jp-eval のコード実行系 (mbpp/jhumaneval)
  も動く (さくらと違い自動スキップされない)

## 手順案

```bash
# 0. リポジトリ取得 (このブランチ)
git clone https://github.com/llm-jp/scripts.git -b feat/intg-eval-installer
cd scripts/evaluation/installers/intg-eval-2508/vllm-serve

# 1. インストール (既存環境への追加コピー + swallow への serve-compat パッチ。
#    既存のオフライン評価挙動は変えない。install.sh と README.md を先に読むこと)
ENV_DIR=/groups/gcg51557/experiments/0230_intg_eval_2509/environment
bash install.sh $ENV_DIR
# ※ 書き込み権限がない場合は自分の実験ディレクトリに環境一式をインストール
#   してから行う (../install.sh 等を参照)

# 2. GPU ジョブ内でスモークテスト (llm-jp-3-150m)
#    swallow + llm-jp-eval v2.1.3 をサーバー1本で実行
bash $ENV_DIR/vllm-serve/run_eval_serve.sh \
  llm-jp/llm-jp-3-150m \
  <output_dir> \
  --serve-venv $ENV_DIR/llm-jp-eval-v2.1.3/environment/src/llm-jp-eval/llm-jp-eval-inference/inference-modules/vllm/.venv \
  --swallow --swallow-env swallow_v202411 \
  --llm-jp-eval-versions v2.1.3

# 3. スコア一致性: 同一モデル・同一バージョンのオフライン実行
#    (qsub.py --llm-jp-eval-versions v2.1.3 / swallow) と result.json を比較
#    - swallow: <out>/swallow/results/result.json
#    - llm-jp-eval: <out>/llm-jp-eval/v2.1.3/results/result.json
#    loglikelihood 系タスクは一致、生成系 (greedy) はほぼ一致が期待値。
#    llm-jp-eval は temperature=1.0 サンプリングのためサンプリングノイズ分の
#    差は許容 (seed 固定でどこまで揃うかは vLLM のバッチ非決定性次第)

# 4. (できれば) 巨大モデルで所要時間比較: 従来 qsub.py 実行 vs serve モード
```

## 既知の注意点 (詳細は README.md)

- クライアントの `model=` はサーバーの served model name と文字列一致が必要
- `OPENAI_API_KEY=EMPTY` を export (スクリプト内で設定済みだが単体実行時は注意)
- サーバーは `--max-logprobs 20` で起動している (serve_common.sh)
- llm-jp-eval の `--reasoning_parser` は serve モード未対応
- スコアはサーバー venv の vLLM バージョンに紐づく (0.11.2 なら v2.1.3
  オフラインと比較可能)
- vllm 0.11.2 で `truncate_prompt_tokens` / トークン配列 prompt が期待通りか
  要確認。問題があれば inference_openai.py / serve_common.sh を調整し、
  このブランチにコミットして戻すこと

## さくら側の状況 (参考)

- さくら側では GPU スモークジョブ 2013 (llm-jp-eval v2.1.5 e2e) / 2014
  (swallow transformers-v5 変種) がキュー待ちのまま
- vllm-serve モードの CPU (mock) 検証はさくら側 login ノードで完了済み
- llm-jp-eval-inference の transformers モジュールに upstream バグ 2 件を発見
  済み (accelerate の lock 不整合 / pipeline への generation_config 渡しが
  transformers 5.x で TypeError)。vllm-serve とは独立の話だが、upstream への
  Issue 報告候補
