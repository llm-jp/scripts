# swallow v202411 — transformers v5 variant (EXPERIMENTAL)

`swallow_v202411` と同一の評価コード・実行スクリプト・パッチを使いつつ、
harness venv のみ transformers 5.x 系に更新した変種インストーラです。
transformers>=5.6 を要求するモデル (例: Gemma 4 系の tokenizer 設定) を
swallow 英語評価にかけるための環境です。

オリジナルの `swallow_v202411` には一切手を加えていません。両者は別ディレクトリに
インストールされ、共存できます。

## オリジナルとの差分 (harness venv のみ)

| パッケージ | swallow_v202411 | 本変種 |
|---|---|---|
| Python | 3.10.14 | 3.11 (vllm 0.19 の要求) |
| transformers | 4.56.2 | 5.6.2 |
| vllm | 0.10.2 | 0.19.1 |
| torch | 2.8.0 (cu121) | 2.10.0 (cu128, vllm の解決に委任) |
| huggingface-hub | 0.x | 1.x (transformers 5.6 の要求) |

実行スクリプト (`run-eval.sh`)・補助スクリプト・
`vllm_causallms-vllm010-compat.patch` は `../swallow_v202411` から
そのままコピーして使用します (単一ソース維持のため本ディレクトリには置きません)。

## 検証状況 (2026-07-20)

- CPU (hf バックエンド, `--limit` 付き) で完走確認済み:
  hellaswag / gsm8k / openbookqa / xwinograd_en / squadv2 / triviaqa
- `lm_eval.models.vllm_causallms` (パッチ適用済み) の vllm 0.19.1 に対する
  import 互換性確認済み
- **GPU (vllm バックエンド) は未検証** — 特にパッチの data parallel (mp) 経路と
  TokensPrompt API は vllm 0.10 向けに書かれており、0.19 での動作確認が必要
- 既知のメタデータ衝突: xgrammar 0.2.4 が transformers<5 を要求 (`pip check` で
  警告)。xgrammar は guided decoding 用で swallow 評価では使用されないため実行時
  影響はない見込み

## インストール

```bash
cd scripts/evaluation/installers/intg-eval-2508/swallow_v202411-tf5
bash install.sh <TARGET_DIR> > logs/install.out 2> logs/install.err
# 例: TARGET_DIR = <experiment_dir>/environment/swallow_v202411-tf5
```

統合インストーラ (`../install.sh`) には含めていません (GPU 検証完了までオプトイン)。
