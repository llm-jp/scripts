# llm-jp-judge インストーラ

[llm-jp-judge](https://github.com/llm-jp/llm-jp-judge) (v2.0.0) による LLM-as-a-Judge 評価環境をインストールするスクリプトです。
生成 (ターゲットモデルをローカル vLLM サーバで推論) と評価 (ジャッジモデルによる採点) の2段階で実行します。

## セットアップ

```bash
# インストール先を指定 (通常は統合評価基盤の install.sh から呼ばれる)
INSTALL_DIR="Path/to/install_dir" # FIX_ME

cd scripts/evaluation/installers/intg-eval-2508/llm-jp-judge

bash install.sh $INSTALL_DIR > logs/install-llm-jp-judge.out 2> logs/install-llm-jp-judge.err
```

> [!NOTE]
> - `uv` がPATH上にない場合、スタンドアロンの `uv` を環境ディレクトリ配下にインストールして使用します。
> - AnswerCarefully (安全性評価用データセット) はgatedデータセットのため、Hugging Face上での利用申請の承認と `HF_TOKEN` が必要です。取得できない場合は警告を出してスキップし、実行時に `safety_ja` / `safety_borderline_ja` ベンチマークが自動的にスキップされます。

## 評価実行

```bash
bash $INSTALL_DIR/run_llm-jp-judge.sh \
  <model_name_or_absolute_path> \
  <output_dir> \
  [--judge-client {openai,azure,bedrock,vllm}] \
  [--judge-model gpt-4o-2024-08-06] \
  [--judge-base-url URL] \
  [--tensor-parallel-size N] \
  [--benchmark-size N] \
  [--disable-mt-bench]
```

- 生成フェーズはターゲットモデルを vLLM サーバとして起動し、OpenAI互換クライアント経由で全ベンチマーク (quality_ja / safety_ja / culture_ja / safety_borderline_ja / safety_boundary_ja / mt_bench_en / mt_bench_ja) の応答を生成します。
- 評価フェーズはデフォルトで OpenAI API (`gpt-4o-2024-08-06`) をジャッジとして使用します。APIキーは環境変数 (`OPENAI_API_KEY` など) か、llm-jp-judgeチェックアウト直下の `.env` から読み込まれます。
- 計算ノードから外部APIへ疎通できない環境では `--judge-client vllm --judge-model <ジャッジ用モデル>` を指定すると、生成後にジャッジモデルをローカル vLLM サーバで起動して評価します (公表スコアとの比較可能性は下がる点に注意)。
- 結果は `<output_dir>/evaluation/score_table.json` に出力されます。
- `--gen-base-url <URL>` を指定すると、起動済みのOpenAI互換サーバ (例: vllm-serve の共有サーバ) を生成に利用します (ローカルでのサーバ起動を省略)。`--generation-only` / `--judge-only` でフェーズを分割実行できます (vllm-serve 統合はこの組み合わせで、生成を共有サーバで行い、サーバ停止後にジャッジを実行します)。

> [!NOTE]
> ジャッジをAPIで行う場合、計算ノードからAPIエンドポイントへのネットワーク疎通が必要です。
> 疎通できないクラスタでは `--judge-client vllm` を使用してください。
