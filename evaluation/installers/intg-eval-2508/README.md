# 統合評価基盤インストーラ (v2508)

llm-jp-evalとswallow英語評価を実行可能な統合基盤をインストールするスクリプトです。

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
  [--llm-jp-eval-versions v1.4.1 v2.1.3] \
  [--partition gpu] \
  [--gpus 1] \
  [--dry-run]
```

- `--experiment-dir` は評価環境をインストールした実験ディレクトリ (`environment/` の親) を指定します。環境変数 `INTG_EVAL_EXPERIMENT_DIR` でも指定可能です。
- その他のオプションは `qsub.py` と共通です (`--help` を参照)。

> [!NOTE]
> - singularity等のコンテナランタイムがないノードでは、llm-jp-eval v2系のコード実行系データセット (`mbpp`, `jhumaneval`) とCGカテゴリは自動的にスキップされます (`DISABLE_CODE_EXEC=1` で明示的な無効化も可能)。そのためAVGスコアはコード実行を含む環境での結果と直接比較できません。なおllm-jp-eval v1.4.1のmbppはインプロセスの`exec()`で評価されるため、コンテナランタイムなしでも実行されます。
> - llm-jp-eval v2.1.0はvllm 0.9.0.1 / torch 2.7.0 (cu126) に依存しており、Blackwell世代 (B200等, sm_100) のGPUでは動作しない可能性があります。その場合はvllm 0.11.2 / torch 2.9.0 (cu128) を使用するv2.1.3を利用してください。

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
