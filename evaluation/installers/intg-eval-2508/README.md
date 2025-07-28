# 統合評価基盤インストーラ (v2508)

llm-jp-evalとswallow英語評価を実行可能な統合基盤をインストールするスクリプトです。

## セットアップ

### インストール手順 (ローカル)

```bash
INSTALL_DIR="Path/to/install_dir" # FIX_ME

# インストーラーを取得
git clone https://github.com/llm-jp/scripts.git -b feat/intg-eval-installer

cd scripts/evaluation/installers/intg-eval-2508

bash install.sh $INSTALL_DIR > logs/install-eval.out 2> logs/install-eval.err
```

### インストール手順 (on ABCI)

```bash
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

> [!NOTE]
> HF_HOMEを指定していない場合は動きません。
> また、`/groups/gcg51557/experiments/<experiment_dir>`以下で指定する必要があります。
> `<experiment_dir>`は適宜、実験ディレクトリ名で置き換えてください。

```
export HF_HOME=/groups/gcg51557/experiments/<experiment_dir>/.cache/huggingface
```

### HF_TOKENの設定

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

### ジョブ形式での実行 (on ABCI)

```
python3 $INSTALL_DIR/scripts/qsub.py \
  <model_name_or_absolute_path> \
  <output_dir_absolute_path> \
  [--swallow_version v202411] \
  [--llm_jp_eval_version v1.4.1] \
  [--job_name 0182_intg_eval] \
  [--rtype rt_HG] \
  [--select 1] \
  [--options ""]
```

必須引数:
- `<model_name_or_absolute_path>` モデル名またはモデルの絶対パス(HFのモデル名を指定する場合は[モデル名を指定する場合](##モデル名を指定する場合)を参照)
- `<output_dir_absolute_path>` 結果出力先の絶対パス

オプション引数:
- `--swallow_version` swallowバージョン（デフォルト: `v202411`）現状は`v202411`のみに対応。
- `--llm_jp_eval_version` llm-jp-evalバージョン（デフォルト: `v1.4.1`）現状は`v1.4.1`のみに対応。
- `--job_name` ジョブ名（デフォルト: `0182_intg_eval`）
- `--rtype` リソースタイプ（デフォルト: `rt_HG`）
- `--select` GPU数またはノード数（デフォルト: 1）
- `--options` qsubスクリプトへの追加オプション
