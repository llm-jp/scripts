# v4-moe convert

`v4-moe-upstream-megatron-abci` においてインストールした環境を使用し、
Megatron-LM の checkpoint を Hugging Face 形式へ変換します。

以下のコマンドで、Hugging Face 形式の checkpoint を事前に初期化します。

```bash
python hf_init.py \
  --config v4-32b-a3.8b \
  --out /path/to/output/hf_ref_weight
```

`/path/to/output/hf_ref_weight` に、v4 用の tokenizer 関連ファイルをコピーしてください。

## 例

事前に `MODELSCOPE_CACHE` ディレクトリを作成しておいてください。

`v4-moe-32b-a3.8b` は 1 GPU で変換可能です。

```bash
qsub -v \
RTYPE=rt_HG,\
VENV_DIR=/path/to/venv,\
MEGATRON_CKPT_STEP=<checkpoint_step>,\
MODELSCOPE_CACHE=/path/to/MODELSCOPE_CACHE,\
MEGATRON_LM_PATH=/path/to/Megatron-LM,\
MCORE_PATH=/path/to/megatron/checkpoints,\
HF_REF_WEIGHT=/path/to/output/hf_ref_weight,\
HF_OUTPUT_DIR=/path/to/hf_output_dir \
qsub_convert.sh
```

### 補足

- `VENV_DIR` 使用する Python 仮想環境のディレクトリを指定します。 例： `/groups/gcg51557/experiments/0279_v4-moe-32b-a3.8b/env/venv`

- `MEGATRON_CKPT_STEP` 変換したい Megatron checkpoint の iteration を指定します。例：`1000`

- `MCORE_PATH` Megatron 形式の checkpoint が配置されているディレクトリを指定します。 例： `/groups/gcg51557/experiments/0279_v4-moe-32b-a3.8b/tasks/base/checkpoints`

- `MEGATRON_LM_PATH` 使用する Megatron-LM リポジトリのパスを指定します。例： `/groups/gcg51557/experiments/0279_v4-moe-32b-a3.8b/env/src/Megatron-LM`

- `HF_OUTPUT_DIR`
  Hugging Face 形式の checkpoint の出力先ディレクトリを指定します。このディレクトリ配下に、iter_XXXXXXX（ゼロ埋めされた checkpoint step）形式のサブディレクトリが自動的に作成されます。 padding について
  MEGATRON_CKPT_STEP のゼロ埋め（padding）は、スクリプト内部で自動的に行われます。
