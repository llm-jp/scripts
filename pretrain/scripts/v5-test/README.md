# v5 test

v4-midtraining のコードをもとにした、v5用のMegatron-LMの学習のテスト用スクリプト

## 環境構築

```bash
cd installer-abci/
bash run_setup.sh <install_path>
```

## 実行

```bash
# bash train/run_train.sh <task_dir> <param_name> <num_nodes>
bash train/run_train.sh $(realpath tasks/v4-dolmino-mix-1124) 7.7b-llama3-ecjk 1
```

## Notes

- `--attention-backend=auto` の設定と競合するため `NVTE_FUSED_ATTN` の設定を除去
- installer-abci/src/install_megatron_lm.sh において extention name を `helpers_cpp` に変更
