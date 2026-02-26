# 重複除去コード

このディレクトリには、[datatrove](https://github.com/huggingface/datatrove) を使用した重複除去スクリプトが含まれています。  
SLURMクラスタおよび複数のローカルサーバーの両方で実行できるように構成されています。

## SLURM 環境

このバージョンは SLURM クラスタ上での実行を想定しています。

### 使い方

```bash
python slurm/minhash_dedup.py {input_dir} {output_dir}
```

- `input_dir`: 入力データが格納されたディレクトリのパス。サブディレクトリも再帰的に走査されます。
- `output_dir`: 重複除去後のデータを保存するディレクトリ。出力サブディレクトリは自動で作成されます。
- ハッシュ処理に関するハイパーパラメータ（例：n-gram サイズ、バケット数、ハッシュ数）を設定可能です。詳細はスクリプト内のコメントを参照してください。

> `slurm/minhash_dedup.py` は [こちらの公式サンプル](https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py) をもとに作成されています。

## ローカルマルチノード環境

このバージョンは、SSH 経由で複数のローカルマシンにまたがって分散実行することを想定しています。

### 構成

- `local_multi_node/submit_minhash_all_subcorpus.sh`: 全てのサブコーパスに対して重複除去を実行するシェルスクリプト。
- `local_multi_node/submit_minhash.py`: ノードリストを読み込み、各ノードで重複除去処理を起動するランチャースクリプト。
- `local_multi_node/minhash_dedup.py`: 各ノード上で実行されるワーカースクリプト。

> ※ このコードはプロトタイプ段階ですが、参考実装として共有します。
