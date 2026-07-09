# 類似重複除去スクリプト

このディレクトリには、コーパスからの類似重複を1台のサーバ上で複数プロセス並列による処理で除去するためのスクリプトが含まれています。
重複除去は、[datatrove](https://github.com/huggingface/datatrove) に実装された Minhash-LSH をベースとしています。

## システム要件

動作確認を行ったシステムは以下の通りです。
- OS: Ubuntu 24.04.3 LTS
- Python: Python 3.12.3, pip 24.0 (apt によりインストール)

GNU Parallel を利用する場合は apt でインストールしてください。

```
$ sudo apt install parallel
```

## スクリプト実行順

以下の手順で実行します。

0. 事前準備

    作業スペースを作成するディレクトリを環境変数 `work_dir` にセットします。
    ここでは "/home/foo/work" とします。

            $ cd corpus/llm-jp-corpus-v4/common/dedup/
            $ export work_dir=/home/foo/work
            $ bash installer/install.sh
   
    "/home/foo/work" の下に "environment" ディレクトリが作成されます。
    次のコマンドを実行して Python 仮想環境を有効にします。

            $ . ${work_dir}/environment/.venv/bin/activate

    以降の作業は仮想環境を有効にした状態で行います。

1. ファイルサイズを均一化して処理時間のバランスを取るためのリシャーディング

    処理対象とするテキストコーパスをだいたい1GBずつになるように分割します。
    テキストコーパスは各レコードが改行で区切られた JSON ファイル (jsonl, ndjson) で
    レコードを識別する "id" とテキスト "text" フィールドを含む必要があります。

    説明のため、コーパスファイルは bzip2 で圧縮された状態、ファイル名は `*.jsonl.bz2` で、環境変数 `CORPUS_DIR` に指定されたディレクトリ内に配置されているものとします。また、分割したファイルは bzip2 で圧縮し、環境変数 `INPUT_DIR` (重複除去処理の入力になるので) で指定されたディレクトリに出力するものとします。

    次のコマンドを実行すると、 "${INPUT_DIR}" の下に "part_0000.jsonl.bz2" から "part_NNNN.jsonl.bz2" が出力されます。

            $ rm -rf ${INPUT_DIR}
            $ mkdir ${INPUT_DIR}
            $ for f in ${CORPUS_DIR}/*.jsonl.bz2; do bzip2 -dc $f; done | split - -d --suffix-length=4 --line-bytes=1G --additional-suffix='.jsonl' --filter='bzip2 -c > $FILE.bz2' ${INPUT_DIR}/part_
   
    bzip2 以外のフォーマットの場合は上のコマンドの `bzip2` を `gzip` などに置き換えてください。また分割するサイズは `--line-bytes=1G` の部分で指定しているので、コーパスが小さいなどの理由でより細かく分割したい場合は、たとえば `--line-bytes=100M` のように変更してください (細かく分割した方がより均等になります) 。

2. 重複レコード検出

    分割したテキストコーパス内のレコードに対し、各レコードのテキストから signature を計算し、十分に近いものをクラスタにまとめて各クラスタの最初の1件のみを抽出します。

    重複除去処理の結果を出力するディレクトリを環境変数 `OUTPUT_DIR` に指定します。このディレクトリはどこを指定しても構いませんが、以下の例では後でまとめて削除できるよう `${work_dir}/environment/` の下に置いています。

            $ export OUTPUT_DIR=${work_dir}/environment/dedup_result
            $ python -m minhash.local_single_node.minhash_dedup ${INPUT_DIR} ${OUTPUT_DIR}

    Stage 2 で "too many open files" というエラーが出る場合、先にオープン可能なファイル数の上限を増やすと解決する可能性があります。

            $ ulimit -n 65535

    (補足1) minhash_dedup.py は以下のオプションパラメータを指定できます。

    - `--ngram`: シグネチャの計算に利用する N-Gram の長さ、指定しない場合 5
    - `--buckets`: バケット数、指定しない場合 20 で、値を大きくすると重複文書検出の recall が向上しますが precision が低下します
    - `--hashes_per_bucket`: バケットあたりのハッシュ数、指定しない場合 10 で、値を大きくすると precision が向上しますが recall が低下します
    - `--max_worker`: 並列実行するワーカーの最大値、指定しない場合 16
    - `--stage`: どのステージまで実行するかを指定、指定しない場合 4

    (補足2) 入力コーパスの JSON の "id" や "text" フィールドの名称が違う場合や、複数の項目を組み合わせて作成する必要がある場合、 `json_format.py` の中で adapter 関数を定義することで動的に変換できます。実装例 `custom_adapter` を参考にしてください。

3. 重複レコードの ID リスト作成

    `${OUTPUT_DIR}/minhash-5gram-20buckets-10hashes/results/removed/*.jsonl.gz` に重複レコードと判定されて除去されたレコードが出力されているので、ここから除去するべきレコードの ID リストを作成します。環境変数 `RESULTS_DIR` に `results` ディレクトリのパスを指定してください。 `create_removed_idlist.py` はこの環境変数を参照するので、必ず指定する必要があります。
   
    もし minhash_dedup.py を実行する際にオプションパラメータをデフォルト値以外に指定した場合、`${OUTPUT_DIR}` の下のディレクトリ名がパラメータに合わせて変化するので、正しいディレクトリ名に置き換えてください。

            $ export RESULTS_DIR=${OUTPUT_DIR}/minhash-5gram-20buckets-10hashes/results
            $ python -m minhash.local_single_node.create_removed_idlist

    実行すると `${RESULTS_DIR}/removed_id_list.txt` に ID リストが出力されます。

4. 元のテキストコーパスから重複レコード除去

    分割前のテキストコーパスから重複レコードの ID リストに含まれるレコードを除去します。元のコーパスのファイル名、レコードの内容および順序を保持したファイルを環境変数 `DEDUPED_DIR` で指定したディレクトリの下に出力します。`filter_removed_ids.py` も環境変数 `RESULTS_DIR` を参照するので、前の手順を実行した後にシェルを再起動した場合などは再度設定してください。
   
            $ export DEDUPED_DIR=${CORPUS_DIR}/deduped  # 任意のディレクトリで良い
            $ mkdir -p ${DEDUPED_DIR}
            (GNU parallel を利用する場合)
            $ ls ${CORPUS_DIR}/*.jsonl.bz2 | parallel -j 10 "bzip2 -dc {} | python -m minhash.local_single_node.filter_removed_ids | bzip2 -c > ${DEDUPED_DIR}/{/}"
            (GNU parallel を利用しない場合)
            $ for f in ${CORPUS_DIR}/*.jsonl.bz2; do bzip2 -dc $f | python -m minhash.local_single_node.filter_removed_ids | bzip2 -c > ${DEDUPED_DIR}/${f##*/}; done
    
    GNU parallel を利用すると並列に処理するので短時間で終わります。並列数は `-j` の後の数字で指定します。 

5. 作業ディレクトリを削除

    作業完了後、 Python 仮想環境を無効化し、 `${work_dir}/environment/` 内の作業ファイルを削除します。

            $ deactivate
            $ rm -rf ${work_dir}/environment/


## 関連リポジトリ

参考：[datatrove](https://github.com/huggingface/datatrove)
