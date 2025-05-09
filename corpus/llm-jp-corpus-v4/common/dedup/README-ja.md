# 類似重複除去スクリプト

このディレクトリには、コーパスからの類似重複を除去するためのスクリプトが含まれています。  
重複除去は、[datatrove](https://github.com/huggingface/datatrove) に実装された Minhash-LSH をベースとしています。

重複除去は以下の2段階で行います：
- 各コーパス内での重複除去
- 各コーパスでの重複除去後、全体での重複除去

## スクリプト実行順

0. 必要なライブラリのインストール  
   - `installer/install.sh`

1. ファイルサイズを均一化して処理時間のバランスを取るためのリシャーディング  
   - `preprocess/reshard_all.sh`

2. 各コーパスごとの重複除去  
   - `minhash`  
   - 詳細は `minhash/README.md` を参照

3. シンボリックリンクを用いて、前処理済みのファイルを1つのディレクトリに集約  
   - `preprocess/make_links.sh`

4. 全コーパスを対象としたグローバルな重複除去  
   - `minhash`  
   - 詳細は `minhash/README.md` を参照

5. 重複除去されたファイルの再配置  
   - 重複除去後のファイルはディレクトリ構造を保持せずに保存されます。  
   - 以下の手順で再配置を行います：
     1. 重複除去中にテキストの順序がランダム化されていないことを確認  
        - `postprocess/check_original_path_consisitency.py`
     2. 各コーパスの元のディレクトリ構造を復元  
        - `postprocess/reconstruct_stracture.py`

## 関連リポジトリ

参考：[datatrove](https://github.com/huggingface/datatrove)
