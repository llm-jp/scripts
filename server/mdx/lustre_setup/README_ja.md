[English](README.md)
# Lustre setup script for mdx

このスクリプトは、mdxの[ストレージマウント](https://docs.mdx.jp/ja/index.html#高速内部ストレージ、大容量ストレージをマウントする)を自動的に行い、`/data`および`/model`ディレクトリをLustreファイルシステムでマウントします。

## Prerequisites

- **OS**: Ubuntu 20.04, Ubuntu 22.04
- **mdxストレージネットワークタイプ**: SR-IOV
- mdxユーザーポータルから確認できる `STORAGE_NETWORK1_IPV4` が必要です。
- root権限が必要です（`sudo`を使用して実行してください）。

## Usage

以下のコマンドを実行してスクリプトを使用します。

```shell
sudo bash setup.sh [-f] [-v] STORAGE_NETWORK1_IPV4
```

スクリプト実行後に再起動が必要です。<br>
スクリプト実行後に `/model`や`/data`がマウントされているか確認してください。

### Arguments

- `STORAGE_NETWORK1_IPV4`: ストレージネットワーク1のIPv4アドレス。このアドレスはmdxユーザーポータルのノードサマリーから確認できます。

### Optional Arguments

- `-f`: 強制再起動を行います。ユーザーに確認せずにシステムを再起動します。
- `-v`: デバッグモードを有効にします。実行中の詳細な情報を表示します。

## Example

ストレージネットワーク1のIPv4アドレスが `192.168.1.100` の場合、以下のように実行します。

```shell
sudo bash setup.sh 192.168.1.100
```

再起動を確認せずに強制的に再起動したい場合は、`-f`オプションを追加します。

```shell
sudo bash setup.sh -f 192.168.1.100
```

デバッグモードを有効にする場合は、`-v`オプションを使用します。

```shell
sudo bash setup.sh -v 192.168.1.100
```

## Notes

- スクリプト実行後に再起動が必要です。`-f` オプションを使用しない場合は、実行後に手動で再起動するか、スクリプトの指示に従って再起動してください。
- スクリプトが正しく実行されない場合は、IPv4アドレスやネットワーク構成を再確認してください。
