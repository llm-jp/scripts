[日本語](README_ja.md)
# Lustre setup script for mdx

This script automatically configures the [storage mount](https://docs.mdx.jp/ja/index.html#高速内部ストレージ、大容量ストレージをマウントする) for mdx, mounting the `/data` and `/model` directories using the Lustre file system.

## Prerequisites

- **OS**: Ubuntu 20.04, Ubuntu 22.04
- **mdx storage network type**: SR-IOV
- The `STORAGE_NETWORK1_IPV4` must be obtained from the node summary in the mdx user portal.
- Root privileges are required (run with `sudo`).

## Usage

Run the following command to use the script:

```shell
git clone https://github.com/llm-jp/scripts.git
cd scripts/server/mdx/lustre_setup
sudo bash setup.sh [-f] [-v] STORAGE_NETWORK1_IPV4
```

A system reboot is required after running the script.<br>
Ensure that `/model` and `/data` are mounted correctly after running the script.

### Arguments

- `STORAGE_NETWORK1_IPV4`: The IPv4 address of storage network 1. This address can be found in the node summary of the mdx user portal.

### Optional Arguments

- `-f`: Forces a reboot without user confirmation.
- `-v`: Enables debug mode, which shows detailed execution steps.

## Example

If the IPv4 address for storage network 1 is `192.168.1.100`, run the following command:

```shell
sudo bash setup.sh 192.168.1.100
```

To force a reboot without confirmation, add the `-f` option:

```shell
sudo bash setup.sh -f 192.168.1.100
```

To enable debug mode, use the `-v` option:

```shell
sudo bash setup.sh -v 192.168.1.100
```

## Notes

- A reboot is required after running the script. If the `-f` option is not used, manually reboot the system or follow the script's prompts to reboot.
- If the script does not execute correctly, recheck the IPv4 address and network configuration.
