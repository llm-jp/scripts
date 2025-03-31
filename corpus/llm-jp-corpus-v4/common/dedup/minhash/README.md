# Deduplication Code

This directory provides deduplication scripts using [datatrove](https://github.com/huggingface/datatrove).
It supports execution in two environments: on a SLURM cluster and on multiple local servers.

## SLURM

This version is designed for use on a SLURM-based cluster.

### Usage

```bash
python slurm/minhash_dedup.py {input_dir} {output_dir}
```

- `input_dir`: Path to the directory containing the input data. The script recursively scans subdirectories for files.
- `output_dir`: Path where deduplicated files will be written. Subdirectories will be automatically created under this path.
- You can also configure hyperparameters related to hashing (e.g., n-gram size, number of buckets, number of hashes per bucket).
  Please refer to the comments in the code for details.

> The script `slurm/minhash_dedup.py` was adapted from [this official example](https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py).

## Local Multi-Node

This version supports deduplication across multiple local machines using distributed processing via SSH.

### Structure

- `local_multi_node/submit_minhash_all_subcorpus.sh`: Main launcher shell script to deduplicate all sub-corpus.
- `local_multi_node/submit_minhash.py`: Main launcher that reads the node list and runs deduplication on each machine.
- `local_multi_node/minhash_dedup.py`: Worker script executed on each node.

> Note: This code is a prototype, but it is shared here for reference.
