# All_Reduce Benchmark

Measure throughput between gpu when all_reduce.  
To check the value returned by all_reduce, it also returns the sum of each RANK.

## Example output

```
MASTER_ADDR=a001
NUM_NODES=53
NUM_GPUS_PER_NODE=8
NUM_GPUS=424
0 data size: 6.0 GB
tput_avg (Gbps): 1499.183349609375 busbw_avg (Gbps): 1495.6475830078125
Local Rank Total: 1484
```

## How to use

Fix `PARTITION_NAME_FIX_ME` and `NUM_NODES_FIX_ME` of `sbatch.sh`.  
If the partition name is `gpu` and `8` nodes, modify them as follows

```
#SBATCH --partition=gpu
#SBATCH --nodes 1
```

Create an `outputs` directory in the same hierarchy as the `scripts`.  
Then, submit a job.

```
sbatch scripts/cluster/gpu/all_reduce_bench/sbatch.sh
```

The output is written to `./outputs/all-reduce-{JOB_NUMBER}.out`.

```
cat ./outputs/all-reduce-{JOB_NUMBER}.out
```
