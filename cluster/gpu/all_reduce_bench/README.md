## How to use

Fix nodes and partition of `sbatch.sh`.

```
#SBATCH --partition=gpu
#SBATCH --nodes 64
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