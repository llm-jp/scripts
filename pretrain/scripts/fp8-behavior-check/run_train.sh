#!/bin/bash

run_job() {
    echo $@
    sbatch \
        --partition=gpu-small \
        --nodes=1 \
        scripts/pretrain/scripts/fp8-behavior-check/sbatch_3.8b.sh \
        $@
}

# arg order: enabled, format, margin, interval, history, algo, wgrad, iter

# All runs are commented out for safety.

#run_job false hybrid 0 1 1 most_recent true 0 1000
#run_job false hybrid 0 1 1 most_recent true 2000 3000
#run_job false hybrid 0 1 1 most_recent true 20000 21000
#run_job false hybrid 0 1 1 most_recent true 200000 201000

#run_job true hybrid 0 1 1 most_recent true 0 1000
#run_job true hybrid 0 1 1 most_recent true 2000 3000
#run_job true hybrid 0 1 1 most_recent true 20000 21000
#run_job true hybrid 0 1 1 most_recent true 200000 201000

#run_job true e3m4 0 1 1 most_recent true 200000 201000

#run_job true hybrid 1 1 1 most_recent true 200000 201000
#run_job true hybrid 2 1 1 most_recent true 200000 201000
#run_job true hybrid 3 1 1 most_recent true 200000 201000
#run_job true hybrid 4 1 1 most_recent true 200000 201000
#run_job true hybrid 5 1 1 most_recent true 200000 201000
#run_job true hybrid 6 1 1 most_recent true 200000 201000
#run_job true hybrid 7 1 1 most_recent true 200000 201000
#run_job true hybrid 8 1 1 most_recent true 200000 201000
#run_job true hybrid 16 1 1 most_recent true 200000 201000
#run_job true hybrid 32 1 1 most_recent true 200000 201000
#run_job true hybrid 64 1 1 most_recent true 200000 201000
#run_job true hybrid 128 1 1 most_recent true 200000 201000
#run_job true hybrid 256 1 1 most_recent true 200000 201000

#run_job true hybrid 0 2 1 most_recent true 200000 201000
#run_job true hybrid 0 4 1 most_recent true 200000 201000
#run_job true hybrid 0 8 1 most_recent true 200000 201000
#run_job true hybrid 0 16 1 most_recent true 200000 201000
#run_job true hybrid 0 32 1 most_recent true 200000 201000
#run_job true hybrid 0 64 1 most_recent true 200000 201000
#run_job true hybrid 0 128 1 most_recent true 200000 201000
#run_job true hybrid 0 256 1 most_recent true 200000 201000

#run_job true hybrid 0 1 2 max true 200000 201000
#run_job true hybrid 0 1 4 max true 200000 201000
#run_job true hybrid 0 1 8 max true 200000 201000
#run_job true hybrid 0 1 16 max true 200000 201000
#run_job true hybrid 0 1 32 max true 200000 201000
#run_job true hybrid 0 1 64 max true 200000 201000
#run_job true hybrid 0 1 128 max true 200000 201000
#run_job true hybrid 0 1 256 max true 200000 201000

#run_job true hybrid 0 1 1 most_recent false 200000 201000
