#!/bin/bash

# LLM-jp v4 model converter (PBS version)
# Usage:
#   bash convert_latest.sh \
#       /path/to/task \    ... TASK_DIR: path to the model to save

set -eu -o pipefail

task_dir=$1; shift

previous_job_id=""

for iter in $(seq 100000 5000 155000)
do
  echo "Submitting job for original iteration: ${iter}"
  
  # Build dependency option dynamically
  depend_opt="${previous_job_id:+-W depend=afterany:$previous_job_id}"
  
  # Submit job with optional dependency
  job_output=$(qsub \
    -v TASK_DIR=${task_dir},ITER=${iter},RTYPE=rt_HC \
    ${depend_opt} \
    -m n \
    -o /dev/null \
    -e /dev/null \
    convert/qsub_convert_cpu.sh)
  
  previous_job_id=$(echo "$job_output" | cut -d'.' -f1)
  echo "Job submitted with ID: $previous_job_id"
done

echo "All jobs submitted. Final job ID: $previous_job_id"
