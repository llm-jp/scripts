#!/bin/bash

# This script compares the file size of each checkpoint between gcp cloud storage and local storage.
# If the file sizes are different, an error is output.
# 
# How to use:
# $ bash check_cloud_filesize.sh
# > iter_xxxx Local size:yyyy Cloud size:yyyy
# If the file sizes are different
# > iter_xxxx Local size:yyyy Cloud size:zzzz
# > Error: iter_xxxx file sizes are different.
 
LOCAL_CKPT_DIR="/lustre/checkpoints/llama-2-172b-exp2/tp4-pp16-cp1" # FIXME: DIR_PATH to checkpoint on local storage
CLOUD_CKPT_DIR="gs://llama2-172b-checkpoint-exp2" # FIXME: DIR_PATH to checkpoint on gcp cloud storage

for iter in $(ls ${LOCAL_CKPT_DIR} | grep iter_); do
	local_size=$(du -scb ${LOCAL_CKPT_DIR}/${iter}/*/*.pt | tail -n1 | cut -f1)
	cloud_size=$(gsutil du -sc "${CLOUD_CKPT_DIR}/${iter}" | tail -n1 | awk '{print $1}')

	echo "${iter} Local size:${local_size} Cloud size:${cloud_size}"

	if [ "$local_size" -ne "$cloud_size" ]; then
		echo "Error: ${iter} file sizes are different."
	fi
done

