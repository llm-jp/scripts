#!/bin/bash
 
LOCAL_CKPT_DIR="/lustre/checkpoints/llama-2-172b-exp2/tp4-pp16-cp1"
CLOUD_CKPT_DIR="gs://llama2-172b-checkpoint-exp2"


for iter in $(ls ${LOCAL_CKPT_DIR} | grep iter_); do
	local_size=$(du -scb ${LOCAL_CKPT_DIR}/${iter}/*/*.pt | tail -n1 | cut -f1)
	cloud_size=$(gsutil du -sc "${CLOUD_CKPT_DIR}/${iter}" | tail -n1 | awk '{print $1}')

	echo "${iter} Local size:${local_size} Cloud size:${cloud_size}"

	if [ "$local_size" -ne "$cloud_size" ]; then
		echo "Error: ${iter} file sizes are different."
	fi
done

