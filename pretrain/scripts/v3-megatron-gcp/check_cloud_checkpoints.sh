#!/bin/bash

# This script compares the file size of each checkpoint between Google Cloud Storage and local storage.
# If the file sizes are different, an error is output.
# 
# How to use:
# $ bash check_cloud_checkpoints.sh
# > iter_xxxx Local size:yyyy Cloud size:yyyy
# > iter_xxx Local md5sum:hogehoge Cloud md5sum:hogehoge
# If the file sizes are different
# > iter_xxxx Local size:yyyy Cloud size:zzzz
# > Error: iter_xxxx file sizes are different.
# > iter_xxxx Local md5sum:hogehoge Cloud md5sum:hugehuge
# > Error: iter_xxx md5 checksums are different.

LOCAL_CKPT_DIR="/lustre/checkpoints/llama-2-172b-exp2/tp4-pp16-cp1" # FIXME: DIR_PATH to checkpoint on local storage
CLOUD_CKPT_DIR="gs://llama2-172b-checkpoint-exp2" # FIXME: DIR_PATH to checkpoint on Google Cloud Storage

for iter in $(ls ${LOCAL_CKPT_DIR} | grep iter_); do
	local_size=$(du -scb ${LOCAL_CKPT_DIR}/${iter}/*/*.pt | tail -n1 | cut -f1)
	cloud_size=$(gsutil du -sc "${CLOUD_CKPT_DIR}/${iter}" | tail -n1 | awk '{print $1}')

	echo "${iter} Local size:${local_size} Cloud size:${cloud_size}"

	if [ "$local_size" -ne "$cloud_size" ]; then
		echo "Error: ${iter} file sizes are different."
	fi

	local_combine_md5=$(bash calculate_combined_md5.sh ${LOCAL_CKPT_DIR}/${iter})
	cloud_combine_md5=$(bash calculate_gcloud_combined_md5.sh ${CLOUD_CKPT_DIR}/${iter})

	echo "${iter} Local md5sum:$local_combine_md5 Cloud md5sum:$cloud_combine_md5"

	if [ "$local_combine_md5" != "$cloud_combine_md5" ]; then
		echo "Error: ${iter} md5 checksums are different."
	fi

done

