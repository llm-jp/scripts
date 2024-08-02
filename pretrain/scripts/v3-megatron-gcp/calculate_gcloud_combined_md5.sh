
#!/bin/bash

# This script calculates the combined md5sum of all files in a given Google Cloud Storage bucket/directory.
# The combined md5sum is calculated by fetching the md5sums of each file and then combining them.
#
# How to use:
# bash calculate_gcloud_combined_md5.sh $GOOGLE_CLOUD_DIR
# > xxxxxxxxxxx(md5sum)

if [ -z "$1" ]; then
  echo "Usage: $0 <gcs-directory>"
  exit 1
fi

GCS_DIRECTORY=$1

# Create a temporary file to store individual md5sums
temp_file=$(mktemp)

# Function to convert base64 md5 to hex
base64_to_hex() {
  echo "$1" | base64 --decode | od -An -t x1 | tr -d ' \n'
}

# List all files in the GCS directory and fetch their md5sums
gsutil ls -r "$GCS_DIRECTORY/**" | grep -v '/$' | while read -r file; do
  md5_base64=$(gsutil stat "$file" | grep "Hash (md5)" | awk '{print $3}')
  if [ -n "$md5_base64" ]; then
    md5_hex=$(base64_to_hex "$md5_base64")
    echo "$md5_hex"
  fi
done > "$temp_file"

# Sort the md5 sums and calculate the combined md5sum
combined_md5=$(sort "$temp_file" | md5sum | awk '{print $1}')

# Clean up the temporary file
rm "$temp_file"

echo "$combined_md5"

