#!/bin/bash

# This script calculates the md5sum of all files in a given directory (and its subdirectories)
# and outputs the combined md5sum using parallel processing for faster execution.
#
# How to use:
# bash calculate_combined_md5.sh $LOCAL_DIR
# > xxxxxxxxxxx(md5sum)

if [ $# -ne 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

DIRECTORY=$1

if [ ! -d "$DIRECTORY" ]; then
  echo "Error: Directory $DIRECTORY does not exist."
  exit 1
fi

# Create a temporary file to store individual md5sums
temp_file=$(mktemp)

# Find all files and calculate their md5sums in parallel, store results in the temp file
find "$DIRECTORY" -type f | xargs -P 80 -I {} md5sum "{}" | awk '{print $1}' >> "$temp_file"

# Sort the results and calculate the combined md5sum
combined_md5=$(sort "$temp_file" | md5sum | awk '{print $1}')

# Clean up the temporary file
rm "$temp_file"

echo "$combined_md5"

