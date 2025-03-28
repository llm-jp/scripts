#!/bin/bash

# This script processes text files in a multi-level directory structure,
# filters out files listed in an exclusion list, and converts the remaining
# files into JSONL format while utilizing parallel processing.

# Define input folder and exclusion file path
DATA_ROOT="/model/llm-jp-corpus/v4.0.0/download/patent"
INPUT_DIR="${DATA_ROOT}/dataset"
EXCLUDE_FILE="${DATA_ROOT}/block_list.tsv"
OUTPUT_ROOT="/model/experiments/0122_filter_patent"
TMP_PARENT_DIR="${OUTPUT_ROOT}"
NUM_PROCESSES=50

# Create a temporary directory
TEMP_DIR=$(mktemp -d -p "$TMP_PARENT_DIR")
TEMP_EXCLUDE_FILE="$TEMP_DIR/exclude_list.txt"
PROGRESS_FILE="$TEMP_DIR/progress.txt"
FILE_COUNT_FILE="$TEMP_DIR/file_counts.txt"

# Generate an exclusion file list
awk -F'\t' 'NR > 1 {print $3}' "$EXCLUDE_FILE" | sed "s|^|$DATA_ROOT/|" > "$TEMP_EXCLUDE_FILE"

# Retrieve the list of folders two levels deep
find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -type d > "$TEMP_DIR/folder_list.txt"

# Initialize the progress file
> "$PROGRESS_FILE"

# Count the number of files in each folder and calculate the total number of files
TOTAL_FILES=0
while read -r folder; do
    file_count=$(find "$folder" -type f -name "*.txt" | grep -v -F -f "$TEMP_EXCLUDE_FILE" | wc -l)
    echo "$folder $file_count" >> "$FILE_COUNT_FILE"
    TOTAL_FILES=$((TOTAL_FILES + file_count))
done < "$TEMP_DIR/folder_list.txt"

# Function to update the progress bar
update_progress() {
    local processed_files=0

    # Calculate the number of processed files
    while read -r folder; do
        if grep -q "^$folder$" "$PROGRESS_FILE"; then
            folder_files=$(grep "^$folder " "$FILE_COUNT_FILE" | awk '{print $2}')
            processed_files=$((processed_files + folder_files))
        fi
    done < "$TEMP_DIR/folder_list.txt"

    # Display progress
    echo -ne "\rProgress: $processed_files/$TOTAL_FILES files"
}

# Function to process each folder in parallel
process_folder() {
    local folder="$1"
    local relative_path=${folder#"$INPUT_DIR/"}
    local dirname=$(dirname "$relative_path")
    local subdirname=$(basename "$relative_path")
    local output_file="${OUTPUT_ROOT}/data/${dirname}-${subdirname}.jsonl"

    # List target files and filter out excluded files
    find "$folder" -type f -name "*.txt" | grep -v -F -f "$TEMP_EXCLUDE_FILE" | while read -r filepath; do
        file_content=$(<"$filepath")
        relative_file_path=${filepath#"$DATA_ROOT/"}
        json_line=$(printf '{"text": %s, "meta": {"local_path": "%s"}}\n' \
            "$(echo "$file_content" | jq -Rs .)" "$relative_file_path")

        # Append to the output file
        echo "$json_line" >> "$output_file"
    done

    # Record progress
    echo "$folder" >> "$PROGRESS_FILE"

    # Update progress bar
    update_progress
}

export -f process_folder update_progress
export OUTPUT_ROOT PROGRESS_FILE INPUT_DIR DATA_ROOT TEMP_EXCLUDE_FILE FILE_COUNT_FILE TOTAL_FILES TEMP_DIR

# Execute parallel processing
xargs -P "$NUM_PROCESSES" -a "$TEMP_DIR/folder_list.txt" -I {} bash -c 'process_folder "$@"' _ {}

# Display final progress update
echo -e "\nCompleted processing $TOTAL_FILES files."

# Remove the temporary directory
rm -rf "$TEMP_DIR"
