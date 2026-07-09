# Duplicate Removal Script

This directory contains scripts for removing similar duplicates from a corpus using parallel processing across multiple processes on a single server.
Duplicate removal is based on Minhash-LSH implemented in [datatrove](https://github.com/huggingface/datatrove).

 ## System Requirements

The following systems have been tested for compatibility:
- OS: Ubuntu 24.04.3 LTS
- Python: Python 3.12.3, pip 24.0 (installed via apt)

If you are going to use GNU Parallel, install it via apt.

```
$ sudo apt install parallel
```

## Script Execution Procedure

Execute in the following order.

0. Prerequisites

    Set the directory for your workspace in the environment variable `work_dir`.
    Here, we use "/home/foo/work".

            $ cd corpus/llm-jp-corpus-v4/common/dedup/
            $ export work_dir=/home/foo/work
            $ bash installer/install.sh

    An "environment" directory will be created under "/home/foo/work".
    Run the following command to activate the Python virtual environment.

            $ . ${work_dir}/environment/.venv/bin/activate

    All subsequent operations should be performed in the activated virtual environment.

1. Resharding to balance processing time by making file sizes uniform

    Split the text corpus to be processed into roughly 1GB chunks.
    The text corpus consists of JSON files (jsonl, ndjson) where each record is separated by a line break.
    Each record must contain an "id" field for identification and a "text" field for the text content.

    For illustrative purposes, assume the corpus files are compressed with bzip2, named `*.jsonl.bz2`, and located in the directory specified by the environment variable `CORPUS_DIR`. Furthermore, assume the split files are compressed with bzip2 and output to the directory specified by the environment variable `INPUT_DIR` (as this serves as input for deduplication processing).

    Executing the following command outputs files named "part_0000.jsonl.bz2" through "part_NNNN.jsonl.bz2" under "${INPUT_DIR}".

            $ rm -rf ${INPUT_DIR}
            $ mkdir ${INPUT_DIR}
            $ for f in ${CORPUS_DIR}/*.jsonl.bz2; do bzip2 -dc $f; done | split - -d --suffix-length=4 --line-bytes=1G --additional-suffix='.jsonl' --filter='bzip2 -c > $FILE.bz2' ${INPUT_DIR}/part_

    For formats other than bzip2, replace `bzip2` in the above command with `gzip` or so. The split size is specified by `--line-bytes=1G`. If you want finer splits (e.g., due to a small corpus), change it to something like `--line-bytes=100M` (finer splits result in more even distribution).

2. Duplicate Record Detection

    For each record in the split text corpus, calculate a signature from its text. Group sufficiently similar records into clusters and extract only the first record from each cluster.

    Specify the directory for outputting the deduplication results using the environment variable `OUTPUT_DIR`. This directory can be specified anywhere, but in the example below, it is placed under `${work_dir}/environment/` so it can be deleted later in bulk.

            $ export OUTPUT_DIR=${work_dir}/environment/dedup_result
            $ python -m minhash.local_single_node.minhash_dedup ${INPUT_DIR} ${OUTPUT_DIR}

    If you encounter a "too many open files" error during Stage 2, increasing the maximum number of open files beforehand may resolve the issue.

            $ ulimit -n 65535

    (Note 1) minhash_dedup.py accepts the following option parameters.

    - `--ngram`: Length of the N-Gram used for signature calculation. Defaults to 5 if not specified.
    - `--buckets`: Number of buckets. Default is 20. Increasing this value improves recall for duplicate document detection but reduces precision.
    - `--hashes_per_bucket`: Number of hashes per bucket. Default is 10. Increasing this value improves precision but reduces recall.
    - `--max_worker`: Maximum number of parallel workers. Default is 16.
    - `--stage`: Specifies the stage up to which to execute. Default is 4.

    (Note 2) If the names of the "id" or "text" fields in the input corpus JSON differ, or if multiple items need to be combined to create the output, you can dynamically transform the data by defining an adapter function within `json_format.py`. Please refer to the implementation example `custom_adapter`.

3. Creating a List of IDs for Duplicate Records

    Records identified as duplicates and removed are output to `${OUTPUT_DIR}/minhash-5gram-20buckets-10hashes/results/removed/*.jsonl.gz`. Create a list of IDs for records to be removed from this file. Specify the path to the `results` directory in the environment variable `RESULTS_DIR`. Since `create_removed_idlist.py` references this environment variable, it must be specified.
   
    If you specified non-default option parameters when running `minhash_dedup.py`, the directory name under `${OUTPUT_DIR}` will change accordingly. Replace it with the correct directory name.

            $ export RESULTS_DIR=${OUTPUT_DIR}/minhash-5gram-20buckets-10hashes/results
            $ python -m minhash.local_single_node.create_removed_idlist

    Running this will output the ID list to `${RESULTS_DIR}/removed_id_list.txt`.

4. Remove duplicate records from the original text corpus

    Removes records contained in the duplicate ID list from the pre-split text corpus. Outputs files that preserve the original corpus filenames, record contents, and order to the directory specified by the environment variable `DEDUPED_DIR`. Since `filter_removed_ids.py` also references the environment variable `RESULTS_DIR`, you must reset it if you restart the shell after executing the previous step.

            $ export DEDUPED_DIR=${CORPUS_DIR}/deduped  # Any directory is acceptable
            $ mkdir -p ${DEDUPED_DIR}
            (When using GNU parallel)
            $ ls ${CORPUS_DIR}/*.jsonl.bz2 | parallel -j 10 "bzip2 -dc {} | python -m minhash.local_single_node.filter_removed_ids | bzip2 -c > ${DEDUPED_DIR}/{/}"
            (When not using GNU parallel)
            $ for f in ${CORPUS_DIR}/*.jsonl.bz2; do bzip2 -dc $f | python -m minhash.local_single_node.filter_removed_ids | bzip2 -c > ${DEDUPED_DIR}/${f##*/}; done

    Using GNU parallel enables parallel processing, completing the task quickly. The number of parallel processes is specified by the number after `-j`. 

5. Delete the working directory

    After completing the work, deactivate the Python virtual environment and delete the working files in `${work_dir}/environment/`.

            $ deactivate
            $ rm -rf ${work_dir}/environment/


 ## Related Repositories

 Reference: [datatrove](https://github.com/huggingface/datatrove)
