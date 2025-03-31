# Deduplication Scripts

This directory contains scripts for near-duplicate removal from corpora.
The deduplication process is based on Minhash-LSH implemented in [datatrove](https://github.com/huggingface/datatrove).

We perform deduplication in two main stages:
- Deduplication within each individual corpus
- Global deduplication across all corpora that have been locally deduplicated

## Script Execution Order

0. Install required libraries  
   - `installer/install.sh`

1. Reshard files to equalize file sizes and balance processing time  
   - `preprocess/reshard_all.sh`

2. Perform deduplication within each corpus  
   - `minhash`
   - See `minhash/README.md` for details

3. Collect all preprocessed files into a single directory using symbolic links  
   - `preprocess/make_links.sh`

4. Perform global deduplication across all corpora  
   - `minhash`
   - See `minhash/README.md` for details

5. Reorganize the deduplicated files  
   - Deduplicated files are saved without preserving directory structure.
   - Steps:
     1. Verify that texts are not randomized during deduplication  
        - `postprocess/check_original_path_consisitency.py`
     2. Reconstruct original directory structure for each corpus  
        - `postprocess/reconstruct_stracture.py`

## Related Repository

See also: [datatrove](https://github.com/huggingface/datatrove)
