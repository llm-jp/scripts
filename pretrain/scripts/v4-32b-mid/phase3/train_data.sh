# v4 mid-training dataset with v4 tokenization

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
mapfile TRAIN_DATA_PATH < <(python ${SCRIPT_DIR}/configure_corpus.py)
