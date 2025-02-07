#!/bin/bash

# Outputs TSV
for f in /home/shared/experiments/0111_v4-setup/corpus/tokenized/*/*.num_tokens; do
    echo -e "$(basename $f .num_tokens)\t$(cat $f)";
done
