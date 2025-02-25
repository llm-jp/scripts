#!/bin/bash

for t in $(ls tasks); do
    if [ -e tasks/$t/checkpoints/latest_checkpointed_iteration.txt ]; then
        i=$(cat tasks/$t/checkpoints/latest_checkpointed_iteration.txt)
        echo $t: $i
    else
        echo $t: -1
    fi
done | sort -k2 -n
