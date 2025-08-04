#!/bin/bash

update_train_data () {
  local IN="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/v4-megamath-pro-max/train_data.all.sh"
  local OUT="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/v4-megamath-pro-max/train_data_100B.sh"

  awk '
    BEGIN {
      FS = OFS = " "
    }
    function ceil(x) { return (x == int(x) ? x : int(x) + 1) }

    /^[[:space:]]*[0-9]/ {
      tok  = $1
      path = $2

      ratio = 1          # default 100%
      if (path ~ /\/dclm\//)      ratio = 0.0685   # 6.85 %
      else if (path ~ /\/pes2o\//) ratio = 0.167   # 16.7 %
      else if (path ~ /\/math\//) ratio = 2.0      # 200 %
      # flan / stackexchange / wiki は ratio = 1

      newtok = ceil(tok * ratio)
      $1 = newtok
      print
      next
    }

    { print }
  ' "$IN" > "$OUT"

  echo "Created $OUT"
}

update_train_data
