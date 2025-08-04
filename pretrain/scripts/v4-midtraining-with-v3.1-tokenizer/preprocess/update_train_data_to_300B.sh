#!/bin/bash

update_train_data () {
  local IN="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/v4-megamath-pro-max/train_data.all.sh"
  local OUT="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining-with-v3.1-tokenizer/tasks/v4-megamath-pro-max/train_data_300B.sh"

  awk '
    BEGIN {
      FS = OFS = " "
    }
    function ceil(x) { return (x == int(x) ? x : int(x) + 1) }

    /^[[:space:]]*[0-9]/ {
      tok  = $1
      path = $2

      ratio = 1          # default 100%
      if (path ~ /\/dclm\//)      ratio = 0.2078            # 20.78 %
      else if (path ~ /\/flan\//) ratio = 2.0               # 200 %
      else if (path ~ /\/stackexchange\//) ratio = 4.0      # 400 %
      else if (path ~ /\/math\//) ratio = 4.0               # 400 %
      else if (path ~ /\/wiki\//) ratio = 4.0               # 400 %
      # peS2o ratio = 1

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
