#!/bin/bash

update_train_data () {
  local IN="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining/tasks/v4-dolmino-mix-1124/train_data.all.sh"
  local OUT="/groups/gcg51557/experiments/0156_olmo2-midtrain-reproduction/scripts/pretrain/scripts/v4-midtraining/tasks/v4-dolmino-mix-1124/train_data.sh"

  awk '
    BEGIN {
      FS = OFS = " "
    }
    function ceil(x) { return (x == int(x) ? x : int(x) + 1) }

    /^[[:space:]]*[0-9]/ {
      tok  = $1
      path = $2

      ratio = 1          # default 100%
      if (path ~ /\/dclm\//)      ratio = 0.0323   # 3.23 %
      else if (path ~ /\/flan\//) ratio = 0.5      # 50 %
      else if (path ~ /\/pes2o\//) ratio = 0.0515  # 5.15 %
      # math / stackexchange / wiki は ratio = 1 のまま

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

