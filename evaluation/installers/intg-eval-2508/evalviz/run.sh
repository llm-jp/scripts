#!/usr/bin/env bash
# evalviz サーバをフォアグラウンドで起動する (常駐は restart.sh)。手元から ssh -L 8766:localhost:8766 <login node> で http://localhost:8766/
#   EVALVIZ_ROOTS: 出力ルート (カンマ区切り。既定は下の DEFAULT_ROOTS)。qsub.py の <output_dir> を並べた親ディレクトリを指定する
#   EVALVIZ_FEATURED: 掲載モデル (トップページのリーダーボード) の設定。既定は gcg51557 で共有の下記ファイル (書式は featured.example.json)
#   EVALVIZ_CACHE: 要約のキャッシュ (既定 ~/.cache/evalviz/runs.json。サンプル分析は隣の analysis/)
#   PY: Python 3.9 以上 (既定 python3)。PORT (既定 8766)。追加の引数は evalviz_server にそのまま渡す (python -m evalviz_server --help)
HERE=$(cd "$(dirname "$0")" && pwd)
E=/groups/gcg51557/experiments
DEFAULT_ROOTS=$E/0219_dev_eval_script,$E/0357_code-rl/eval,$E/0312_mid-phase1-hparams/final_eval
ROOTS=${EVALVIZ_ROOTS:-$DEFAULT_ROOTS}
CACHE=${EVALVIZ_CACHE:-$HOME/.cache/evalviz/runs.json}
FEATURED=${EVALVIZ_FEATURED:-/groups/gcg51557/evaluation2/evalviz/featured.json}
cd "$HERE"
exec "${PY:-python3}" -m evalviz_server --root "$ROOTS" --cache "$CACHE" --featured "$FEATURED" --port "${PORT:-8766}" "$@"
