#!/usr/bin/env bash
# evalviz サーバをログインノードでバックグラウンド起動 (再起動) する。セッションを閉じても残る (nohup + setsid)。
# サーバのコード (evalviz_server/) を変えたら再起動する。UI だけなら build.sh のみでよい。
#   使い方: bash evalviz/restart.sh            停止: bash evalviz/restart.sh stop
#   ログ: ~/.cache/evalviz/evalviz-<ログインノード>.log。手元から ssh -L 8766:localhost:8766 <login node> で http://localhost:8766/
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=$HOME/.cache/evalviz; mkdir -p "$LOG_DIR"
# 注意: pkill -f は自分のシェルも殺しうるので、[e] で自分のコマンドラインを除外する
for p in $(pgrep -u "$USER" -f '[e]valviz_server --root'); do kill "$p"; done
[ "${1:-}" = stop ] && { echo "stopped"; exit 0; }
sleep 1
LOG="$LOG_DIR/evalviz-$(hostname -s).log"
nohup setsid bash "$HERE/run.sh" > "$LOG" 2>&1 < /dev/null &
disown
sleep 2
echo "started: $(pgrep -u "$USER" -f '[e]valviz_server --root' | tr '\n' ' ') (log: $LOG。キャッシュが無い初回のインデックスは 1 分ほど)"
