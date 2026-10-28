#!/usr/bin/env bash
# evalviz の UI (evalviz/web: Vite + React + TS) をビルドする。成果物 web/dist/ をサーバ (evalviz_server) が配信する。
#   使い方: bash evalviz/build.sh            依存を入れ直す: bash evalviz/build.sh --install
# UI だけの変更ならサーバの再起動は要らない (ブラウザの再読込のみ)。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)/web
# mise の node を使う (無ければ PATH の node)
NODE_BIN=$(ls -d "$HOME"/.local/share/mise/installs/node/*/bin 2>/dev/null | sort -V | tail -1 || true)
[ -n "$NODE_BIN" ] && export PATH="$NODE_BIN:$PATH"
command -v node >/dev/null || { echo "node が見つかりません (mise install node)" >&2; exit 1; }
# pnpm は aqua / mise 等で入っていればそれを、無ければ npx 経由で使う
if command -v pnpm >/dev/null; then PNPM=(pnpm); else PNPM=(npx --yes pnpm@12); fi
cd "$HERE"
{ [ "${1:-}" = --install ] || [ ! -d node_modules ]; } && "${PNPM[@]}" install --frozen-lockfile
"${PNPM[@]}" run build
echo "built: $HERE/dist"
