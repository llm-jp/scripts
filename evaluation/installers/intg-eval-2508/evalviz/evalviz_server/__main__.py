"""evalviz サーバ (統合評価の結果の横断分析 UI)。標準ライブラリのみ (Python 3.9 以上)。

  python -m evalviz_server --root <出力ルート[,...]> [--port 8766]       (通常は evalviz/run.sh / restart.sh)
  python -m evalviz_server check [--samples] --root ...                   全実行をインデックスし、コンポーネントごとの件数と
                                                                          読めなかった結果を報告する。--samples でサンプル分析も作り、
                                                                          再計算した exact_match・ool を result.json と突き合わせる
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import ljeval
from .httpd import Handler
from .featured import Featured
from .index import Index

HERE = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = os.environ.get("EVALVIZ_ROOTS", "")


def main():
    ap = argparse.ArgumentParser(prog="evalviz_server")
    ap.add_argument("cmd", nargs="?", default="serve", choices=["serve", "index", "check"])
    ap.add_argument("--root", default=DEFAULT_ROOTS, help="出力ルート (カンマ区切りで複数可。既定は環境変数 EVALVIZ_ROOTS)")
    ap.add_argument("--featured", default=os.environ.get("EVALVIZ_FEATURED", ""), help="掲載モデルの設定 (JSON。既定は環境変数 EVALVIZ_FEATURED。featured.example.json 参照)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--interval", type=float, default=180.0, help="再インデックス間隔 (秒)")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "evalviz" / "runs.json"), help="要約のキャッシュ ('' で無効)。サンプル分析は隣の analysis/ に置く")
    ap.add_argument("--web", default=str(HERE / "web" / "dist"), help="UI のビルド成果物")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--samples", action="store_true", help="check: サンプル分析も作って result.json と突き合わせる")
    a = ap.parse_args()
    roots = [Path(x) for x in a.root.split(",") if x]
    if not roots:
        ap.error("--root か EVALVIZ_ROOTS で出力ルートを指定する")
    idx = Index(roots, Path(a.cache) if a.cache else None, a.workers, Featured(Path(a.featured) if a.featured else None))
    t0 = time.time()
    n = idx.refresh()
    print(f"indexed {len(idx.runs)} runs ({n} read) in {time.time() - t0:.1f}s", file=sys.stderr)
    if a.cmd == "index":
        return
    if a.cmd == "check":
        sys.exit(check(idx, a.samples))
    idx.run_background(a.interval)
    Handler.index = idx
    Handler.web_dist = Path(a.web)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"serving http://{a.host}:{a.port}/  (roots={a.root})", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def check(idx: Index, samples: bool) -> int:
    """要約の失敗とサンプル分析の再計算のずれを数える (終了コードに反映)。結果の無いコンポーネントは警告だけ出す"""
    counts = collections.Counter()
    errors, warnings = [], []   # warnings は結果が無い・空のもの (未完了の投入など。データの側の事情)
    for run in idx.runs:
        if run.get("error"):
            errors.append(("summary", run["id"], run["error"]))
        for name, c in (run.get("components") or {}).items():
            items = c.items() if name == "llm-jp-eval" else [("", c)]
            for ver, cc in items:
                key = f"{name} {ver}".strip()
                counts[(key, cc.get("status"))] += 1
                if cc.get("status") != "done":
                    warnings.append((key, run["id"], cc.get("status")))
    for (k, st), n in sorted(counts.items()):
        print(f"{k:24} {st:8} {n:5}")
    if samples:
        n_ok = 0
        for run in idx.runs:
            for ver, c in ((run.get("components") or {}).get("llm-jp-eval") or {}).items():
                if c.get("status") != "done" or not c.get("has_offline"):
                    continue
                try:
                    a = idx.analysis(run["id"], ver)
                except Exception as ex:
                    errors.append(("analysis", f"{run['id']} {ver}", f"{type(ex).__name__}: {ex}"))
                    continue
                if a is None:
                    continue
                n_ok += 1
                for b in ljeval.check_against_scores(a, ljeval.raw_scores(Path(c["dir"]))):
                    errors.append(("mismatch", f"{run['id']} {ver}", b))
        print(f"サンプル分析を作った実行 × バージョン: {n_ok}")
    if idx.featured.path:
        from .featured import resolve
        cfg = idx.featured.load()
        if idx.featured.error:
            errors.append(("featured", str(idx.featured.path), idx.featured.error))
        res = resolve(cfg, {r["dir"]: r for r in idx.runs})
        print(f"掲載モデル: {len(res['models'])} ({idx.featured.path})")
        for pr in res["problems"]:
            warnings.append(("featured", pr, ""))
    for w in warnings[:80]:
        print("WARN", *w)
    for e in errors[:80]:
        print("ERROR", *e)
    print(f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    main()
