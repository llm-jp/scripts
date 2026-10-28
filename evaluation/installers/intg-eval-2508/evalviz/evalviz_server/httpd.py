"""JSON API と SPA の配信。

GET /api/meta                                     状態・フラグ・カテゴリの定義、インデックスの状態
GET /api/refresh                                  再インデックスを要求 (非同期)
GET /api/featured                                 掲載モデル (リーダーボード) と標準条件
GET /api/runs                                     実行一覧 (コンポーネントごとのスコアの要約付き)
GET /api/runs/<run>                               実行の詳細 (全スコア)
GET /api/analysis?runs=a,b&ver=v2.1.5             サンプル分析の実行ごとの集計 (状態・フラグ・データセット別・ラベル分布)
GET /api/samples?runs=a,b&ver=&ds=                データセット 1 つのサンプル × 実行の表
GET /api/sample?runs=a,b&ver=&ds=&key=            サンプル 1 件の全実行の生成 (プロンプト・推論・出力)
それ以外のパスは web/dist の静的ファイル、無ければ index.html (クライアント側ルーティング)。
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import featured, ljeval, meta
from .index import Index


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


ROUTES = []


def route(pattern: str):
    rx = re.compile("^" + pattern + "$")

    def deco(fn):
        ROUTES.append((rx, fn))
        return fn
    return deco


def _ids(q, key) -> list:
    return [x for x in (q.get(key) or "").split(",") if x]


def _public(run: dict) -> dict:
    """一覧用。config (qsub.py の引数全体) は詳細でだけ返す"""
    return {k: v for k, v in run.items() if k != "config"}


@route(r"/api/meta")
def api_meta(idx: Index, q):
    return {**meta.meta(), "built_at": idx.built_at, "refreshing": idx.refreshing, "roots": [str(r) for r in idx.roots], "last_error": idx.last_error}


@route(r"/api/refresh")
def api_refresh(idx: Index, q):
    idx.request_refresh()
    return {"ok": True, "built_at": idx.built_at, "refreshing": idx.refreshing}


@route(r"/api/featured")
def api_featured(idx: Index, q):
    """掲載モデルと、モデルごとに評価の値を取る実行 (source)。設定に書かれた実行がまだインデックスに無ければ再インデックスを要求する"""
    cfg = idx.featured.load()
    with idx.lock:
        by_dir = {r["dir"]: r for r in idx.runs}
    if any(d not in by_dir for d in idx.featured.run_dirs()):
        idx.request_refresh()
    out = featured.resolve(cfg, by_dir)
    out.update(path=str(idx.featured.path or ""), error=idx.featured.error)
    return out


@route(r"/api/runs")
def api_runs(idx: Index, q):
    with idx.lock:
        return [_public(r) for r in idx.runs]


@route(r"/api/runs/([^/]+)")
def api_run(idx: Index, q, rid):
    run = idx.run(rid)
    if not run:
        raise ApiError(404, f"unknown run {rid}")
    out = dict(run)
    out["raw_scores"] = {ver: ljeval.raw_scores(vd) for ver, vd in ljeval.version_dirs(Path(run["dir"])).items() if not ver.startswith("v1")}
    return out


def _analyses(idx: Index, rids: list, ver: str) -> dict:
    """実行 ID -> サンプル分析 (無ければ None)。初回は推論結果を読むので並列に作る"""
    if not ver:
        raise ApiError(400, "ver が必要")
    with ThreadPoolExecutor(min(8, max(1, len(rids)))) as ex:
        return dict(zip(rids, ex.map(lambda r: idx.analysis(r, ver), rids)))


@route(r"/api/analysis")
def api_analysis(idx: Index, q):
    ver = q.get("ver", "")
    out = []
    for rid, a in _analyses(idx, _ids(q, "runs"), ver).items():
        if a is None:
            out.append({"run": rid, "version": ver, "available": False})
        else:
            out.append({"run": rid, "version": ver, "available": True, **ljeval.summarize_analysis(a)})
    return out


@route(r"/api/samples")
def api_samples(idx: Index, q):
    ver, ds = q.get("ver", ""), q.get("ds", "")
    rids = _ids(q, "runs")
    ana = _analyses(idx, rids, ver)
    rows: dict = {}
    info = None
    for i, rid in enumerate(rids):
        a = (ana.get(rid) or {}).get(ds)
        if not a:
            continue
        info = info or {k: a[k] for k in ("dataset", "lang", "metrics", "pattern", "labels", "judge", "approx")}
        for r in a["rows"]:
            row = rows.get(r["k"])
            if row is None:
                row = rows[r["k"]] = {"key": r["k"], "index": r["i"], "input": r["in"], "gold": r["g"], "cells": [None] * len(rids)}
            row["cells"][i] = {"st": r["st"], "p": r["p"], "fl": r["fl"], "ol": r["ol"], "rl": r["rl"]}
    return {"info": info, "rows": sorted(rows.values(), key=lambda r: r["index"])}


@route(r"/api/sample")
def api_sample(idx: Index, q):
    ver, ds, key = q.get("ver", ""), q.get("ds", ""), q.get("key", "")
    rids = _ids(q, "runs")
    ana = _analyses(idx, rids, ver)
    out = []
    for rid in rids:
        od = idx.offline(rid, ver)
        a = (ana.get(rid) or {}).get(ds)
        row = next((r for r in (a or {}).get("rows", []) if r["k"] == key), None)
        d = ljeval.sample_detail(od, ds, key) if od and row else None
        out.append({"run": rid, "found": d is not None, **(d or {}),
                    **({"status": row["st"], "pred": row["p"], "flags": row["fl"]} if row else {})})
    info = next(({k: a[k] for k in ("dataset", "lang", "metrics", "pattern", "labels", "judge", "approx")}
                 for a in ((ana.get(r) or {}).get(ds) for r in rids) if a), None)
    return {"info": info, "key": key, "runs": out}


class Handler(BaseHTTPRequestHandler):
    index: Index = None
    web_dist: Path = None

    def log_message(self, fmt, *args):
        if os.environ.get("EVALVIZ_DEBUG"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def _static(self, path: str):
        dist = self.web_dist
        if not dist or not (dist / "index.html").is_file():
            return self._send(503, "UI が未ビルド: bash evalviz/build.sh".encode(), "text/plain; charset=utf-8")
        target = None
        if path not in ("", "/"):
            t = (dist / path.lstrip("/")).resolve()
            target = t if str(t).startswith(str(dist.resolve()) + os.sep) else None
        if target is None or not target.is_file():
            target = dist / "index.html"   # クライアント側ルート
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        # Vite の assets/ はファイル名にハッシュが入るので長くキャッシュしてよい。index.html は毎回取り直す
        cache = "public, max-age=31536000, immutable" if "/assets/" in str(target) else "no-store"
        return self._send(200, target.read_bytes(), ctype, cache)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if not u.path.startswith("/api/"):
                return self._static(u.path)
            for rx, fn in ROUTES:
                m = rx.match(u.path)
                if m:
                    return self._json(fn(self.index, q, *[urllib.parse.unquote(g) for g in m.groups()]))
            raise ApiError(404, "not found")
        except ApiError as ex:
            return self._json({"error": str(ex)}, ex.code)
        except BrokenPipeError:
            pass
        except Exception as ex:
            return self._json({"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()}, 500)
