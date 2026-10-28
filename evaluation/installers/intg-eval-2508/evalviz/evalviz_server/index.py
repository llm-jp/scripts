"""実行のインデックスとサンプル分析のキャッシュ。

実行の要約 (スコア) は、結果ファイルの mtime が変わらない限り読み直さない。要約はディスク (--cache) にも保存し、
再起動直後から前回の一覧を返す。再インデックスはバックグラウンドで定期実行する。
サンプル分析 (llm-jp-eval の推論結果を 1 件ずつ分類したもの) は重いので、UI が要求した実行 × バージョンだけ作り、
推論結果ファイルの mtime をキーにディスク (--cache の隣の analysis/) に保存する。
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from . import components, ljeval
from .discover import discover_runs
from .featured import Featured
from .util import iso, mtime, read_json

SUMMARY_VERSION = 2   # 要約の形を変えたら上げる (ディスクキャッシュを無効にする)
ANALYSIS_MEM = 48     # メモリに置くサンプル分析の数 (実行 × バージョン)


def summary_key(d: Path) -> list:
    """要約の再計算が要るかどうかを決めるファイルの mtime"""
    files = [d / "config.json", d / "swallow/results/result.json", d / "llm-jp-judge/evaluation/score_table.json"]
    for _, vd in sorted(ljeval.version_dirs(d).items()):
        files.append(vd / "results/result.json")
        files.append(vd / "offline")
    sc = d / "safety-eval" / "evaluate_count"
    if sc.is_dir():
        files += sorted(sc.glob("*_evaluate"))
    return [SUMMARY_VERSION] + [mtime(f) for f in files]


def summarize_run(run: dict) -> dict:
    d = Path(run["dir"])
    comps = {}
    lj = {ver: ljeval.summarize(ver, vd) for ver, vd in ljeval.version_dirs(d).items()}
    if lj:
        comps["llm-jp-eval"] = lj
    for name, fn in (("swallow", components.swallow), ("llm-jp-judge", components.judge), ("safety-eval", components.safety)):
        c = fn(d)
        if c is not None:
            comps[name] = c
    last = max([0.0] + [c.get("result_mtime") or 0.0 for c in lj.values()]
               + [c.get("result_mtime") or 0.0 for k, c in comps.items() if k != "llm-jp-eval"])
    return {"components": comps, "finished_at": iso(last), "created_at": iso(mtime(d / "config.json") or mtime(d))}


class Index:
    def __init__(self, roots: list, cache_path: Optional[Path], workers: int = 16, featured: Optional[Featured] = None):
        self.roots = roots
        self.featured = featured or Featured(None)
        self.cache_path = cache_path
        self.workers = workers
        self.lock = threading.Lock()
        self.runs: list = []
        self._cache: dict = {}                    # 実行ディレクトリ -> {key, summary}
        self._analysis: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self._analysis_locks: dict = collections.defaultdict(threading.Lock)
        self._offline: dict = {}                  # (バージョンのディレクトリ, mtime...) -> (推論結果のディレクトリ,)
        self.built_at = 0.0
        self.refreshing = False
        self.last_error = ""
        self._wake = threading.Event()
        self._load_cache()

    # ---- ディスクキャッシュ
    def _load_cache(self):
        if not self.cache_path or not self.cache_path.is_file():
            return
        data = read_json(self.cache_path) or {}
        if data.get("version") == SUMMARY_VERSION:
            self._cache = data.get("runs") or {}

    def _save_json(self, path: Path, obj) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.uname().nodename}.{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except Exception as ex:
            print(f"cache save failed ({path}): {ex}", file=sys.stderr)

    # ---- 再インデックス
    def request_refresh(self):
        self._wake.set()

    def run_background(self, interval: float):
        def loop():
            while True:
                self._wake.wait(interval)
                self._wake.clear()
                self.refresh_logged()
        threading.Thread(target=loop, daemon=True).start()

    def refresh_logged(self):
        self.refreshing = True
        t0 = time.time()
        try:
            n = self.refresh()
            print(f"reindexed {len(self.runs)} runs ({n} re-read) in {time.time() - t0:.1f}s", file=sys.stderr)
            self.last_error = ""
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.last_error = str(ex)
        self.refreshing = False

    def refresh(self) -> int:
        runs = discover_runs(self.roots, tuple(self.featured.run_dirs()))
        keys = {r["dir"]: summary_key(Path(r["dir"])) for r in runs}
        todo = [r for r in runs if (self._cache.get(r["dir"]) or {}).get("key") != keys[r["dir"]]]

        def one(r):
            try:
                return summarize_run(r)
            except Exception as ex:
                return {"components": {}, "error": f"{type(ex).__name__}: {ex}"}
        with ThreadPoolExecutor(self.workers) as ex:
            fresh = dict(zip((r["dir"] for r in todo), ex.map(one, todo)))
        new_cache = {}
        for r in runs:
            c = {"key": keys[r["dir"]], "summary": fresh[r["dir"]]} if r["dir"] in fresh else self._cache[r["dir"]]
            new_cache[r["dir"]] = c
            r.update(c["summary"])
        runs.sort(key=lambda r: (r["group"], r["name"]))
        with self.lock:
            self.runs, self.built_at = runs, time.time()
        changed = bool(fresh) or set(new_cache) != set(self._cache)
        self._cache = new_cache
        if changed and self.cache_path:
            self._save_json(self.cache_path, {"version": SUMMARY_VERSION, "runs": self._cache})
        return len(fresh)

    # ---- 参照
    def run(self, rid: str) -> Optional[dict]:
        return next((r for r in self.runs if r["id"] == rid), None)

    def offline(self, rid: str, ver: str) -> Optional[Path]:
        run = self.run(rid)
        if not run:
            return None
        vd = ljeval.version_dirs(Path(run["dir"])).get(ver)
        if not vd or ver.startswith("v1"):
            return None
        # result.json の config (推論結果のディレクトリ名) は大きいファイルだと読むのに時間がかかるので、mtime をキーに覚えておく
        rpath = vd / "results" / "result.json"
        key = (str(vd), mtime(rpath), mtime(vd / "offline"))
        hit = self._offline.get(key)
        if hit is None:
            cfg = ljeval.read_json_keys(rpath, ("config",)).get("config") or {}
            hit = self._offline[key] = (ljeval.offline_dir(vd, cfg),)
        return hit[0]

    def analysis(self, rid: str, ver: str) -> Optional[dict]:
        """{dataset: analyze_dataset の結果}。推論結果が無ければ None"""
        od = self.offline(rid, ver)
        if od is None:
            return None
        key = ljeval.analysis_key(od)
        mem_key = str(od)
        with self._analysis_locks[mem_key]:
            hit = self._analysis.get(mem_key)
            if hit and hit["key"] == key:
                self._analysis.move_to_end(mem_key)
                return hit["datasets"]
            disk = self._analysis_path(od)
            data = read_json(disk) if disk else None
            if not (isinstance(data, dict) and data.get("key") == json.loads(json.dumps(key))):
                t0 = time.time()
                data = {"key": key, "datasets": {ds: ljeval.analyze_dataset(p, ds) for ds, p in ljeval.dataset_files(od).items()}}
                print(f"analyzed {rid} {ver}: {sum(len(a['rows']) for a in data['datasets'].values())} samples in {time.time() - t0:.1f}s", file=sys.stderr)
                if disk:
                    self._save_json(disk, data)
            self._analysis[mem_key] = data
            while len(self._analysis) > ANALYSIS_MEM:
                self._analysis.popitem(last=False)
            return data["datasets"]

    def _analysis_path(self, od: Path) -> Optional[Path]:
        if not self.cache_path:
            return None
        return self.cache_path.parent / "analysis" / (hashlib.sha1(str(od).encode()).hexdigest()[:16] + ".json")
