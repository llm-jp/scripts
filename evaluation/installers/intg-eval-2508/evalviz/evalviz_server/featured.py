"""掲載モデルの設定 (トップページのリーダーボードに載せるモデルと、比較の標準条件)。

設定ファイル (JSON。既定は run.sh の EVALVIZ_FEATURED) の形:
  {
    "title": "LLM-jp モデルの評価結果",                       省略可
    "standard": {"llm-jp-eval": {"version": "v2.1.5", "max_num_samples": 100}},
    "models": [
      {"name": "llm-jp-4-8b-thinking",                      表示名 (必須)
       "runs": ["/groups/.../<qsub.py の output_dir>", ...],  評価の出力ディレクトリ (必須。評価ごとに別の投入でもよい)
       "family": "LLM-jp", "params": "8B", "type": "thinking",  表示用 (省略可)
       "baseline": true,                                     他組織のモデル (比較対象として色を変える)
       "note": "reasoning_effort=medium"}                    補足 (省略可)
    ]
  }
キーが "_" で始まる項目は無視する (JSON にはコメントが無いので、説明を "_comment" に書く)。

モデルの評価ごとの値は、runs を先頭から見て最初に結果のある実行から取る。llm-jp-eval だけは標準条件
(バージョンとデータセットあたりのサンプル数) に合う実行に限る。条件の違う結果はリーダーボードに載せない。
ファイルは mtime が変わったら読み直すので、編集はサーバの再起動なしに反映される。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from .util import mtime

COMPONENTS = ("llm-jp-eval", "swallow", "llm-jp-judge", "safety-eval")


class Featured:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self.lock = threading.Lock()
        self._mtime = -1.0
        self.config: dict = {}
        self.error = ""

    def load(self) -> dict:
        """設定を返す (変わっていれば読み直す)。読めなければ空の設定と error"""
        with self.lock:
            m = mtime(self.path) if self.path else 0.0
            if m == self._mtime:
                return self.config
            self._mtime = m
            self.config, self.error = {}, ""
            if not self.path:
                self.error = "設定ファイルが指定されていない (EVALVIZ_FEATURED)"
            elif not m:
                self.error = f"設定ファイルが無い: {self.path}"
            else:
                try:
                    with open(self.path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    if not isinstance(cfg, dict) or not isinstance(cfg.get("models"), list):
                        raise ValueError('"models" (配列) が無い')
                    self.config = cfg
                except Exception as ex:
                    self.error = f"設定ファイルを読めない ({self.path}): {type(ex).__name__}: {ex}"
            return self.config

    def run_dirs(self) -> list:
        """設定に書かれた実行ディレクトリ (出力ルートの外にあってもインデックスに載せる)"""
        out = []
        for m in self.load().get("models") or []:
            for r in (m.get("runs") or []) if isinstance(m, dict) else []:
                if isinstance(r, str):
                    out.append(str(Path(r)))
        return out


def _ok_ljeval(c: dict, std: dict) -> bool:
    if c.get("status") != "done":
        return False
    n = std.get("max_num_samples")
    return n is None or c.get("n_samples") == n


def resolve(cfg: dict, runs_by_dir: dict) -> dict:
    """設定のモデルを実行に結び付ける。{title, standard, models: [...], problems: [...]}"""
    std = ((cfg.get("standard") or {}).get("llm-jp-eval")) or {}
    ver = std.get("version")
    models, problems = [], []
    for i, m in enumerate(cfg.get("models") or []):
        if not isinstance(m, dict) or not m.get("name"):
            problems.append(f"models[{i}]: name が無い")
            continue
        name = m["name"]
        runs = []
        for r in m.get("runs") or []:
            run = runs_by_dir.get(str(Path(r)))
            if run is None:
                problems.append(f"{name}: 実行が見つからない ({r})")
            else:
                runs.append(run)
        if not runs:
            problems.append(f"{name}: runs に実行が無い")
        src: dict = {}
        for comp in COMPONENTS:
            for run in runs:
                c = (run.get("components") or {}).get(comp)
                if comp == "llm-jp-eval":
                    if ver and _ok_ljeval((c or {}).get(ver) or {}, std):
                        src[comp] = run["id"]
                        break
                elif c and c.get("status") == "done":
                    src[comp] = run["id"]
                    break
        if runs and ver and "llm-jp-eval" not in src:
            have = sorted({f"{v} ({c.get('n_samples')} 件)" for run in runs for v, c in ((run.get("components") or {}).get("llm-jp-eval") or {}).items()
                           if c.get("status") == "done"})
            problems.append(f"{name}: 標準条件 (llm-jp-eval {ver}" + (f"・{std['max_num_samples']} 件" if std.get("max_num_samples") is not None else "")
                            + ") の結果が無い" + (f"。ある結果: {', '.join(have)}" if have else ""))
        models.append({
            "name": name,
            "family": m.get("family"), "params": m.get("params"), "type": m.get("type"),
            "baseline": bool(m.get("baseline")), "note": m.get("note"),
            "runs": [r["id"] for r in runs], "source": src,
        })
    return {"title": cfg.get("title"), "standard": cfg.get("standard") or {}, "models": models, "problems": problems}
