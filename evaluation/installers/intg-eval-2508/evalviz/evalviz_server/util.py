"""ファイル読み込みと小物。どれも失敗時に例外を投げず空値を返す (書き込み中・壊れたファイルを一覧の 1 行の欠けで済ませる)。"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# 巨大な result.json (全件の records を含む v2.1.0 等) は全体を読まずに先頭のキーだけ拾う
_BIG_JSON = 64 * 1024 * 1024
_HEAD_BYTES = 32 * 1024 * 1024


def read_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_json_keys(p: Path, keys: tuple) -> dict:
    """トップレベルの keys だけを返す。大きいファイルは先頭 _HEAD_BYTES から `"key": {` を探して部分的に読む
    (llm-jp-eval v2 の result.json は config, metadata, evaluation, records, scores の順、v1.4.1 は scores が先頭)"""
    try:
        size = p.stat().st_size
    except OSError:
        return {}
    if size <= _BIG_JSON:
        d = read_json(p)
        return {k: d[k] for k in keys if isinstance(d, dict) and k in d}
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            head = f.read(_HEAD_BYTES)
    except OSError:
        return {}
    dec = json.JSONDecoder()
    out = {}
    for k in keys:
        for m in re.finditer(r'"%s"\s*:\s*' % re.escape(k), head):
            try:
                v, _ = dec.raw_decode(head, m.end())
            except ValueError:
                continue
            out[k] = v
            break
    return out


def read_text(p: Path, limit: Optional[int] = None) -> str:
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            s = f.read()
    except Exception:
        return ""
    if limit and len(s) > limit:
        return s[: limit // 2] + f"\n\n... [{len(s) - limit} 文字省略] ...\n\n" + s[-limit // 2:]
    return s


def tail_text(p: Path, n_bytes: int = 4000) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def iso(ts: float) -> Optional[str]:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None


def num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def mean(xs) -> Optional[float]:
    v = [x for x in xs if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None
