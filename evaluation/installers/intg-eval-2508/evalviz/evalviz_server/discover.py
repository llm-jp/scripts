"""出力ルートから実行 (= qsub.py / run_eval.sh の出力ディレクトリ 1 つ = 1 モデル × 1 回の投入) を列挙する。

実行の目印は、llm-jp-eval/・llm-jp-eval_v*/・swallow/・llm-jp-judge/・safety-eval/ のどれか、または qsub.py が書く
config.json (model_name_or_path を含む)。ルートから MAX_DEPTH 段まで潜り、実行を見つけたらその下には潜らない。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .util import read_json

MAX_DEPTH = 4
COMPONENT_DIRS = ("llm-jp-eval", "swallow", "llm-jp-judge", "safety-eval")
# 実行を探すときに潜らないディレクトリ (学習の出力・キャッシュ・エージェント評価のジョブ等)
SKIP_DIRS = {"logs", "models", "pbs-logs", "checkpoints", "checkpoints_hf", "cache", ".cache", "environment", "node_modules",
             ".git", "offline", "datasets", "tmp", "wandb", "__pycache__"}
_STEP = re.compile(r"(?:global_step|step|iter|ckpt|checkpoint)[-_]?0*(\d+)", re.IGNORECASE)
# モデルパスの末尾にあって、それだけでは何のモデルか分からない名前
_GENERIC = re.compile(r"^((global_step|step|iter|checkpoint|ckpt)[-_]?\d+.*|final_hf|hf|converted|checkpoints?_hf|latest|model)$", re.IGNORECASE)


def is_run_dir(d: Path, names: set) -> bool:
    if any(n in names for n in COMPONENT_DIRS) or any(n.startswith("llm-jp-eval_v") for n in names):
        return True
    if "config.json" in names and "qsub.sh" in names:
        return True
    return False


def discover_runs(roots: list, extra_dirs: tuple = ()) -> list:
    """roots の下の実行と、extra_dirs (掲載モデルの設定に書かれた実行。出力ルートの外でもよい) を列挙する"""
    runs, seen = [], set()

    def add(root: Path, d: Path):
        run = describe_run(root, d)
        rid, n = run["id"], 2
        while run["id"] in seen:
            run["id"] = f"{rid}~{n}"
            n += 1
        seen.add(run["id"])
        runs.append(run)

    for root in roots:
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                entries = [e for e in os.scandir(d) if e.is_dir(follow_symlinks=False)]
            except OSError:
                continue
            names = {e.name for e in entries}
            if depth > 0 and is_run_dir(d, names):
                add(root, d)
                continue
            if depth >= MAX_DEPTH:
                continue
            for e in sorted(entries, key=lambda e: e.name, reverse=True):
                if e.name not in SKIP_DIRS and not e.name.startswith((".", "failed-attempt")) and not e.name.startswith("prompts"):
                    stack.append((Path(e.path), depth + 1))
    found = {r["dir"] for r in runs}
    for x in extra_dirs:
        d = Path(x)
        if str(d) not in found and d.is_dir():
            add(d.parent, d)
            found.add(str(d))
    return runs


def model_label(path: str) -> str:
    """モデルの表示名。Hub の ID はそのまま、パスは末尾の名前。末尾が汎用的な名前 (iter_0024000・final_hf 等) なら、
    意味のある最も近い親を前に付ける (ratio0.8_lr1e-4/iter_0024000)"""
    if not path:
        return "?"
    if not path.startswith("/"):
        return path
    parts = [p for p in path.rstrip("/").split("/") if p]
    leaf = parts[-1]
    if not _GENERIC.match(leaf):
        return leaf
    anc = next((p for p in reversed(parts[:-1]) if not _GENERIC.match(p) and p not in ("tasks", "models", "hf_models")), "")
    return f"{anc}/{leaf}" if anc else leaf


def root_label(root: Path) -> str:
    s = str(root)
    m = re.match(r"^/groups/[^/]+/experiments/(.+)$", s)
    return m.group(1) if m else s


def describe_run(root: Path, d: Path) -> dict:
    cfg = read_json(d / "config.json") or {}
    if not isinstance(cfg, dict) or "model_name_or_path" not in cfg:
        cfg = {}
    rel = d.relative_to(root)
    model_path = cfg.get("model_name_or_path") or ""
    if not model_path:
        # config.json の無い古い出力: llm-jp-eval の推論ディレクトリ名 (<model を -- 区切り>::vllm...) から推定
        od = next((p for pat in ("llm-jp-eval/*/offline/*::*", "llm-jp-eval_v*/offline/*::*", "llm-jp-eval/offline/*::*") for p in sorted(d.glob(pat))), None)
        if od:
            model_path = od.name.split("::")[0].replace("--", "/")
            model_path = ("/" + model_path) if model_path.startswith("groups/") else model_path
    m = _STEP.search(d.name) or _STEP.search(model_path.rsplit("/", 1)[-1] if model_path else "")
    step = int(m.group(1)) if m else None
    series = _STEP.sub(lambda mm: mm.group(0)[: mm.start(1) - mm.start(0)] + "*", d.name) if _STEP.search(d.name) else None
    settings = {
        "apply_chat_template": cfg.get("apply_chat_template"),
        "reasoning_parser": cfg.get("reasoning_parser"),
        "chat_template_args": " ".join(cfg.get("chat_template_args") or []) or None,
        "basemodel": cfg.get("basemodel"),
        "max_num_samples": cfg.get("llm_jp_eval_max_num_samples"),
        "max_tokens": cfg.get("llm_jp_eval_max_tokens"),
        "max_model_len": cfg.get("max_model_len"),
        "swallow_version": None if cfg.get("disable_swallow") else cfg.get("swallow_version"),
        "judge_model": cfg.get("judge_model") if cfg.get("llm_jp_judge") else None,
        "vllm_serve": cfg.get("vllm_serve"),
    }
    return {
        "id": str(rel).replace("/", ":"),
        "name": str(rel),
        "group": root_label(root) + ("/" + str(rel.parent) if str(rel.parent) != "." else ""),
        "root": str(root),
        "dir": str(d),
        "model": model_label(model_path) if model_path else d.name,
        "model_path": model_path,
        "step": step,
        "series": series,
        "settings": {k: v for k, v in settings.items() if v not in (None, False, "")},
        "config": cfg,
    }
