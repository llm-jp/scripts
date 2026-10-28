"""llm-jp-eval 以外の評価 (swallow・llm-jp-judge・safety-eval) のスコアの要約。

  <run>/swallow/results/result.json                          {"scores": {タスク: 値}}。-1.0 は未実行
  <run>/llm-jp-judge/evaluation/score_table.json             [{"<bench>:<観点>": 値, generation_model, evaluation_model}]
  <run>/llm-jp-judge/evaluation/evaluate_error_rate_table.json  ジャッジ出力の解析失敗率
  <run>/safety-eval/evaluate_count/<model>_evaluate/<bench>_base_evaluated_<日付>.json
"""
from __future__ import annotations

import re
from pathlib import Path

from .util import mean, mtime, num, read_json

# swallow の主要タスク (平均に使う)。値はどれも 0〜1
SWALLOW_EN = ["gsm8k", "squad2", "triviaqa", "hellaswag", "openbookqa", "xwinograd_en", "bbh_cot", "mmlu", "math_500",
              "gpqa_main_meta_llama3_cot_zeroshot", "humaneval@1"]
SWALLOW_JA = ["jcommonsenseqa", "jemhopqa", "niilc", "jsquad", "XLSUM_ja_1shot", "MATH (mgsm_ja)", "jmmlu", "jhumaneval@1",
              "jamp (NLI)", "janli (NLI)", "jnli", "jsem", "jsick (NLI)"]


def swallow(run_dir: Path) -> dict | None:
    p = run_dir / "swallow" / "results" / "result.json"
    if not (run_dir / "swallow").is_dir():
        return None
    d = read_json(p)
    if not isinstance(d, dict):
        return {"status": "missing", "result_mtime": 0.0}
    scores = {k: v for k, v in (d.get("scores") or {}).items() if num(v) is not None and v != -1.0}
    return {
        "status": "done" if scores else "broken", "result_mtime": mtime(p), "scores": scores,
        "avg": {"EN": mean(scores.get(k) for k in SWALLOW_EN), "JA": mean(scores.get(k) for k in SWALLOW_JA)},
        "n_en": sum(1 for k in SWALLOW_EN if k in scores), "n_ja": sum(1 for k in SWALLOW_JA if k in scores),
    }


def judge(run_dir: Path) -> dict | None:
    base = run_dir / "llm-jp-judge"
    if not base.is_dir():
        return None
    p = base / "evaluation" / "score_table.json"
    rows = read_json(p)
    if not isinstance(rows, list) or not rows:
        return {"status": "missing", "result_mtime": 0.0}
    row = rows[0]
    scores: dict = {}
    for k, v in row.items():
        if ":" in k and num(v) is not None:
            bench, aspect = k.split(":", 1)
            scores.setdefault(bench, {})[aspect] = v
    err = read_json(base / "evaluation" / "evaluate_error_rate_table.json")
    errors: dict = {}
    for k, v in ((err[0] if isinstance(err, list) and err else {}) or {}).items():
        if ":" in k and num(v) is not None:
            bench, kind = k.split(":", 1)
            errors.setdefault(bench, {})[kind] = v
    return {"status": "done", "result_mtime": mtime(p), "judge_model": row.get("evaluation_model"), "scores": scores, "errors": errors}


# safety-eval のベンチマークごとの代表値 (UI の表示名・向き・範囲)
SAFETY_HEADLINE = {
    "llm_as_a_judge": ("attempt_avg", "ジャッジ平均"),
    "jtruthfulqa": ("truthful_rate", "真実性の率"),
    "jbbq": ("acc", "正解率"),
}


def safety(run_dir: Path) -> dict | None:
    base = run_dir / "safety-eval" / "evaluate_count"
    if not (run_dir / "safety-eval").is_dir():
        return None
    files = sorted(base.glob("*_evaluate/*_evaluated_*.json")) if base.is_dir() else []
    benches = {}
    last = 0.0
    for f in files:
        m = re.match(r"(.+?)_[a-z]+_evaluated_\d{4}-\d{2}-\d{2}\.json$", f.name)
        name = m.group(1) if m else f.stem
        d = read_json(f)
        if not isinstance(d, dict):
            continue
        last = max(last, mtime(f))
        kind = d.get("task_type") or "?"
        metrics = {}
        if kind == "llm_as_a_judge":
            metrics = {"attempt_avg": d.get("attempt_avg"), "prompt_avg": d.get("prompt_avg"), "valid_count": d.get("valid_count")}
        elif kind == "jtruthfulqa":
            ov = d.get("overall") or {}
            metrics = {"truthful_rate": ov.get("truthful_rate"), "valid_count": ov.get("valid_count"), "invalid_count": ov.get("invalid_count")}
        elif kind == "jbbq":
            metrics = {k: v for k, v in d.items() if num(v) is not None}
            acc = next((v for k, v in metrics.items() if k.endswith("_0shot_acc")), None)
            if acc is not None:
                metrics["acc"] = acc
        else:
            metrics = {k: v for k, v in d.items() if num(v) is not None}
        key = SAFETY_HEADLINE.get(kind, ("", ""))[0]
        benches[name] = {"kind": kind, "metrics": metrics, "headline": metrics.get(key), "headline_label": SAFETY_HEADLINE.get(kind, ("", ""))[1]}
    return {"status": "done" if benches else "missing", "result_mtime": last, "benchmarks": benches}
