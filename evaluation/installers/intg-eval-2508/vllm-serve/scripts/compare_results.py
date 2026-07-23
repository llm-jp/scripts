#!/usr/bin/env python3
"""Compare two intg-eval output directories (scores and wall-clock time).

Intended for validating the vllm-serve mode against offline runs, but works
for any two runs produced by qsub.py / sbatch.py / run_eval_serve.sh:

    python3 compare_results.py OFFLINE_RUN_DIR SERVE_RUN_DIR

Compares, for whatever is present in both directories:
  - llm-jp-eval:   <run>/llm-jp-eval*/<version>/results/result.json ("scores")
  - swallow (agg): <run>/swallow/results/result.json ("scores"; -1.0 = not run)
  - swallow (raw): every lm_eval results.json under <run>/swallow/ (per-task
                   metrics; finer grained than the aggregate, which stays -1.0
                   unless the full task set ran)
  - timing:        "[intg-eval] job start/end:" markers in logs/sbatch.out
                   (emitted by sbatch.py), falling back to logs/* mtimes;
                   per-phase durations are estimated from phase-log mtimes

Exit code is 0 even when scores differ; this is a reporting tool.
"""
import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

EPS_DEFAULT = 1e-9


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def compare_score_dicts(name, a, b, eps, label_a, label_b):
    """Print a comparison of two flat {metric: number} dicts."""
    keys_a, keys_b = set(a), set(b)
    common = sorted(keys_a & keys_b)
    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)
    diffs = []
    for k in common:
        va, vb = a[k], b[k]
        try:
            same = abs(float(va) - float(vb)) <= eps
        except (TypeError, ValueError):
            same = va == vb
        if not same:
            diffs.append((k, va, vb))

    print(f"\n## {name}: {len(common)} common metrics, "
          f"{len(common) - len(diffs)} equal (eps={eps:g}), {len(diffs)} differ")
    if diffs:
        w = max(len(k) for k, _, _ in diffs)
        print(f"   {'metric'.ljust(w)}  {label_a:>12}  {label_b:>12}  {'delta':>12}")
        for k, va, vb in diffs:
            try:
                delta = f"{float(vb) - float(va):+.6g}"
            except (TypeError, ValueError):
                delta = "n/a"
            print(f"   {k.ljust(w)}  {fmt(va):>12}  {fmt(vb):>12}  {delta:>12}")
    for label, only in ((label_a, only_a), (label_b, only_b)):
        if only:
            print(f"   only in {label}: {', '.join(only)}")
    return len(diffs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def llm_jp_eval_results(run_dir):
    """{version: scores_dict} from llm-jp-eval result.json files."""
    out = {}
    for p in sorted(run_dir.glob("llm-jp-eval*/**/results/result.json")):
        m = re.search(r"v\d+\.\d+\.\d+", str(p.relative_to(run_dir)))
        version = m.group(0) if m else str(p.parent.parent.relative_to(run_dir))
        out[version] = load_json(p).get("scores", {})
    return out


def swallow_aggregate(run_dir):
    p = run_dir / "swallow" / "results" / "result.json"
    if not p.is_file():
        return {}
    scores = load_json(p).get("scores", {})
    return {k: v for k, v in scores.items() if v != -1.0}


def swallow_per_task(run_dir):
    """{task: {metric: value}} merged from all lm_eval results.json files."""
    out = {}
    swallow = run_dir / "swallow"
    if not swallow.is_dir():
        return out
    for p in sorted(swallow.rglob("results.json")):
        try:
            results = load_json(p).get("results", {})
        except (json.JSONDecodeError, OSError):
            continue
        for task, metrics in results.items():
            for metric, value in metrics.items():
                if metric == "alias" or "stderr" in metric:
                    continue
                if isinstance(value, (int, float)):
                    out.setdefault(task, {})[metric] = value
    return out


def parse_job_markers(run_dir):
    """(start, end) datetimes from sbatch.out markers, or (None, None)."""
    start = end = None
    out = run_dir / "logs" / "sbatch.out"
    if out.is_file():
        for line in out.read_text(errors="replace").splitlines():
            m = re.search(r"\[intg-eval\] job (start|end): (\S+)", line)
            if m:
                try:
                    ts = datetime.datetime.fromisoformat(m.group(2))
                except ValueError:
                    continue
                if m.group(1) == "start":
                    start = ts
                else:
                    end = ts
    return start, end


def report_timing(run_dir, label):
    print(f"\n## timing: {label} ({run_dir})")
    start, end = parse_job_markers(run_dir)
    logs = sorted((run_dir / "logs").glob("*"), key=lambda p: p.stat().st_mtime) \
        if (run_dir / "logs").is_dir() else []
    if start and end:
        print(f"   job start: {start}   end: {end}   elapsed: {end - start}")
    elif logs:
        mtimes = [p.stat().st_mtime for p in logs]
        span = datetime.timedelta(seconds=round(max(mtimes) - min(mtimes)))
        print(f"   no job markers in sbatch.out; log mtime span >= {span} "
              "(lower bound, excludes time before the first log write)")
    else:
        print("   no logs/ directory")
        return end - start if start and end else None
    # Phase-completion times: each phase writes its own log, so successive
    # mtimes approximate phase boundaries.
    prev = start.timestamp() if start else None
    for p in logs:
        mt = p.stat().st_mtime
        t = datetime.datetime.fromtimestamp(mt).strftime("%m-%d %H:%M:%S")
        dur = f"  (+{datetime.timedelta(seconds=round(mt - prev))})" if prev else ""
        print(f"   {t}{dur}  {p.name}")
        prev = mt
    return end - start if start and end else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--label-a", default=None, help="display label (default: dir name)")
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--epsilon", type=float, default=EPS_DEFAULT,
                    help="absolute tolerance for 'equal' (default: %(default)g)")
    args = ap.parse_args()

    la = args.label_a or args.run_a.name
    lb = args.label_b or args.run_b.name
    print(f"# intg-eval comparison\n#   A ({la}): {args.run_a}\n#   B ({lb}): {args.run_b}")

    total_diffs = compared = 0

    lje_a, lje_b = llm_jp_eval_results(args.run_a), llm_jp_eval_results(args.run_b)
    for version in sorted(set(lje_a) & set(lje_b)):
        compared += 1
        total_diffs += compare_score_dicts(
            f"llm-jp-eval {version}", lje_a[version], lje_b[version],
            args.epsilon, la, lb)
    for version in sorted(set(lje_a) ^ set(lje_b)):
        side = la if version in lje_a else lb
        print(f"\n## llm-jp-eval {version}: only in {side}; skipped")

    agg_a, agg_b = swallow_aggregate(args.run_a), swallow_aggregate(args.run_b)
    if agg_a or agg_b:
        compared += 1
        total_diffs += compare_score_dicts(
            "swallow (aggregate result.json, -1.0 excluded)",
            agg_a, agg_b, args.epsilon, la, lb)

    task_a, task_b = swallow_per_task(args.run_a), swallow_per_task(args.run_b)
    if task_a or task_b:
        flat_a = {f"{t}/{m}": v for t, ms in task_a.items() for m, v in ms.items()}
        flat_b = {f"{t}/{m}": v for t, ms in task_b.items() for m, v in ms.items()}
        compared += 1
        total_diffs += compare_score_dicts(
            "swallow (per-task lm_eval results.json)",
            flat_a, flat_b, args.epsilon, la, lb)

    if not compared:
        print("\nWARNING: found nothing comparable in the two directories.",
              file=sys.stderr)

    ea = report_timing(args.run_a, la)
    eb = report_timing(args.run_b, lb)
    if ea and eb:
        ratio = ea.total_seconds() / eb.total_seconds() if eb.total_seconds() else float("inf")
        print(f"\n## elapsed: {la}={ea}  {lb}={eb}  (A/B ratio: {ratio:.2f}x)")

    print(f"\n# summary: {total_diffs} differing metrics across {compared} comparison(s)")


if __name__ == "__main__":
    main()
