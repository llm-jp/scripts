"""llm-jp-eval の結果。スコアの要約 (result.json) と、推論結果 (offline/*.eval-generated.json) からのサンプル単位の分析。

レイアウト (qsub.py):
  <run>/llm-jp-eval/<ver>/results/result.json                 スコア (v2 は evaluation.scores / lang_scores、v1.4.1 は scores)
  <run>/llm-jp-eval/<ver>/offline/<run_name>/<ds>.eval-generated.json   サンプルごとの生成 (v2 のみ。gold・reasoning_content 付き)
  旧レイアウト (--legacy-output): <run>/llm-jp-eval/ (v1.4.1)、<run>/llm-jp-eval_<ver>/

サンプルの正誤は result.json に残らない (records は既定で各データセット上位 5 件) ので、llm-jp-eval の回答抽出
(answer_parser.py・BBH の extract_answer) を移植して生成から pred を作り直す。exact_match 系は llm-jp-eval と一致する
(check コマンドで突き合わせる)。数式の同値判定などは近似 (approx) で、BLEU・F1 等の連続値の指標は正誤を出さない。
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zlib
from pathlib import Path
from typing import Optional

from .util import mean, mtime, num, read_json, read_json_keys

# 英語のデータセット (llm-jp-eval v2.1.5 の jaster/*.py で language = "en" のもの)。残りは日本語
EN_DATASETS = {
    "aime2024", "aime2025", "bigbenchhard_direct", "bigbenchhard_cot", "drop", "gsm8k", "gpqa_diamond_en", "gpqa_extended_en",
    "gpqa_main_en", "hle", "mif_eval_en", "mmlu_en", "mmlu_prox_en", "openbookqa", "polymath-en", "structeval", "triviaqa",
    "winogrande_xl",
}
# 生成の目標言語が英語・コードの日本語データセット (英語で答えるのが正しい)
NON_JA_TARGET = {"alt-j-to-e", "wikicorpus-j-to-e", "mbpp", "jhumaneval"}

# サンプルごとに正誤を決められる指標。値は判定方法 (exact = NFKC 後の完全一致、math = 数式の同値の近似)
BINARY_METRICS = {
    "exact_match": "exact", "exact_match_figure": "exact", "triviaqa_exact_match": "exact", "hle_exact_match": "exact",
    "mathematical_equivalence": "math", "polymath_weighted_accuracy": "math",
}
# result.json の "<ds>_<metric>" を分けるための指標名 (長いものから当てる)
KNOWN_METRICS = sorted({
    "ool", "exact_match", "exact_match_figure", "char_f1", "set_f1", "bleu_en", "bleu_ja", "bert_score_en_f1", "bert_score_ja_f1",
    "comet_wmt22", "pearson", "spearman", "mathematical_equivalence", "polymath_weighted_accuracy", "code_exec", "code_exec_sandbox",
    "pylint_check", "mifeval_strict", "mifeval_loose", "rouge1", "rouge2", "rougeLsum", "rouge2_scaling", "drop_f1",
    "triviaqa_exact_match", "triviaqa_f1", "hle_exact_match", "structeval_score", "jfinqa_accuracy",
}, key=len, reverse=True)

# v1.4.1 にはカテゴリの定義が result.json に無い。カテゴリ名のスコアだけ使う
V1_CATEGORIES = ["NLI", "QA", "RC", "MC", "HE", "EL", "FA", "MR", "MT", "CG", "SUM"]

ANALYSIS_VERSION = 4   # サンプル分析の形や判定を変えたら上げる (ディスクキャッシュを無効にする)


def lang_of(ds: str) -> str:
    return "EN" if ds in EN_DATASETS else "JA"


def version_dirs(run_dir: Path) -> dict:
    """{version: dir}。新レイアウト llm-jp-eval/<ver>/ と旧レイアウト llm-jp-eval/ (v1.4.1)・llm-jp-eval_<ver>/"""
    out = {}
    base = run_dir / "llm-jp-eval"
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and re.match(r"v\d", d.name):
                out[d.name] = d
        if "v1.4.1" not in out and (base / "results").is_dir():
            out["v1.4.1"] = base
    for d in sorted(run_dir.glob("llm-jp-eval_v*")):
        if d.is_dir():
            out.setdefault(d.name.split("_", 1)[1], d)
    return out


def split_score_key(k: str) -> tuple:
    for m in KNOWN_METRICS:
        if k.endswith("_" + m):
            return k[: -len(m) - 1], m
    return k, ""


def offline_dir(vdir: Path, cfg: dict) -> Optional[Path]:
    p = cfg.get("inference_result_dir")
    if p and Path(p).is_dir() and str(Path(p)).startswith(str(vdir)):
        return Path(p)
    cands = [d for d in (vdir / "offline").glob("*") if d.is_dir() and any(d.glob("*.eval-generated.json"))] if (vdir / "offline").is_dir() else []
    return max(cands, key=mtime) if cands else None


def summarize(version: str, vdir: Path) -> dict:
    """1 バージョン分のスコアの要約"""
    rpath = vdir / "results" / "result.json"
    out = {"version": version, "dir": str(vdir), "status": "missing", "result_mtime": mtime(rpath)}
    if not rpath.is_file():
        out["has_offline"] = bool(list((vdir / "offline").glob("*/*.eval-generated.json"))[:1]) if (vdir / "offline").is_dir() else False
        return out
    if version.startswith("v1"):
        d = read_json_keys(rpath, ("scores",))
        scores = {k: v for k, v in (d.get("scores") or {}).items() if num(v) is not None}
        if not scores:
            out["status"] = "broken"
            return out
        cats = {c: scores[c] for c in V1_CATEGORIES if c in scores}
        datasets = {}
        for k, v in scores.items():
            if k in cats or k == "AVG":
                continue
            ds, m = split_score_key(k)
            if m and m != "ool":
                datasets.setdefault(ds, {"lang": lang_of(ds), "category": None, "metric": m, "score": v, "ool": None})
        out.update(status="done", avg={"ALL": scores.get("AVG")}, categories={"ALL": cats}, datasets=datasets,
                   n_samples=None, few_shots=None, has_offline=False)
        return out

    d = read_json_keys(rpath, ("config", "evaluation", "scores", "lang_scores"))
    ev = d.get("evaluation") or {}
    scores = ev.get("scores") or d.get("scores") or {}
    lang_scores = ev.get("lang_scores") or d.get("lang_scores") or {}
    cfg = d.get("config") or {}
    if not scores:
        out["status"] = "broken"
        return out
    cat_defs = ((cfg.get("eval_dataset_config") or {}).get("categories")) or {}
    datasets = {}
    for cname, c in cat_defs.items():
        for ds in c.get("datasets") or []:
            metric = (c.get("metrics") or {}).get(ds, c.get("default_metric"))
            if f"{ds}_{metric}" in scores:
                datasets[ds] = {"lang": lang_of(ds), "category": cname, "metric": metric, "score": scores[f"{ds}_{metric}"],
                                "ool": scores.get(f"{ds}_ool")}
    for k, v in scores.items():   # カテゴリに属さないデータセット
        ds, m = split_score_key(k)
        if m and m != "ool" and ds not in datasets and num(v) is not None:
            datasets[ds] = {"lang": lang_of(ds), "category": None, "metric": m, "score": v, "ool": scores.get(f"{ds}_ool")}
    cats = {}
    for lang, cs in lang_scores.items():
        cats[lang] = {k: v for k, v in cs.items() if k != "AVG"}
    avg = {lang: (cs.get("AVG") if num(cs.get("AVG")) is not None else mean(v for k, v in cs.items() if k != "AVG"))
           for lang, cs in lang_scores.items()}
    avg["ALL"] = scores.get("AVG")
    od = offline_dir(vdir, cfg)
    out.update(
        status="done", avg=avg, categories=cats, datasets=datasets,
        category_desc={k: c.get("description") for k, c in cat_defs.items()},
        n_samples=cfg.get("max_num_samples"), few_shots=cfg.get("num_few_shots"),
        infer_s=num((ev.get("time_profile") or {}).get("(infer)")),
        run_name=(cfg.get("inference_result_config") or {}).get("run_name") or cfg.get("_resolved_run_name"),
        has_offline=od is not None,
    )
    return out


def raw_scores(vdir: Path) -> dict:
    d = read_json_keys(vdir / "results" / "result.json", ("evaluation", "scores"))
    return (d.get("evaluation") or {}).get("scores") or d.get("scores") or {}


# ---------------------------------------------------------------- 回答抽出 (llm-jp-eval answer_parser.py の移植)
_PATTERNS = {
    "answer_tags_jp": r"<answer>(.*?)</answer>", "answer_tags_en": r"<answer>(.*?)</answer>",
    "choice_only_jp": r"(?s)^(.*?)(?=\n|\Z)", "choice_only_en": r"(?s)^(.*?)(?=\n|\Z)",
    "json_output_jp": r"```json\s*\n?(.*?)\n?```", "json_output_en": r"```json\s*\n?(.*?)\n?```",
    "code_output_jp": r"```(?:\w+)?\s*\n?(.*?)\n?```", "code_output_en": r"```(?:\w+)?\s*\n?(.*?)\n?```",
    "latex_boxed_jp": r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", "latex_boxed_en": r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
}
_BBH_EVAL = {"binary": r"\b(True|False|Yes|No|yes|no)\b", "choice": r"(\([A-R]\))", "number": r"(\d+)", "valid_invalid": r"\b(valid|invalid)\b"}


def _bbh_extract(text: str, meta: dict, ja: bool) -> str:
    if ja and meta.get("prompt_type") == "cot":
        m = re.search(r"答えは「(.+?)」", text)
        if m:
            return m.group(1)
    ptype, etype = meta.get("prompt_type", "direct"), meta.get("evaluation_type", "word")
    if ptype == "cot":
        m = re.search(r"So the answer is (.+?)(?:\.|$)", text, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if etype == "choice":
                c = re.search(r"\([A-R]\)|([A-R])", ans)
                if c:
                    return c.group(0) if c.group(0).startswith("(") else f"({c.group(0)})"
            return ans
    if etype == "exact":
        return text.strip()
    pat = _BBH_EVAL.get(etype)
    if pat:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if ms:
            m = ms[-1]
            return m.group(1).strip() if m.groups() else m.group(0).strip()
    words = text.strip().split()
    return words[0] if words else ""


def extract(text: str, pattern_id: str, custom_regex: str, ds: str, meta: dict) -> tuple:
    """(pred, 一致した区間の終端 or None)。pred は llm-jp-eval と同じく NFKC 正規化して strip する"""
    end = None
    if pattern_id == "dataset":
        raw = _bbh_extract(text, meta or {}, ds.startswith("bigbenchhard_ja")) if ds.startswith("bigbenchhard") else ""
    else:
        regex = _PATTERNS.get(pattern_id) or custom_regex or r"(?s)^(.*?)(?=\n\n|\Z)"
        m = re.search(regex, text, re.DOTALL)
        raw = "".join(g for g in m.groups() if g).strip() if m else ""
        end = m.end() if m else None
    return unicodedata.normalize("NFKC", raw).strip(), end


_MATH_STRIP = re.compile(r"\\[!,;: ]|\\left|\\right|\\displaystyle|\\?\$|\\?%|\^\{?\\circ\}?|\{,\}|\s")
_LEADING_NUM = re.compile(r"^-?\d[\d,]*(?:\.\d+)?")


def _math_norm(s: str) -> str:
    """数式の比較用の正規化 (math_verify の近似)。単位の \\text{...}・$・%・度記号・桁区切りを落とす"""
    s = unicodedata.normalize("NFKC", s or "").strip()
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|mbox)\{[^}]*\}", "", s)
    s = _MATH_STRIP.sub("", s).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.rstrip(".")


def _to_float(s: str):
    m = _LEADING_NUM.match(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _math_equal(pred: str, gold: str) -> bool:
    a, b = _math_norm(pred), _math_norm(gold)
    if a == b and a:
        return True
    m = re.fullmatch(r"(-?)\\frac\{(\d+)\}\{(\d+)\}", a)
    x = (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3)) if m and int(m.group(3)) else _to_float(a)
    y = _to_float(b)
    # 先頭の数値で比べる (「82,285百万円」と「82,285」など)。数値の後に別の式が続く場合は一致させない
    if x is None or y is None or (m is None and re.search(r"[\d=+*/^]", a[len(_LEADING_NUM.match(a).group(0)):])):
        return False
    return abs(x - y) <= 1e-6 * max(1.0, abs(y))


# ---------------------------------------------------------------- 誤りの分類
# 状態 (排他): correct / wrong (抽出でき、選択肢内だが不正解) / out_of_label (選択肢外) / unparsed (回答を抽出できない) /
#              empty (最終出力が空) / scored (連続値の指標のみで正誤なし)
# フラグ (重複可): 下の FLAGS
# 簡体字にしかない字 (日本語の字体と異なるもの)。万・与・参・国・没 等の日本語でも使う字は入れない
_ZH_ONLY = set("们这么说吗呢还时让从对为实种现进应过问题给经关头动发样"
               "门见长东车书产务办华单卖历压县变员团图园场处备复够")
_SPECIAL = re.compile(r"<\|[A-Za-z_]{1,30}\|>|^\s*(analysis|assistantfinal)\b|assistantfinal")
_KANA = re.compile(r"[\u3040-\u30ff]")
_OTHER_SCRIPT = re.compile(r"[\u0370-\u03ff\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u0900-\u0dff\u0e00-\u0e7f\u10a0-\u10ff\u1100-\u11ff\uac00-\ud7af]")


def _repetitive(s: str) -> bool:
    """末尾 4000 バイトの圧縮率が極端に小さい (同じ文字列の反復で生成が終わらない)。実データで正常な出力は 0.15 未満にならない"""
    if len(s) < 1000:
        return False
    tail = s[-4000:].encode("utf-8")
    return len(zlib.compress(tail)) / len(tail) < 0.15


def flags_of(ds: str, gen: str, reas: str, pattern_id: str, match_end: Optional[int]) -> list:
    fl = []
    if not gen.strip() and reas.strip():
        fl.append("reasoning_cut")
    if _repetitive(gen) or _repetitive(reas):
        fl.append("repetition")
    if sum(1 for ch in gen + reas[-2000:] if ch in _ZH_ONLY) >= 3:
        fl.append("zh_mix")
    if lang_of(ds) == "EN" and len(_KANA.findall(gen)) >= 5:
        fl.append("ja_in_en")
    if len(_OTHER_SCRIPT.findall(gen)) >= 10:
        fl.append("other_script")
    if _SPECIAL.search(gen):
        fl.append("special_token")
    # 回答のあとも生成が続く (ベースモデルが次の few-shot 例を書き始める等)。抽出が先頭区切りのパターンに限る
    if match_end is not None and pattern_id in ("answer_tags_jp", "answer_tags_en", "choice_only_jp", "choice_only_en", "custom"):
        if len(re.sub(r"\s", "", gen[match_end:])) > 30:
            fl.append("continued")
    return fl


def sample_key(s: dict) -> str:
    return hashlib.sha1((str(s.get("input")) + "\0" + str(s.get("gold"))).encode("utf-8")).hexdigest()[:12]


def analyze_dataset(path: Path, ds: str) -> dict:
    d = read_json(path) or {}
    metrics = d.get("metrics") or []
    labels = [unicodedata.normalize("NFKC", x) for x in (d.get("label_list") or [])]
    pattern = d.get("answer_pattern_id") or "custom"
    custom = d.get("answer_extract_pattern") or ""
    judge = next((BINARY_METRICS[m] for m in metrics if m in BINARY_METRICS), None)
    rows, seen = [], {}
    for i, s in enumerate(d.get("samples") or []):
        gen, reas = s.get("generated") or "", s.get("reasoning_content") or ""
        pred, end = extract(gen, pattern, custom, ds, s.get("metadata") or {})
        gold = unicodedata.normalize("NFKC", str(s.get("gold", "")))
        if not gen.strip():
            st = "empty"
        elif not pred:
            st = "unparsed"
        elif labels and pred not in labels:
            st = "out_of_label"
        elif judge == "exact":
            st = "correct" if pred == gold else "wrong"
        elif judge == "math":
            st = "correct" if _math_equal(pred, gold) else "wrong"
        else:
            st = "scored"
        k = sample_key(s)
        if k in seen:   # 入力と正解が同じサンプルが複数ある
            seen[k] += 1
            k = f"{k}~{seen[k]}"
        else:
            seen[k] = 1
        rows.append({"k": k, "i": i, "st": st, "p": pred[:200], "g": gold[:200], "fl": flags_of(ds, gen, reas, pattern, end),
                     "ol": len(gen), "rl": len(reas), "in": str(s.get("input", ""))[:160]})
    return {"dataset": ds, "lang": lang_of(ds), "metrics": metrics, "pattern": pattern, "labels": labels,
            "judge": judge, "approx": judge is not None and not (judge == "exact" and "exact_match" in metrics),
            "rows": rows}


def dataset_files(od: Path) -> dict:
    return {p.name[: -len(".eval-generated.json")]: p for p in sorted(od.glob("*.eval-generated.json"))}


def analysis_key(od: Path) -> list:
    return [ANALYSIS_VERSION] + [[p.name, mtime(p)] for p in sorted(od.glob("*.eval-generated.json"))]


def sample_detail(od: Path, ds: str, key: str) -> Optional[dict]:
    p = dataset_files(od).get(ds)
    if not p:
        return None
    d = read_json(p) or {}
    seen = {}
    for i, s in enumerate(d.get("samples") or []):
        k = sample_key(s)
        if k in seen:
            seen[k] += 1
            k = f"{k}~{seen[k]}"
        else:
            seen[k] = 1
        if k == key:
            return {"index": i, "input": s.get("input"), "prompt": s.get("prompt"), "system_prompt": s.get("system_prompt"),
                    "gold": s.get("gold"), "generated": s.get("generated"), "reasoning": s.get("reasoning_content"),
                    "metadata": s.get("metadata"), "instruction": d.get("instruction"), "output_length": d.get("output_length")}
    return None


def summarize_analysis(datasets: dict) -> dict:
    """サンプル分析を実行単位に集計する (UI の誤りパターン画面用)。行そのものは返さない"""
    out_ds = {}
    tot_st, tot_fl = {}, {}
    ols, rls = [], []
    for ds, a in datasets.items():
        st, fl = {}, {}
        for r in a["rows"]:
            st[r["st"]] = st.get(r["st"], 0) + 1
            for f in r["fl"]:
                fl[f] = fl.get(f, 0) + 1
            ols.append(r["ol"])
            rls.append(r["rl"])
        for k, v in st.items():
            tot_st[k] = tot_st.get(k, 0) + v
        for k, v in fl.items():
            tot_fl[k] = tot_fl.get(k, 0) + v
        e = {"n": len(a["rows"]), "lang": a["lang"], "judge": a["judge"], "approx": a["approx"], "status": st, "flags": fl,
             "out_len": mean(r["ol"] for r in a["rows"]), "reas_len": mean(r["rl"] for r in a["rows"])}
        if a["labels"]:
            pred, gold = {}, {}
            for r in a["rows"]:
                p = r["p"] if r["p"] in a["labels"] else "(選択肢外)"
                pred[p] = pred.get(p, 0) + 1
                gold[r["g"]] = gold.get(r["g"], 0) + 1
            e.update(labels=a["labels"], pred_dist=pred, gold_dist=gold)
        out_ds[ds] = e
    ols.sort()
    rls.sort()
    return {"n": sum(e["n"] for e in out_ds.values()), "status": tot_st, "flags": tot_fl, "datasets": out_ds,
            "out_len_median": ols[len(ols) // 2] if ols else None, "reas_len_median": rls[len(rls) // 2] if rls else None,
            "out_len_mean": mean(ols), "reas_len_mean": mean(rls)}


def check_against_scores(datasets: dict, scores: dict) -> list:
    """再計算した exact_match と ool の率を result.json の値と突き合わせる。[(ds, 指標, 再計算, result.json)] のずれ"""
    bad = []
    for ds, a in datasets.items():
        n = len(a["rows"])
        if not n:
            continue
        ool_ref = scores.get(f"{ds}_ool")
        ool = sum(1 for r in a["rows"] if r["st"] in ("empty", "unparsed", "out_of_label")) / n
        if num(ool_ref) is not None and abs(ool - ool_ref) > 1e-6:
            bad.append((ds, "ool", round(ool, 4), round(ool_ref, 4)))
        em_ref = scores.get(f"{ds}_exact_match")
        if a["judge"] == "exact" and "exact_match" in a["metrics"] and num(em_ref) is not None:
            em = sum(1 for r in a["rows"] if r["st"] == "correct") / n
            if abs(em - em_ref) > 1e-6:
                bad.append((ds, "exact_match", round(em, 4), round(em_ref, 4)))
        for m in ("mathematical_equivalence",):
            ref = scores.get(f"{ds}_{m}")
            if a["judge"] == "math" and m in a["metrics"] and num(ref) is not None:
                v = sum(1 for r in a["rows"] if r["st"] == "correct") / n
                if abs(v - ref) > 0.05:
                    bad.append((ds, m + " (近似)", round(v, 4), round(ref, 4)))
    return bad


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
