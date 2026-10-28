"""evalviz_server のテスト (標準ライブラリの unittest)。実データに依存しないよう、qsub.py の出力レイアウトを一時ディレクトリに組み立てる。

  cd evalviz && python3 -m unittest discover -s tests
実データでの点検は python -m evalviz_server check --samples (README)。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalviz_server import httpd, ljeval  # noqa: E402
from evalviz_server.discover import model_label  # noqa: E402
from evalviz_server.featured import Featured  # noqa: E402
from evalviz_server.index import Index  # noqa: E402
from evalviz_server.util import read_json_keys  # noqa: E402


def write(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding="utf-8")


CATEGORIES = {
    "NLI": {"description": "Natural Language Inference", "default_metric": "exact_match", "metrics": {}, "datasets": ["jnli"]},
    "MR": {"description": "Math Reasoning", "default_metric": "mathematical_equivalence", "metrics": {}, "datasets": ["gsm8k"]},
    "QA": {"description": "Question Answering", "default_metric": "exact_match", "metrics": {}, "datasets": ["aio"]},
}


def generated(ds, metrics, pattern, labels, samples):
    return {"target_dataset": ds, "output_length": 64, "metrics": metrics, "instruction": "", "prompt_template": "",
            "answer_extract_pattern": "", "answer_pattern_id": pattern, "system_prompt": None, "label_list": labels,
            "max_num_samples": 10, "config": {}, "samples": samples}


def sample(inp, gold, gen, reas="", meta=None):
    return {"input": inp, "prompt": f"P: {inp}", "system_prompt": None, "gold": gold, "metadata": meta or {}, "generated": gen, "reasoning_content": reas}


class Fixture:
    """出力ルート 1 つに、llm-jp-eval v2.1.5 (推論結果付き)・swallow・llm-jp-judge・safety-eval を持つ実行を 2 つ置く"""

    def __init__(self, root: Path):
        self.root = root
        for name, step, good in (("exp-step100", 100, False), ("exp-step200", 200, True)):
            d = root / "grp" / name
            write(d / "config.json", {"model_name_or_path": f"/groups/x/tasks/exp/checkpoints_hf/iter_{step:07d}", "apply_chat_template": True,
                                      "reasoning_parser": "llmjp4", "llm_jp_eval_max_num_samples": 10})
            write(d / "qsub.sh", "#!/bin/bash\n")
            v = d / "llm-jp-eval" / "v2.1.5"
            run_name = f"x::vllm::H200"
            jnli = [
                sample("a", "entailment", "<answer>entailment</answer>"),
                sample("b", "neutral", "<answer>entailment</answer>" if not good else "<answer>neutral</answer>"),
                sample("c", "contradiction", "<answer>矛盾</answer>"),                    # 選択肢外
                sample("d", "neutral", "", reas="考え中" * 10),                           # 推論の打ち切り
                sample("e", "neutral", "answer: neutral"),                                # 抽出不可
            ]
            gsm = [sample("q1", "16", "So \\boxed{16\\text{ hours}}"), sample("q2", "70000", "\\boxed{70{,}000}"), sample("q3", "5", "\\boxed{6}")]
            aio = [sample("x", "東京", "<answer>东京</answer>这是我们的回答说明进行"),         # 中国語の混入
                   sample("y", "大阪", "<answer>大阪</answer>\n\n### 入力:\n質問: 次の問題はなんでしょうかという長い続きの文章がここにある")]
            write(v / "offline" / run_name / "jnli.eval-generated.json", generated("jnli", ["exact_match"], "answer_tags_jp", ["entailment", "contradiction", "neutral"], jnli))
            write(v / "offline" / run_name / "gsm8k.eval-generated.json", generated("gsm8k", ["mathematical_equivalence"], "latex_boxed_en", [], gsm))
            write(v / "offline" / run_name / "aio.eval-generated.json", generated("aio", ["exact_match", "char_f1"], "answer_tags_jp", [], aio))
            jn = (2 if good else 1) / 5
            scores = {"jnli_ool": 3 / 5, "jnli_exact_match": jn, "gsm8k_ool": 0.0, "gsm8k_mathematical_equivalence": 2 / 3,
                      "aio_ool": 0.0, "aio_exact_match": 0.5, "aio_char_f1": 0.7, "NLI": jn, "MR": 2 / 3, "QA": 0.5}
            lang = {"JA": {"NLI": jn, "QA": 0.5, "AVG": (jn + 0.5) / 2}, "EN": {"MR": 2 / 3, "AVG": 2 / 3}}
            write(v / "results" / "result.json", {
                "config": {"max_num_samples": 10, "num_few_shots": 4, "eval_dataset_config": {"datasets": ["jnli", "gsm8k", "aio"], "categories": CATEGORIES},
                           "inference_result_dir": str(v / "offline" / run_name)},
                "metadata": {}, "evaluation": {"scores": scores, "lang_scores": lang, "time_profile": {"(infer)": 12.0}},
                "records": [], "scores": scores, "lang_scores": lang})
            write(d / "swallow" / "results" / "result.json", {"scores": {"gsm8k": 0.5, "mmlu": 0.6, "jcommonsenseqa": -1.0}})
            write(d / "llm-jp-judge" / "evaluation" / "score_table.json", [{"generation_model": "m", "evaluation_model": "judge-x", "quality_ja:総合評価": 3.5, "safety_ja:違反回答率(%)": 10.0}])
            write(d / "safety-eval" / "evaluate_count" / "m_evaluate" / "jtruthfulqa_base_evaluated_2026-09-02.json",
                  {"task_type": "jtruthfulqa", "overall": {"truthful_rate": 0.2, "valid_count": 15, "invalid_count": 0}})
        # 実行ではないディレクトリ (エージェント評価のジョブ等)
        write(root / "grp" / "not-a-run" / "0449_job" / "result.json", {})


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        Fixture(base / "out")
        # 掲載モデル: 出力ルート内の実行 2 つ、ルート外の実行 (safety-eval だけ)、存在しない実行
        outside = base / "elsewhere" / "safety-only"
        write(outside / "safety-eval" / "evaluate_count" / "m_evaluate" / "jbbq_age_base_evaluated_2026-09-02.json", {"task_type": "jbbq", "Age_0shot_acc": 0.7})
        cls.featured_path = base / "featured.json"
        write(cls.featured_path, {"_comment": "x", "standard": {"llm-jp-eval": {"version": "v2.1.5", "max_num_samples": 10}}, "models": [
            {"name": "A", "runs": [str(base / "out/grp/exp-step200"), str(outside)], "params": "8B"},
            {"name": "B", "baseline": True, "runs": [str(base / "out/grp/exp-step100")]},
            {"name": "C", "runs": [str(base / "missing")]},
        ]})
        cls.idx = Index([base / "out"], base / "cache" / "runs.json", workers=2, featured=Featured(cls.featured_path))
        cls.idx.refresh()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_discover(self):
        ids = sorted(r["id"] for r in self.idx.runs)
        self.assertEqual(ids, ["grp:exp-step100", "grp:exp-step200", "safety-only"])   # safety-only は掲載モデルの設定から
        r = self.idx.run("grp:exp-step200")
        self.assertEqual(r["step"], 200)
        self.assertEqual(r["series"], "exp-step*")
        self.assertEqual(r["model"], "exp/iter_0000200")
        self.assertEqual(r["settings"]["reasoning_parser"], "llmjp4")

    def test_summary(self):
        c = self.idx.run("grp:exp-step200")["components"]
        lj = c["llm-jp-eval"]["v2.1.5"]
        self.assertEqual(lj["status"], "done")
        self.assertAlmostEqual(lj["avg"]["EN"], 2 / 3)
        self.assertEqual(lj["datasets"]["jnli"]["category"], "NLI")
        self.assertEqual(lj["datasets"]["gsm8k"]["lang"], "EN")
        self.assertTrue(lj["has_offline"])
        self.assertEqual(c["swallow"]["scores"], {"gsm8k": 0.5, "mmlu": 0.6})
        self.assertAlmostEqual(c["swallow"]["avg"]["EN"], 0.55)
        self.assertEqual(c["llm-jp-judge"]["scores"]["quality_ja"]["総合評価"], 3.5)
        self.assertEqual(c["safety-eval"]["benchmarks"]["jtruthfulqa"]["headline"], 0.2)

    def test_analysis_matches_scores(self):
        for rid in ("grp:exp-step100", "grp:exp-step200"):
            a = self.idx.analysis(rid, "v2.1.5")
            vd = Path(self.idx.run(rid)["dir"]) / "llm-jp-eval" / "v2.1.5"
            self.assertEqual(ljeval.check_against_scores(a, ljeval.raw_scores(vd)), [])

    def test_statuses_and_flags(self):
        a = self.idx.analysis("grp:exp-step100", "v2.1.5")
        st = [r["st"] for r in a["jnli"]["rows"]]
        self.assertEqual(st, ["correct", "wrong", "out_of_label", "empty", "unparsed"])
        self.assertIn("reasoning_cut", a["jnli"]["rows"][3]["fl"])
        self.assertEqual([r["st"] for r in a["gsm8k"]["rows"]], ["correct", "correct", "wrong"])
        self.assertIn("zh_mix", a["aio"]["rows"][0]["fl"])
        self.assertIn("continued", a["aio"]["rows"][1]["fl"])
        s = ljeval.summarize_analysis(a)
        self.assertEqual(s["datasets"]["jnli"]["pred_dist"]["(選択肢外)"], 3)
        self.assertEqual(s["flags"]["reasoning_cut"], 1)

    def test_api_samples(self):
        runs = "grp:exp-step100,grp:exp-step200"
        out = httpd.api_samples(self.idx, {"runs": runs, "ver": "v2.1.5", "ds": "jnli"})
        self.assertEqual(out["info"]["labels"], ["entailment", "contradiction", "neutral"])
        row = next(r for r in out["rows"] if r["index"] == 1)
        self.assertEqual([c["st"] for c in row["cells"]], ["wrong", "correct"])
        d = httpd.api_sample(self.idx, {"runs": runs, "ver": "v2.1.5", "ds": "jnli", "key": row["key"]})
        self.assertEqual([r["generated"] for r in d["runs"]], ["<answer>entailment</answer>", "<answer>neutral</answer>"])
        agg = httpd.api_analysis(self.idx, {"runs": runs, "ver": "v2.1.5"})
        self.assertTrue(all(a["available"] for a in agg))

    def test_featured(self):
        out = httpd.api_featured(self.idx, {})
        a, b, c = out["models"]
        self.assertEqual(a["source"], {"llm-jp-eval": "grp:exp-step200", "swallow": "grp:exp-step200", "llm-jp-judge": "grp:exp-step200", "safety-eval": "grp:exp-step200"})
        self.assertTrue(b["baseline"])
        self.assertEqual(c["runs"], [])
        self.assertTrue(any("C: 実行が見つからない" in p for p in out["problems"]))
        # 標準条件のサンプル数が違えば llm-jp-eval の値は取らない
        cfg = json.loads(self.featured_path.read_text())
        cfg["standard"]["llm-jp-eval"]["max_num_samples"] = 100
        from evalviz_server.featured import resolve
        res = resolve(cfg, {r["dir"]: r for r in self.idx.runs})
        self.assertNotIn("llm-jp-eval", res["models"][0]["source"])
        self.assertTrue(any("A: 標準条件" in p and "v2.1.5 (10 件)" in p for p in res["problems"]))

    def test_featured_missing_file(self):
        f = Featured(Path(self.tmp.name) / "nope.json")
        self.assertEqual(f.load(), {})
        self.assertIn("設定ファイルが無い", f.error)

    def test_cache_reuse(self):
        idx2 = Index(self.idx.roots, self.idx.cache_path, workers=2, featured=Featured(self.featured_path))
        self.assertEqual(idx2.refresh(), 0)   # 要約はディスクキャッシュから


class TestUnits(unittest.TestCase):
    def test_model_label(self):
        self.assertEqual(model_label("llm-jp/llm-jp-4-8b-thinking"), "llm-jp/llm-jp-4-8b-thinking")
        self.assertEqual(model_label("/g/tasks/ratio0.8_lr1e-4/checkpoints_hf/iter_0024000"), "ratio0.8_lr1e-4/iter_0024000")
        self.assertEqual(model_label("/g/LLMJP4.8B-DolciCode/global_step_960_fp16"), "LLMJP4.8B-DolciCode/global_step_960_fp16")

    def test_bbh_extract(self):
        meta = {"prompt_type": "cot", "evaluation_type": "choice"}
        self.assertEqual(ljeval.extract("... So the answer is B.", "dataset", "", "bigbenchhard_cot", meta)[0], "(B)")
        self.assertEqual(ljeval.extract("よって答えは「はい」です", "dataset", "", "bigbenchhard_ja_cot", {"prompt_type": "cot"})[0], "はい")

    def test_math_equal(self):
        self.assertTrue(ljeval._math_equal("\\frac{1}{2}", "0.5"))
        self.assertTrue(ljeval._math_equal("82,285", "82,285百万円"))
        self.assertFalse(ljeval._math_equal("2+3", "2"))

    def test_repetition(self):
        self.assertTrue(ljeval._repetitive("写" * 3000))
        self.assertFalse(ljeval._repetitive("".join(chr(0x4E00 + (i * 7919) % 20000) for i in range(3000))))

    def test_read_json_keys_large(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "r.json"
            p.write_text(json.dumps({"config": {"a": 1}, "evaluation": {"scores": {"x": 1}}, "records": ["y" * 100]}))
            self.assertEqual(read_json_keys(p, ("evaluation",)), {"evaluation": {"scores": {"x": 1}}})
            import evalviz_server.util as u
            old = u._BIG_JSON
            u._BIG_JSON = 10   # 部分読みの経路
            try:
                self.assertEqual(read_json_keys(p, ("config", "evaluation")), {"config": {"a": 1}, "evaluation": {"scores": {"x": 1}}})
            finally:
                u._BIG_JSON = old


if __name__ == "__main__":
    unittest.main()
