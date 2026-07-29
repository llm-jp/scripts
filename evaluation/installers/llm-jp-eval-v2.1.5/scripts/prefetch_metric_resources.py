"""Prefetch eval-phase metric resources at install time.

Usage: prefetch_metric_resources.py CACHE_DIR

CACHE_DIR must be <output_dir>/cache as seen by the evaluator; our run
scripts pass output_dir=<env>/data/llm-jp-eval, so the installer passes
<env>/data/llm-jp-eval/cache. The BERTScore models and the COMET encoder
land in the HF hub cache (HF_HOME), so install with the same HF_HOME that
evaluation jobs will use.
"""

import sys

from pathlib import Path


cache_dir = Path(sys.argv[1])
cache_dir.mkdir(parents=True, exist_ok=True)

# COMET: mirrors llm_jp_eval.metrics.metrics.get_comet_model, which is called
# with resource_dir=<output_dir>/cache. load_from_checkpoint additionally
# pulls the XLM-R encoder/tokenizer into the HF hub cache.
from comet import download_model, load_from_checkpoint  # noqa: E402

checkpoint_path = download_model("Unbabel/wmt22-comet-da", saving_directory=str(cache_dir))
load_from_checkpoint(checkpoint_path)
print("prefetched COMET model: Unbabel/wmt22-comet-da", flush=True)

# BERTScore default models: lang=ja -> bert-base-multilingual-cased,
# lang=en -> roberta-large. A dummy scoring downloads model and tokenizer.
import bert_score  # noqa: E402

for lang in ("ja", "en"):
    bert_score.score(["a"], ["a"], lang=lang, device="cpu")
    print(f"prefetched BERTScore model for lang={lang}", flush=True)

# NLTK tokenizer data used by mifeval (downloaded to ~/nltk_data otherwise
# at evaluation time).
import nltk  # noqa: E402

nltk.download("punkt_tab")
print("prefetched NLTK punkt_tab", flush=True)
