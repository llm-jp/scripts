"""Prefetch every model/tokenizer that the llm-jp-eval v2.x eval phase would
otherwise download at runtime, into caches passed on the command line.

Shared by all llm-jp-eval v2.x installers (v2.1.0 / v2.1.3 / v2.1.5 / ...): the
set of eval-time downloads is identical across these versions, so keeping this
in one place avoids duplicating the logic in each install.sh.

Usage (run with the llm-jp-eval venv, e.g. `uv run python`):
    prefetch_eval_caches.py COMET_CACHE_DIR

Environment:
    HF_HOME    must point at the shared HF cache to populate (encoders below).
    NLTK_DATA  must point at the shared nltk data dir to populate.

Populates:
    COMET_CACHE_DIR : COMET checkpoint (Unbabel/wmt22-comet-da)
    HF_HOME         : COMET encoder (xlm-roberta-large) + BERTScore encoders
                      (roberta-large for en, bert-base-multilingual-cased for ja)
    NLTK_DATA       : punkt_tab (init_metrics) and punkt (mifeval)

Forced onto CPU so it runs on a GPU-less install node and needs no
Blackwell/sm_100 kernels (the locked torch may predate the eval-phase override).
"""

import sys

from comet import download_model, load_from_checkpoint

comet_cache = sys.argv[1]
# Downloads the checkpoint into comet_cache and pulls its encoder into HF_HOME.
load_from_checkpoint(download_model("Unbabel/wmt22-comet-da", saving_directory=comet_cache))

import bert_score

# lang="en" -> roberta-large, lang="ja" -> bert-base-multilingual-cased.
for lang in ("en", "ja"):
    bert_score.score(["x"], ["x"], lang=lang, device="cpu")

import nltk

nltk.download("punkt_tab")  # used by init_metrics (mifeval)
nltk.download("punkt")  # used by mifeval's punkt/english.pickle loader
