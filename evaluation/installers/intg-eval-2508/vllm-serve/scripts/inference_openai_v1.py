"""OpenAI-compatible-endpoint inference for llm-jp-eval v1.4.x.

Drop-in alternative to llm-jp-eval v1.4.x's offline_inference_vllm.py:
instead of loading the model in-process, it sends the dumped prompts to a
pre-launched vLLM server (/v1/completions). The dump (dump_prompts.py) and
eval (evaluate_llm.py) phases are unchanged.

Run inside the v1.4.1 environment's venv-vllm (it provides openai and
transformers; no llm-jp-eval import is needed):

    python inference_openai_v1.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model llm-jp/llm-jp-3-150m \
        --prompt-json-path '<out>/prompts/*.eval-prompt.json' \
        --output-dir <out>/offline

Parity notes vs offline_inference_vllm.py:
- Prompts are encoded with the same HF tokenizer and the same encode kwargs
  (add_special_tokens=False) and sent as token-array prompts.
- Generation is greedy (temperature=0.0), as in the offline config
  (generator_kwargs of config_offline_inference_vllm.yaml).
- max_tokens = output_length + output_length_delta per dataset, clamped per
  request to the server's remaining context, which is what offline vLLM
  does implicitly.
- The _config.json required by evaluate_llm.py's offline mode is written
  with the same keys the offline script produces.
"""

import argparse
import copy
import glob
import json
import logging
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

import openai
import transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENCODE_KWARGS = {"add_special_tokens": False}  # pipeline_kwargs of the offline config
GENERATOR_KWARGS = {"temperature": 0.0, "repetition_penalty": 1.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--tokenizer", default=None, help="default: --model")
    parser.add_argument("--prompt-json-path", required=True, help="glob of *.eval-prompt.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-concurrent", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def get_server_max_model_len(client: openai.OpenAI) -> int:
    model = client.models.list().data[0]
    # vLLM reports its context size as a non-standard field.
    max_model_len = (getattr(model, "model_extra", None) or {}).get("max_model_len")
    if not max_model_len:
        raise RuntimeError("server did not report max_model_len in /v1/models")
    return int(max_model_len)


def generate(client, cfg, target_data, tokenizer, max_model_len):
    max_tokens = target_data["output_length"] + target_data["config"].get("output_length_delta", 0)

    def _complete(tokens: List[int]) -> str:
        # Offline vLLM keeps the prompt and implicitly clamps the generation
        # length to the remaining context; mirror that per request (the
        # OpenAI server rejects prompt + max_tokens > max_model_len).
        prompt_len = min(len(tokens), max_model_len - 1)
        tokens = tokens[:prompt_len]
        request_max_tokens = min(max_tokens, max_model_len - prompt_len)
        last_exc = None
        for attempt in range(cfg.max_retries + 1):
            try:
                response = client.completions.create(
                    model=cfg.model,
                    prompt=tokens,
                    max_tokens=request_max_tokens,
                    **GENERATOR_KWARGS,
                )
                return response.choices[0].text
            except (openai.APIConnectionError, openai.APIStatusError, openai.APITimeoutError) as e:
                last_exc = e
                logger.warning(f"completion request failed (attempt {attempt + 1}): {e}")
                time.sleep(cfg.retry_backoff * (attempt + 1))
        raise RuntimeError(f"completion request failed after retries: {last_exc}")

    results = copy.deepcopy(target_data)
    prompt_tokens = [
        tokenizer.encode(sample["prompt"], **ENCODE_KWARGS) for sample in results["samples"]
    ]
    with ThreadPoolExecutor(max_workers=cfg.num_concurrent) as executor:
        generated_texts = list(executor.map(_complete, prompt_tokens))
    for sample, generated in zip(results["samples"], generated_texts):
        sample["generated"] = generated
    return results


def main():
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    tokenizer_name = cfg.tokenizer or cfg.model
    logger.info(f"loading tokenizer: {tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    client = openai.OpenAI(
        base_url=cfg.base_url, api_key="EMPTY", timeout=cfg.timeout, max_retries=0
    )
    max_model_len = get_server_max_model_len(client)
    logger.info(f"server max_model_len: {max_model_len}")
    time_profile = {"(total)": None, "(init)": (datetime.now() - start).total_seconds()}

    dump_prompts_config = None
    dump_prompts_config_dataset = None
    target_files = sorted(glob.glob(cfg.prompt_json_path))
    if not target_files:
        raise RuntimeError(f"No files matched to {cfg.prompt_json_path}")
    for target_file in target_files:
        logger.info(f"loading {target_file}")
        with open(target_file, encoding="utf8") as fin:
            target_data = json.load(fin)
        start_generate = datetime.now()
        results = generate(client, cfg, target_data, tokenizer, max_model_len)
        time_profile[target_data["target_dataset"]] = (
            datetime.now() - start_generate
        ).total_seconds()
        if not dump_prompts_config:
            dump_prompts_config = target_data["config"]
            dump_prompts_config_dataset = target_data["target_dataset"]
        elif dump_prompts_config != target_data["config"]:
            logger.warning(
                f"Inconsistent config found.\n"
                f'{dump_prompts_config_dataset}: {dump_prompts_config}\n'
                f'{target_data["target_dataset"]}: {target_data["config"]}'
            )

        result_path = output_dir / f"{target_data['target_dataset']}.eval-generated.json"
        with open(result_path, "w", encoding="utf8") as fout:
            json.dump(results, fout, indent=1, ensure_ascii=False)
        logger.info(f"results were saved to {result_path}")

    time_profile["(total)"] = (datetime.now() - start).total_seconds()
    # The keys evaluate_llm.py merges from the offline run's config; mirror
    # what offline_inference_vllm.py records (model comes from the server).
    offline_config = {
        "offline_inference": {
            "prompt_json_path": [cfg.prompt_json_path],
            "output_base_dir": None,
            "exact_output_dir": cfg.output_dir,
            "base_url": cfg.base_url,
            "model_name": cfg.model,
            "run_name": cfg.model.strip(" \t\r\n./").replace("/", "--"),
        },
        "model": {
            "_target_": "vllm.LLM (via vllm-serve)",
            "model": cfg.model,
        },
        "tokenizer": {
            "_target_": "transformers.AutoTokenizer.from_pretrained",
            "pretrained_model_name_or_path": tokenizer_name,
            "trust_remote_code": False,
            "use_fast": True,
        },
        "pipeline_kwargs": dict(ENCODE_KWARGS),
        "generator_kwargs": {"_target_": "vllm.SamplingParams", **GENERATOR_KWARGS},
        "dump_prompts_config": dump_prompts_config,
        "time_profile": time_profile,
    }
    with open(output_dir / "_config.json", "w", encoding="utf8") as fout:
        json.dump(offline_config, fout, indent=1, ensure_ascii=False)
    logger.info("Done")


if __name__ == "__main__":
    main()
