"""OpenAI-compatible-endpoint inference for llm-jp-eval (v2 series).

Drop-in alternative to llm-jp-eval-inference's offline vllm module: instead
of loading the model in-process, it sends the already-tokenized prompts to a
pre-launched vLLM server (/v1/completions), so one server process can be
shared by multiple evaluation runs and versions.

Run inside an llm-jp-eval-inference vllm-module venv (it provides
llm_jp_eval, llm_jp_eval_inference, transformers and openai):

    python inference_openai.py inference --config <config.yaml>
    python inference_openai.py get_run_name --config <config.yaml>

Parity notes vs the offline vllm module:
- Prompts are the same token IDs GeneratorBase produces (including the
  apply_chat_template path), sent as token-array prompts.
- generation_config defaults mirror VLLMSamplingParams (temperature=1.0,
  top_p=1.0, seed=None); vLLM-specific params (repetition_penalty, top_k,
  min_p, ...) are forwarded via extra_body.
- server.max_model_len emulates the offline `model.max_model_len` cap via
  vLLM's truncate_prompt_tokens extension.
- reasoning_parser is NOT supported; use the offline module for models that
  need one.
"""

import copy
import logging
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import openai
import transformers

from pydantic import BaseModel, ConfigDict, Field

from llm_jp_eval.cli import setup_cli
from llm_jp_eval.schemas import DatasetProfile
from llm_jp_eval_inference.generator import GeneratorBase
from llm_jp_eval_inference.schemas import BaseInferenceConfig

logger = logging.getLogger(__name__)

# Parameters accepted natively by the OpenAI completions API; everything else
# in generation_config is passed through vLLM's extra_body extension.
OPENAI_NATIVE_PARAMS = {
    "n",
    "presence_penalty",
    "frequency_penalty",
    "temperature",
    "top_p",
    "seed",
    "stop",
    "logprobs",
}


class ServerConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str
    api_key: str = "EMPTY"
    num_concurrent: int = 16
    max_retries: int = 3
    retry_backoff: float = 5.0
    timeout: float = 3600.0
    # Emulates the offline module's model.max_model_len: prompts are truncated
    # server-side so that prompt + max_tokens fits. Set null to disable.
    max_model_len: Optional[int] = 4096


class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    use_fast: bool = True


# Mirrors VLLMSamplingParams of the offline vllm module.
class SamplingParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None


class InferenceConfig(BaseInferenceConfig):
    server: ServerConfig
    tokenizer: TokenizerConfig
    generation_config: SamplingParams = Field(default_factory=SamplingParams)


def inference(cfg: InferenceConfig):
    generator = OpenAIGenerator(cfg)
    generator.main()


def get_run_name(cfg: InferenceConfig):
    generator = OpenAIGenerator(cfg)
    run_name = cfg.run_name or generator._get_default_run_name()
    print(run_name)


class OpenAIGenerator(GeneratorBase[InferenceConfig]):
    def __init__(self, cfg: InferenceConfig):
        super().__init__(cfg, "vllm-serve", openai)
        self.model_name = cfg.server.model.strip(" \t\r\n./").replace("/", "--")
        self.base_model_name = cfg.meta.base_model_name or self.model_name
        self.quantization = cfg.meta.quantization or "serve"
        self.max_len = cfg.server.max_model_len or 0

    def load_tokenizer(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(**self.cfg.tokenizer.model_dump())

    def load_model(self, dataset_profiles: Dict[str, DatasetProfile]):
        self.client = openai.OpenAI(
            base_url=self.cfg.server.base_url,
            api_key=self.cfg.server.api_key,
            timeout=self.cfg.server.timeout,
            max_retries=0,  # retries are handled below with backoff
        )

    def generate(
        self,
        max_input_len: int,
        max_output_len: int,
        target_data: Dict[str, Any],
        prompt_tokens: list,
        prompt_lengths: list,
    ) -> Dict[str, Any]:
        params = self.cfg.generation_config.model_dump(exclude_none=True)
        max_tokens = params.pop("max_tokens", None) or max_output_len
        native = {k: v for k, v in params.items() if k in OPENAI_NATIVE_PARAMS}
        extra_body = {k: v for k, v in params.items() if k not in OPENAI_NATIVE_PARAMS}
        max_model_len = self.cfg.server.max_model_len

        def _request_params(prompt_len: int) -> tuple:
            """(max_tokens, truncate_prompt_tokens) for one request.

            Offline parity under model.max_model_len: offline vLLM keeps the
            prompt as-is and implicitly clamps the generation length to the
            remaining context (prompt + output <= max_model_len), e.g. jhle
            requests output_length=8192 against a 4096 context. The OpenAI
            server instead rejects such requests up front, so apply the same
            clamp client-side. Prompts that alone fill the context (offline
            vLLM would raise) are truncated to leave room for one token.
            """
            if not max_model_len:
                return max_tokens, None
            truncate = None
            if prompt_len > max_model_len - 1:
                truncate = max_model_len - 1
                prompt_len = truncate
            return min(max_tokens, max_model_len - prompt_len), truncate

        def _complete(tokens: List[int]) -> str:
            request_max_tokens, truncate_prompt_tokens = _request_params(len(tokens))
            request_extra_body = dict(extra_body)
            if truncate_prompt_tokens is not None:
                request_extra_body["truncate_prompt_tokens"] = truncate_prompt_tokens
            last_exc: Exception | None = None
            for attempt in range(self.cfg.server.max_retries + 1):
                try:
                    response = self.client.completions.create(
                        model=self.cfg.server.model,
                        prompt=tokens,
                        max_tokens=request_max_tokens,
                        extra_body=request_extra_body or None,
                        **native,
                    )
                    return response.choices[0].text
                except (openai.APIConnectionError, openai.APIStatusError, openai.APITimeoutError) as e:
                    last_exc = e
                    logger.warning(f"completion request failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.cfg.server.retry_backoff * (attempt + 1))
            raise RuntimeError(f"completion request failed after retries: {last_exc}")

        results = copy.deepcopy(target_data)
        with ThreadPoolExecutor(max_workers=self.cfg.server.num_concurrent) as executor:
            generated_texts = list(executor.map(_complete, prompt_tokens))
        for sample, generated in zip(results["samples"], generated_texts, strict=True):
            sample["generated"] = generated
        return results


if __name__ == "__main__":
    cfg = setup_cli(
        InferenceConfig,
        commands={
            "inference": inference,
            "get_run_name": get_run_name,
        },
    )
