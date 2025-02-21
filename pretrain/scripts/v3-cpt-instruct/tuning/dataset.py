import json
from pathlib import Path
from typing import Callable, Optional

import torch
from megatron.core import parallel_state
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def build_dataloader(
    dataset: Dataset,
    consumed_samples: int,
    micro_batch_size: int,
    global_batch_size: int,
    collate_fn: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    common_params: dict = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "drop_last": True,
        "pad_samples_to_global_batch_size": False,
    }

    if seed is not None and seed >= 0:
        batch_sampler = MegatronPretrainingRandomBatchSampler(
            **common_params, seed=seed
        )
    else:
        batch_sampler = MegatronPretrainingBatchSampler(**common_params)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def custom_build_dataloader(
    dataset: Dataset,
    consumed_samples: int,
    mbs: int,
    gbs: int,
    num_workers: int = 0,
    drop_last: bool = True,
    pad_samples_to_global_batch_size: bool = False,
    load_gbs: bool = True,
    seed: Optional[int] = None,
    use_random_sampler: bool = True,
    collate_fn=None,
):
    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": mbs,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }

    if use_random_sampler:
        cls = (
            MegatronPretrainingRandomBatchSampler
            if load_gbs
            else MegatronPretrainingRandomSampler
        )
        common_params["seed"] = seed
    else:
        cls = (
            MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
        )
    batch_sampler = cls(**common_params)

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def load_datasets(
    cfg: DictConfig, res_quality_threshold: float
) -> tuple[list[dict], list[dict]]:
    data_name2num_examples: dict[str, dict[str, int]] = {}
    total_train_examples: list[dict] = []
    total_dev_examples: list[dict] = []
    for data_name, data_info in cfg.datasets.items():
        dataset_path: Path = Path(f"{cfg.data_dir}/{data_name}.jsonl")
        if not dataset_path.exists():
            raise FileNotFoundError(f"{dataset_path} does not exist.")
        if data_info.max_train_samples == 0:
            if is_global_rank_zero():
                logging.info(
                    f"max_train_samples for {data_name} is set to 0. Skip them."
                )
            continue

        if is_global_rank_zero():
            logging.info(f"processing {dataset_path}...")
        loaded_examples: list[dict] = []
        with dataset_path.open(encoding="utf-8") as f:
            for line in f:
                loaded_examples.append(json.loads(line))

        quality_filtered_examples: list[dict] = []
        for loaded_example in loaded_examples:
            if res_quality_threshold > 0:
                res_quality_scores: list[float] = []
                for message in loaded_example["messages"]:
                    if message["role"] == "assistant" and "reward" in message:
                        res_quality_scores.append(
                            0.65 * message["reward"]["helpfulness"]
                            + 0.8 * message["reward"]["correctness"]
                            + 0.45 * message["reward"]["coherence"]
                        )
                if (
                    len(res_quality_scores) != 0
                    and min(res_quality_scores) < res_quality_threshold
                ):
                    continue
            quality_filtered_examples.append(loaded_example)

        if (
            data_info.max_train_samples > len(quality_filtered_examples)
            and is_global_rank_zero()
        ):
            logging.warning(
                f"{data_name} has only {len(quality_filtered_examples)} examples, "
                f"but max_train_samples is set to {data_info.max_train_samples}. "
                "Use all examples."
            )

        max_train_samples: int = (
            data_info.max_train_samples
            if data_info.max_train_samples != -1
            else len(quality_filtered_examples)
        )
        max_dev_samples: int = 0
        if data_info.split_dev:
            max_dev_samples = min(
                cfg.max_dev_samples,
                int(len(quality_filtered_examples) * cfg.max_dev_ratio),
            )
        train_examples: list[dict] = (
            quality_filtered_examples[
                max_dev_samples : max_dev_samples + max_train_samples
            ]
            * data_info.upsampling_factor
        )
        dev_examples: list[dict] = (
            quality_filtered_examples[:max_dev_samples] * data_info.upsampling_factor
        )

        total_train_examples.extend(train_examples)
        total_dev_examples.extend(dev_examples)
        data_name2num_examples[data_name] = {
            "train": len(train_examples),
            "dev": len(dev_examples),
            "original": len(loaded_examples),
            "upsampling_factor": data_info.upsampling_factor,
        }

    if is_global_rank_zero():
        num_total_original_examples: int = 0
        logging.info("------------------------------")
        logging.info("Dataset summary (original -> train/dev)")
        for data_name, num_examples in data_name2num_examples.items():
            num_total_original_examples += num_examples["original"]
            logging.info(
                f"{data_name}: {num_examples['original']} -> {num_examples['train']}/{num_examples['dev']} (upsampling factor: {num_examples['upsampling_factor']})"
            )
        logging.info(
            f"Total: {num_total_original_examples} -> {len(total_train_examples)}/{len(total_dev_examples)}"
        )
        logging.info("------------------------------")

    return total_train_examples, total_dev_examples


class LLMJPSFTDataset(Dataset):
    def __init__(
        self,
        loaded_examples: list[dict],
        tokenizer: TokenizerSpec,
        use_loss_mask: bool,
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.use_loss_mask: bool = use_loss_mask
        self.max_seq_length: int = max_seq_length

        self.examples: list[dict[str, list[int]]] = self._process_examples(
            loaded_examples
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]

    def _process_examples(
        self, loaded_examples: list[dict]
    ) -> list[dict[str, list[int]]]:
        all_input_ids: list[int] = []
        all_loss_mask: list[int] = []
        for example_idx, loaded_example in enumerate(loaded_examples):
            conversation: list[dict[str, str]] = loaded_example["messages"]
            assert len(conversation) >= 3
            assert conversation[0]["role"] == "system"

            input_ids: list[int] = [self.tokenizer.bos_id] + self.tokenizer.text_to_ids(
                conversation[0]["content"]
            )
            loss_mask: list[int] = (
                [0] * len(input_ids) if self.use_loss_mask else [1] * len(input_ids)
            )
            for turn_idx in range(1, len(conversation[1:]) // 2 + 1):
                user_message: dict[str, str] = conversation[2 * turn_idx - 1]
                assistant_message: dict[str, str] = conversation[2 * turn_idx]
                assert user_message["role"] == "user"
                assert assistant_message["role"] == "assistant"

                if self.use_loss_mask:
                    prompt_ids: list[int] = self.tokenizer.text_to_ids(
                        f"\n\n### 指示:\n{user_message['content']}\n\n### 応答:\n"
                    )[1:]
                    response_ids: list[int] = self.tokenizer.text_to_ids(
                        f"\n{assistant_message['content']}"
                    )[2:] + [self.tokenizer.eos_id]
                    input_ids.extend(prompt_ids + response_ids)
                    loss_mask.extend([0] * len(prompt_ids) + [1] * len(response_ids))
                else:
                    prompt_response_ids: list[int] = self.tokenizer.text_to_ids(
                        f"\n\n### 指示:\n{user_message['content']}\n\n### 応答:\n{assistant_message['content']}"
                    )[1:] + [self.tokenizer.eos_id]
                    input_ids.extend(prompt_response_ids)
                    loss_mask.extend([1] * len(prompt_response_ids))

            if is_global_rank_zero() and example_idx < 2:
                logging.info(f"{example_idx = }")
                logging.info(f"{input_ids = }")
                logging.info(f"{loss_mask = }")

            all_input_ids.extend(input_ids)
            all_loss_mask.extend(loss_mask)

        examples: list[dict[str, list[int]]] = []
        for i in range(0, len(all_input_ids), self.max_seq_length + 1):
            chunked_input_ids: list[int] = all_input_ids[
                i : i + self.max_seq_length + 1
            ]
            chunked_loss_mask: list[int] = all_loss_mask[
                i : i + self.max_seq_length + 1
            ]
            if len(chunked_input_ids) == self.max_seq_length + 1:
                if set(chunked_loss_mask) == {0}:  # Skip if all loss_mask is 0
                    continue
                examples.append(
                    {"input_ids": chunked_input_ids, "loss_mask": chunked_loss_mask}
                )
        return examples

    @torch.no_grad()
    def _create_attention_mask(self, seq_length: int) -> torch.Tensor:
        attention_mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(
            0
        )  # (1, seq_length, seq_length)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids: list[list[int]] = [item["input_ids"][:-1] for item in batch]
        labels: list[list[int]] = [item["input_ids"][1:] for item in batch]
        loss_mask: list[list[int]] = [item["loss_mask"][1:] for item in batch]

        pro_batch = {
            "tokens": torch.LongTensor(input_ids),
            "position_ids": torch.LongTensor(
                [list(range(self.max_seq_length)) for _ in batch]
            ),
            "attention_mask": torch.stack(
                [self._create_attention_mask(self.max_seq_length) for _ in batch]
            ),
            "labels": torch.LongTensor(labels),
            "loss_mask": torch.LongTensor(loss_mask),
        }

        return pro_batch
