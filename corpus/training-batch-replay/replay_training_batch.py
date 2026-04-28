#!/usr/bin/env python3

import argparse
import gzip
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# The global variables for Megatron-LM imports are populated by configure_megatron_imports().
numpy: Any = None
torch: Any = None
BlendedMegatronDatasetBuilder: Any = None
build_tokenizer: Any = None
GPTDataset: Any = None
GPTDatasetConfig: Any = None
Split: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Megatron pretraining global batches on CPU without initializing "
            "distributed training or loading a model. Activate the Python virtual "
            "environment for Megatron-LM before running this script."
        )
    )
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Path to the Megatron-LM repository to import.",
    )
    parser.add_argument(
        "--iter-index",
        type=int,
        default=None,
        help=(
            "Replay only this training iteration/global batch index. This is "
            "a shorthand for --iter-start INDEX --iter-end INDEX+1."
        ),
    )
    parser.add_argument(
        "--iter-start",
        type=int,
        default=None,
        help=(
            "First training iteration/global batch index to replay, inclusive."
        ),
    )
    parser.add_argument(
        "--iter-end",
        type=int,
        default=None,
        help=(
            "Training iteration end, exclusive. Defaults to the end of the "
            "training loop when --iter-start is set."
        ),
    )
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument(
        "--train-iters",
        type=int,
        default=6000,
        help="Set this to the value used when the reference dump was generated.",
    )
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split", type=str, default="1,0,0")
    parser.add_argument("--tokenizer-type", type=str, default="Llama2Tokenizer")
    parser.add_argument("--tokenizer-model", type=str, required=True, help="Path to the tokenizer model file, e.g., a SentencePiece model.")
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=128)
    parser.add_argument("--vocab-extra-ids", type=int, default=0)
    parser.add_argument("--data-path", nargs="*", required=True, help="One or more dataset paths or weight-path pairs for blended training.")
    parser.add_argument("--data-cache-path", type=str, default="./cache")
    parser.add_argument("--mmap-bin-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/global-batches.jsonl.gz",
        help="Write gzip-compressed JSONL output to this .jsonl.gz path.",
    )
    return parser.parse_args()


def configure_megatron_imports(megatron_path: str) -> None:
    global numpy
    global torch
    global BlendedMegatronDatasetBuilder
    global build_tokenizer
    global GPTDataset
    global GPTDatasetConfig
    global Split

    resolved = Path(megatron_path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"--megatron-path does not exist: {resolved}")
    if not (resolved / "megatron").is_dir():
        raise FileNotFoundError(
            f"--megatron-path must point to a Megatron-LM repository: {resolved}"
        )

    resolved_str = str(resolved)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)

    try:
        import numpy as imported_numpy
        import torch as imported_torch
        from megatron.core.datasets.blended_megatron_dataset_builder import (
            BlendedMegatronDatasetBuilder as ImportedBlendedMegatronDatasetBuilder,
        )
        from megatron.core.datasets.gpt_dataset import (
            GPTDataset as ImportedGPTDataset,
            GPTDatasetConfig as ImportedGPTDatasetConfig,
        )
        from megatron.core.datasets.utils import Split as ImportedSplit
        from megatron.training.tokenizer import build_tokenizer as imported_build_tokenizer
    except ImportError as exc:
        raise ImportError(
            "Failed to import Megatron-LM modules. Make sure the Python virtual "
            "environment for Megatron-LM is activated before running this script."
        ) from exc

    numpy = imported_numpy
    torch = imported_torch
    BlendedMegatronDatasetBuilder = ImportedBlendedMegatronDatasetBuilder
    build_tokenizer = imported_build_tokenizer
    GPTDataset = ImportedGPTDataset
    GPTDatasetConfig = ImportedGPTDatasetConfig
    Split = ImportedSplit


@contextmanager
def rank_zero_when_distributed_is_not_initialized():
    if torch.distributed.is_initialized():
        yield
        return

    original_get_rank = torch.distributed.get_rank
    torch.distributed.get_rank = lambda *args, **kwargs: 0
    try:
        yield
    finally:
        torch.distributed.get_rank = original_get_rank


def build_tokenizer_for_replay(args: argparse.Namespace):
    tokenizer_args = SimpleNamespace(
        rank=0,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        vocab_file=None,
        merge_file=None,
        vocab_size=None,
        vocab_extra_ids=args.vocab_extra_ids,
        make_vocab_size_divisible_by=args.make_vocab_size_divisible_by,
        tensor_model_parallel_size=1,
        padded_vocab_size=None,
    )
    return build_tokenizer(tokenizer_args)


def get_iteration_range(args: argparse.Namespace) -> range:
    if args.iter_index is not None:
        if args.iter_index < 0:
            raise ValueError("--iter-index must be non-negative")
        start = args.iter_index
        end = args.iter_index + 1
    else:
        start = 0 if args.iter_start is None else args.iter_start
        end = args.train_iters if args.iter_end is None else args.iter_end

    if start < 0:
        raise ValueError("--iter-start must be non-negative")
    if end < start:
        raise ValueError("--iter-end must be greater than or equal to --iter-start")
    if end > args.train_iters:
        raise IndexError(
            f"requested iteration range exceeds training loop: end={end}, "
            f"total={args.train_iters}"
        )
    return range(start, end)


def build_train_dataset(args: argparse.Namespace):
    tokenizer = build_tokenizer_for_replay(args)
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[None, None, None],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        mock=False,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=True,
    )
    train_num_samples = args.train_iters * args.global_batch_size
    sizes = [train_num_samples, 0, 0]
    with rank_zero_when_distributed_is_not_initialized():
        train_ds, _, _ = BlendedMegatronDatasetBuilder(
            GPTDataset,
            sizes,
            lambda: True,
            config,
        ).build()
    if train_ds is None:
        raise RuntimeError("failed to build the train dataset")
    return train_ds, tokenizer


def get_dataset_paths_from_data_path(data_path: list[str]) -> list[str]:
    if len(data_path) == 1:
        return data_path

    if len(data_path) % 2 != 0:
        raise ValueError(
            "--data-path must be either a single dataset path or weight-path pairs"
        )

    dataset_paths = []
    for weight_index in range(0, len(data_path), 2):
        float(data_path[weight_index])
        dataset_paths.append(data_path[weight_index + 1])
    return dataset_paths


def get_global_batch_dataset_indices(
    args: argparse.Namespace,
    total_samples: int,
    iteration: int,
) -> list[int]:
    batch_start = iteration * args.global_batch_size
    if batch_start >= total_samples:
        raise IndexError(
            f"requested global batch exceeds dataset length: start={batch_start}, total={total_samples}"
        )
    batch_end = min(batch_start + args.global_batch_size, total_samples)
    return list(range(batch_start, batch_end))


def resolve_source_dataset_sample(
    dataset: Any,
    dataset_index: int,
    item: dict[str, Any],
) -> tuple[Any, int, int | None]:
    if hasattr(dataset, "datasets"):
        dataset_id = int(item.get("dataset_id", dataset.dataset_index[dataset_index]))
        dataset_sample_index = int(
            item.get("dataset_sample_index", dataset.dataset_sample_index[dataset_index])
        )
        return dataset.datasets[dataset_id], dataset_sample_index, dataset_id

    return dataset, dataset_index, None


def get_indexed_dataset_spans(
    source_dataset: Any,
    dataset_sample_index: int,
) -> list[dict[str, int]]:
    if not all(
        hasattr(source_dataset, attr)
        for attr in ("document_index", "sample_index", "shuffle_index", "dataset")
    ):
        return []

    shuffled_sample_index = int(source_dataset.shuffle_index[dataset_sample_index])
    doc_index_beg, doc_index_beg_offset = source_dataset.sample_index[shuffled_sample_index]
    doc_index_end, doc_index_end_offset = source_dataset.sample_index[shuffled_sample_index + 1]

    spans = []
    for doc_index in range(int(doc_index_beg), int(doc_index_end) + 1):
        document_id = int(source_dataset.document_index[doc_index])
        offset = 0 if doc_index > doc_index_beg else int(doc_index_beg_offset)
        if doc_index < doc_index_end:
            length = int(source_dataset.dataset.sequence_lengths[document_id]) - offset
        else:
            length = int(doc_index_end_offset) - offset + 1

        spans.append(
            {
                "document_id": document_id,
                "offset": offset,
                "length": length,
            }
        )

    return spans


def build_sample_record(
    dataset: Any,
    dataset_index: int,
    iteration: int,
    item: dict[str, Any],
    tokenizer: Any,
    dataset_paths: list[str],
) -> dict[str, Any]:
    token_ids = item["tokens"].tolist() + [int(item["labels"][-1])]
    record = {
        "iteration": iteration,
        "token_ids": token_ids,
        "text": tokenizer.detokenize(token_ids),
    }
    record["dataset_id"] = int(item.get("dataset_id", 0))
    if record["dataset_id"] < len(dataset_paths):
        record["dataset_path"] = dataset_paths[record["dataset_id"]]
    if hasattr(dataset, "dataset_sample_index"):
        record["dataset_sample_index"] = int(dataset.dataset_sample_index[dataset_index])
    else:
        record["dataset_sample_index"] = dataset_index

    source_dataset, source_sample_index, _source_dataset_id = resolve_source_dataset_sample(
        dataset,
        dataset_index,
        item,
    )
    indexed_dataset_spans = get_indexed_dataset_spans(source_dataset, source_sample_index)
    record["indexed_dataset_spans"] = indexed_dataset_spans
    if "dataset_path" not in record and hasattr(source_dataset, "dataset_path"):
        record["dataset_path"] = source_dataset.dataset_path
    return record


def iter_global_batch_records(
    args: argparse.Namespace,
    dataset: Any,
    tokenizer: Any,
    dataset_paths: list[str],
    iteration: int,
):
    dataset_indices = get_global_batch_dataset_indices(
        args,
        len(dataset),
        iteration,
    )
    for dataset_index in dataset_indices:
        item = dataset[dataset_index]
        yield build_sample_record(
            dataset,
            dataset_index,
            iteration,
            item,
            tokenizer,
            dataset_paths,
        )


def iter_output_payloads(args: argparse.Namespace):
    dataset, tokenizer = build_train_dataset(args)
    dataset_paths = get_dataset_paths_from_data_path(args.data_path)
    for iteration in get_iteration_range(args):
        yield from iter_global_batch_records(
            args,
            dataset,
            tokenizer,
            dataset_paths,
            iteration,
        )


def write_output(payloads: Any, output_path: str) -> None:
    resolved = Path(output_path)
    if resolved.suffixes[-2:] != [".jsonl", ".gz"]:
        raise ValueError(f"--output must end with .jsonl.gz: {output_path}")
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(resolved, "wt", encoding="utf-8") as writer:
        for payload in payloads:
            writer.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            writer.write("\n")


def main() -> None:
    args = parse_args()
    configure_megatron_imports(args.megatron_path)
    write_output(iter_output_payloads(args), args.output)


if __name__ == "__main__":
    main()
