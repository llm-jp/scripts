# Calculates model statistics from a given Megatron-LM checkpoint.
#
# This script depends on torch and Megatron-LM used to train the given checkpoint.
#
# This script can be run on both CPU or GPU, but GPU is recommended for faster
# computation of statistics.
#
# Usage:
#     source {venv used to train model}/bin/activate
#     python calculate_model_stats.py \
#         --megatron /path/to/Megatron-LM \
#         --checkpoint /path/to/checkpoints/iter_XXXXXXX \
#         --output /path/to/output.json
import logging
logging.basicConfig(level=logging.INFO)

# Immediately show this message when starting the script.
logging.info("Loading libraries")

import argparse
import collections
import dataclasses
import io
import json
import logging
import pathlib
import re
import sys
from typing import TypedDict

import torch


@dataclasses.dataclass(frozen=True)
class Args:
    megatron: pathlib.Path
    checkpoint: pathlib.Path
    output: pathlib.Path


class MegatronCheckpointV3(TypedDict):
    args: argparse.Namespace
    checkpoint_version: float  # 3.0
    iteration: int
    tokens: int
    model: collections.OrderedDict[str, torch.Tensor | io.BytesIO]
    optimizer: dict
    opt_param_scheduler: dict
    rng_state: list
    num_floating_point_operations_so_far: int


@dataclasses.dataclass(frozen=True)
class WeightStats:
    shape: list[int]
    min: list[float]
    max: list[float]
    sum: list[float]
    abs_min: list[float]
    abs_max: list[float]
    abs_sum: list[float]


@dataclasses.dataclass(frozen=True)
class SplitStats:
    tp_rank: int
    pp_rank: int
    weight_stats: dict[str, WeightStats]


def parse_args() -> Args:
    """Parses commandline arguments.

    Returns:
        Parsed arguments
    """
    p = argparse.ArgumentParser(
        description="Calculates model statistics from a given Megatron-LM checkpoint."
    )

    p.add_argument(
        "--megatron",
        type=pathlib.Path,
        required=True,
        help="Megatron-LM path (cloned repository) used to train the model",
    )
    p.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the checkpoint directory, containing all TP/PP splits",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="output JSON file",
    )

    args = p.parse_args()

    return Args(
        megatron=args.megatron,
        checkpoint=args.checkpoint,
        output=args.output,
    )


def load_checkpoint(path: pathlib.Path) -> MegatronCheckpointV3:
    """Loads Megatron-LM checkpoint.

    Args:
        path: Path to the PyTorch checkpoint file (.pt)

    Returns:
        Statistics calculated from the checkpoint file
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        logging.warn("No GPU detected")
        device = torch.device("cpu")

    ckpt = torch.load(path, map_location=device)

    ckpt_ver = ckpt["checkpoint_version"]
    if ckpt_ver != 3.0:  # float, but should match exactly
        raise ValueError(f"Checkpoint version {ckpt_ver} is not supported.")

    return ckpt


def calculate_stats(weight: torch.Tensor) -> WeightStats:
    """Calculates statistics of given weight tensors.

    Args:
        weight: Target tensor

    Returns:
        Statistics calculated from `weight`
    """
    shape = weight.shape
    w = weight.flatten().double()
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w_list = [w, w2, w3, w4]
    a = w.abs()
    a2 = a * a
    a3 = a2 * a
    a4 = a2 * a2
    a_list = [a, a2, a3, a4]

    del weight

    return WeightStats(
        shape=list(shape),
        min=[float(x.min()) for x in w_list],
        max=[float(x.max()) for x in w_list],
        sum=[float(x.sum()) for x in w_list],
        abs_min=[float(x.min()) for x in a_list],
        abs_max=[float(x.max()) for x in a_list],
        abs_sum=[float(x.sum()) for x in a_list],
    )


filename_pattern = re.compile(r"^mp_rank_([0-9]{2})_([0-9]{3})$")


def process_split(path: pathlib.Path) -> SplitStats:
    """Calculates statistics and other information from given TP/PP split.

    Args:
        path: Path to the split directory

    Returns:
        Statistics calculated from the given split
    """
    m = filename_pattern.match(path.name)
    if m is None:
        raise ValueError("Directory name is not supported: it should be mp_rank_XX_XXX")

    tp_rank = int(m.group(1))
    pp_rank = int(m.group(2))

    logging.info(f"Loading checkpoint from {path}")
    ckpt = load_checkpoint(path / "model_optim_rng.pt")

    logging.info(f"Calculating statistics for {path}")
    model_weights = {
        k: v
        for k, v in ckpt["model"].items()
        if isinstance(v, torch.Tensor)
    }
    weight_stats = {k: calculate_stats(v) for k, v in model_weights.items()}

    return SplitStats(tp_rank=tp_rank, pp_rank=pp_rank, weight_stats=weight_stats)


# Which dimension should be aggregated,
# or None to indicate just copy tensors across all TP ranks.
aggregate_dims: list[tuple[str, int | None]] = [
    (r"^embedding\.word_embeddings\.weight$", 0),
    (r"^decoder\.layers\.[0-9]+\.self_attention\.linear_proj\.weight$", 1),
    (r"^decoder\.layers\.[0-9]+\.self_attention\.linear_qkv\.layer_norm_weight$", None),
    (r"^decoder\.layers\.[0-9]+\.self_attention\.linear_qkv\.weight$", 0),
    (r"^decoder\.layers\.[0-9]+\.mlp\.linear_fc1\.layer_norm_weight$", None),
    (r"^decoder\.layers\.[0-9]+\.mlp\.linear_fc1\.weight$", 0),
    (r"^decoder\.layers\.[0-9]+\.mlp\.linear_fc2\.weight$", 1),
    (r"^decoder\.final_layernorm\.weight$", None),
    (r"^output_layer\.weight$", 0),
]
aggregate_dim_patterns: list[tuple[re.Pattern, int | None]] = [
    (re.compile(p), d) for p, d in aggregate_dims
]


def aggregate_weight_stats(name: str, a: WeightStats, b: WeightStats) -> WeightStats:
    """Aggregates two WeightStats.

    Args:
        name: Name of WeightStats, determining how stats are aggregated.
        a: A WeightStats.
        b: Another WeightStats.

    Returns:
        Aggregated WeightStats.
    """
    if not (1 <= len(a.shape) == len(b.shape) <= 2):
        raise ValueError(f"Unsupported shape to aggregate: {a.shape} and {b.shape}")
    ndims = len(a.shape)

    for p, d in aggregate_dim_patterns:
        if p.match(name):
            if d is None:
                # Weights are just copied across all TP ranks.
                return a
            else:
                # Weights are split into small fractions for each TP rank,
                # should be aggregated.
                if ndims < d + 1:
                    raise ValueError(f"Too few number of dimensions: {ndims}")
                return WeightStats(
                    shape=[
                        a.shape[i] + b.shape[i] if i == d else a.shape[i]
                        for i in range(ndims)
                    ],
                    min=[min(x, y) for x, y in zip(a.min, b.min)],
                    max=[max(x, y) for x, y in zip(a.max, b.max)],
                    sum=[x + y for x, y in zip(a.sum, b.sum)],
                    abs_min=[min(x, y) for x, y in zip(a.abs_min, b.abs_min)],
                    abs_max=[max(x, y) for x, y in zip(a.abs_max, b.abs_max)],
                    abs_sum=[x + y for x, y in zip(a.abs_sum, b.abs_sum)],
                )

    raise ValueError(f"No matches for the weight name {name}.")


def aggregate_split_stats(a: SplitStats, b: SplitStats) -> SplitStats:
    """Aggregates two SplitStats.

    Args:
        a: A SplitStats
        b: Another SplitStats

    Returns:
        Aggregated SplitStats
    """
    if a.weight_stats.keys() != b.weight_stats.keys():
        raise ValueError("Incompatible tensor parallel splits.")

    weight_stats = {
        k: aggregate_weight_stats(k, v, b.weight_stats[k])
        for k, v in a.weight_stats.items()
    }

    return SplitStats(
        tp_rank=min(a.tp_rank, b.tp_rank),
        pp_rank=a.pp_rank,
        weight_stats=weight_stats,
    )


def aggregate_tensor_parallel(stats: list[SplitStats]) -> list[SplitStats]:
    """Integrate tensor parallel splits into one statistics.

    Args:
        stats: List of statistics for each split

    Returns:
        Aggregated statistics. Since tensor parallel splits are aggregated, the
        `tp_rank` fileds of returned statistics becomes 0, while `pp_rank` is
        maintained.
    """
    logging.info("Aggregating tensor parallel splits")

    # {pp_rank: aggregated_stats}
    pp_stats: dict[int, SplitStats] = {}

    for s in stats:
        pp_rank = s.pp_rank
        if pp_rank not in pp_stats:
            pp_stats[pp_rank] = s
        else:
            pp_stats[pp_rank] = aggregate_split_stats(pp_stats[pp_rank], s)

    return list(pp_stats.values())


def add_offset_to_decoder(stats: SplitStats, offset: int) -> tuple[SplitStats, int]:
    """Rename decoder weights according to offset.

    Thsi function renames all weights with name `decoder.layers.n` to
    `decoder.layers.(n+offset)` and returns new offset number.

    Args:
        stats: SplitStats to transform.
        offset: Offset number, representing the number of preceding decoder layers.

    Returns:
        Transformed SplitStats.
    """
    weight_stats: dict[str, WeightStats] = {}
    new_offset = 0

    for k, v in stats.weight_stats.items():
        if k.startswith("decoder.layers."):
            k_splits = k.split(".")
            new_layer_id = int(k_splits[2]) + offset
            new_offset = max(new_offset, new_layer_id + 1)
            new_k = ".".join(k_splits[:2] + [str(new_layer_id)] + k_splits[3:])
            weight_stats[new_k] = v
        else:
            # Other weights
            weight_stats[k] = v

    return (
        SplitStats(tp_rank=stats.tp_rank, pp_rank=0, weight_stats=weight_stats),
        new_offset,
    )


def aggregate_pipeline_parallel(stats: list[SplitStats]) -> list[SplitStats]:
    """Integrate pipeline parallel splits into one statistics.

    Args:
        stats: List of statistics for each split

    Returns:
        Aggregated statistics. Since pipeline parallel splits are aggregated, the
        `pp_rank` fileds of returned statistics becomes 0, while `tp_rank` is
        maintained.
    """
    logging.info("Aggregating pipeline parallel splits")

    # {tp_rank: (aggregated_stats, decoder_offset)}
    tp_stats: dict[int, tuple[SplitStats, int]] = {}

    # Aggregate PP splits from smaller ranks.
    for s in sorted(stats, key=lambda x: x.pp_rank):
        tp_rank = s.tp_rank
        if tp_rank not in tp_stats:
            tp_stats[tp_rank] = add_offset_to_decoder(s, 0)
        else:
            old_s, offset = tp_stats[tp_rank]
            new_s, new_offset = add_offset_to_decoder(s, offset)
            weight_stats = {**old_s.weight_stats, **new_s.weight_stats}
            tp_stats[tp_rank] = (
                SplitStats(tp_rank=tp_rank, pp_rank=0, weight_stats=weight_stats),
                new_offset,
            )

    return [x[0] for x in tp_stats.values()]


def main() -> None:
    """Main routine."""
    args = parse_args()
    sys.path.append(str(args.megatron))

    stats = [process_split(p) for p in args.checkpoint.glob("mp_rank_??_???")]
    stats = aggregate_tensor_parallel(stats)
    stats = aggregate_pipeline_parallel(stats)

    assert len(stats) == 1

    logging.info("Writing outputs")
    weight_stats_json = {
        k: dataclasses.asdict(v)
        for k, v in stats[0].weight_stats.items()
    }

    with args.output.open("w") as fp:
        json.dump(weight_stats_json, fp, indent=2)


if __name__ == "__main__":
    main()
