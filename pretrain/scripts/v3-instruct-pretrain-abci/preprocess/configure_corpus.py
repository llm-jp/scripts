# Usage:
# Python configure_experiment.py \
#  --config ./experiments/exp1/config.csv \
#  --token-info ./token_info
#
# Config file should be CSV of "corpus_subset_name", "ratio"
# Token info file should be CSV of "original_file", "corpus_prefix", "num_tokens"

import argparse
import logging
import pathlib

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--token-info",
        type=pathlib.Path,
        required=True,
        help="Directory of token info",
    )
    p.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to experiment config YAML",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output shell script",
    )
    return p.parse_args()


def calc_corpus(content: list, corpus: dict[str, list[tuple[str, int]]]) -> list[tuple[str, float]]:
    # stats[i]: (prefix, num_tokens)
    stats: list[tuple[str, float]] = []

    for c in content:
        weight = float(c.get("weight", 1.0))
        if weight == 0.0:
            continue

        for prefix, num_tokens in corpus[c["name"]]:
            if num_tokens == 0:
                continue

            stats.append((prefix, num_tokens * weight))
    
    total_tokens = sum(x[1] for x in stats)

    return [(x[0], x[1] / total_tokens) for x in stats]


def calc_mixture(content: list, corpus: dict[str, list[tuple[str, int]]]) -> list[tuple[str, float]]:
    raw_weights = [float(c.get("weight", 1.0)) for c in content]
    total_weight = sum(raw_weights)
    weights = [r / total_weight for r in raw_weights]

    result = []

    for c, weight in zip(content, weights):
        if weight == 0.0:
            continue

        if c["type"] == "mixture":
            child = calc_mixture(c["content"], corpus)
        elif c["type"] == "corpus":
            child = calc_corpus(c["content"], corpus)
        else:
            continue
        
        for prefix, child_weight in child:
            result.append((prefix, child_weight * weight))
    
    return result


def main():
    args = parse_args()

    # corpus[subset] = [(prefix, num_tokens)]
    corpus: dict[str, list[tuple[str, int]]] = {}

    for token_info_file in args.token_info.glob("*.csv"):
        logging.info(f"Processing token info: {token_info_file}")
        file_list = []
        with token_info_file.open() as fp:
            for line in fp:
                _, prefix, num_tokens_str = line.strip().split(",")
                num_tokens = int(num_tokens_str)
                file_list.append((prefix, num_tokens))
        corpus[token_info_file.stem] = file_list

    # config[subset] = ratio
    config: dict[str, int] = {}

    logging.info(f"Processing config: {token_info_file}")
    with args.config.open() as fp:
        content = [yaml.safe_load(fp)]
    
    result = calc_mixture(content, corpus)

    # Dump Megatron-LM corpus config
    with args.output.open("w") as fp:
        print("export TRAIN_DATA_PATH=(", file=fp)
    
        for prefix, weight in result:
            print(f"    {weight} {prefix}", file=fp)

        print(")", file=fp)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
