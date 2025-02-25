import argparse
import pathlib
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--megatron-path", type=pathlib.Path, help="Path to Megatron-LM repository")
    p.add_argument("--prefix", type=pathlib.Path, help="Prefix of .bin and .idx files.")
    args = p.parse_args()

    sys.path.append(str(args.megatron_path))
    from megatron.core.datasets import indexed_dataset

    ds = indexed_dataset.IndexedDataset(str(args.prefix) + "_text_document")
    
    with open(str(args.prefix) + ".num_tokens", "w") as fp:
        print(int(ds.sequence_lengths.sum()), file=fp)


if __name__ == "__main__":
    main()
