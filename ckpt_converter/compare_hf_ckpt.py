#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Dict, Tuple, List, Optional

import torch
from safetensors.torch import load_file as safe_load_file

INDEX_NAME = "model.safetensors.index.json"


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_index_file(dirpath: str) -> Optional[str]:
    p = os.path.join(dirpath, INDEX_NAME)
    return p if os.path.isfile(p) else None


def _list_safetensors_files(dirpath: str) -> List[str]:
    return sorted(
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if f.endswith(".safetensors")
    )


def _load_state_dict_from_dir(dirpath: str) -> Dict[str, torch.Tensor]:
    index_path = _find_index_file(dirpath)
    state: Dict[str, torch.Tensor] = {}
    if index_path is not None:
        index = _read_json(index_path)
        weight_map: Dict[str, str] = index.get("weight_map") or {}
        shards = sorted(set(weight_map.values()))
        for shard in shards:
            shard_path = os.path.join(dirpath, shard)
            shard_sd = safe_load_file(shard_path, device="cpu")
            for k, v in shard_sd.items():
                if weight_map.get(k) == shard:
                    state[k] = v.detach().cpu()
        return state

    files = _list_safetensors_files(dirpath)
    if not files:
        raise RuntimeError(f"No .safetensors found in: {dirpath}")
    for wf in files:
        shard_sd = safe_load_file(wf, device="cpu")
        for k, v in shard_sd.items():
            state[k] = v.detach().cpu()
    return state


def compare_checkpoints(dir_a: str, dir_b: str, max_diffs: int = 20) -> Tuple[bool, str]:
    sd_a = _load_state_dict_from_dir(dir_a)
    sd_b = _load_state_dict_from_dir(dir_b)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    msgs: List[str] = []
    missing_in_b = sorted(keys_a - keys_b)
    extra_in_b = sorted(keys_b - keys_a)
    if missing_in_b:
        msgs.append(f"Missing in B ({len(missing_in_b)}): {missing_in_b[:max_diffs]}")
    if extra_in_b:
        msgs.append(f"Extra in B ({len(extra_in_b)}): {extra_in_b[:max_diffs]}")

    common = sorted(keys_a & keys_b)
    shape_mismatch = []
    dtype_mismatch = []
    value_mismatch = []

    for name in common:
        t1 = sd_a[name]
        t2 = sd_b[name]
        if t1.shape != t2.shape:
            shape_mismatch.append((name, t1.shape, t2.shape))
            if len(shape_mismatch) >= max_diffs:
                break
    for name, _, _ in shape_mismatch:
        if name in common:
            common.remove(name)

    for name in common:
        t1 = sd_a[name]
        t2 = sd_b[name]
        if t1.dtype != t2.dtype:
            dtype_mismatch.append((name, str(t1.dtype), str(t2.dtype)))
            if len(dtype_mismatch) >= max_diffs:
                break
    for name, _, _ in dtype_mismatch:
        if name in common:
            common.remove(name)

    for name in common:
        t1 = sd_a[name]
        t2 = sd_b[name]
        if not torch.equal(t1, t2):
            value_mismatch.append(name)
            if len(value_mismatch) >= max_diffs:
                break

    if shape_mismatch:
        msgs.append(f"Shape mismatches ({len(shape_mismatch)}): {shape_mismatch[:max_diffs]}")
    if dtype_mismatch:
        msgs.append(f"Dtype mismatches ({len(dtype_mismatch)}): {dtype_mismatch[:max_diffs]}")
    if value_mismatch:
        msgs.append(f"Value mismatches ({len(value_mismatch)}): {value_mismatch[:max_diffs]}")

    ok = not (missing_in_b or extra_in_b or shape_mismatch or dtype_mismatch or value_mismatch)
    if ok:
        return True, "All tensors identical."
    summary = f"Summary: missing={len(missing_in_b)}, extra={len(extra_in_b)}, shape={len(shape_mismatch)}, dtype={len(dtype_mismatch)}, value={len(value_mismatch)}."
    return False, "\n".join(msgs + [summary])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two HF safetensors checkpoints (strict equality).")
    p.add_argument("ckpt_a", type=str)
    p.add_argument("ckpt_b", type=str)
    p.add_argument("--max-diffs", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    a = os.path.abspath(args.ckpt_a)
    b = os.path.abspath(args.ckpt_b)
    if not os.path.isdir(a) or not os.path.isdir(b):
        print("Both inputs must be directories.", file=sys.stderr)
        return 2
    try:
        ok, report = compare_checkpoints(a, b, max_diffs=args.max_diffs)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    if ok:
        print("OK: All tensors identical.")
        return 0
    print("NOT EQUAL:")
    print(report)
    return 1


if __name__ == "__main__":
    sys.exit(main())