"""
保存する必要のないckptディレクトリを移動するスクリプト。
デフォルトだと1000iter刻み以外を移動。(例外として最後のチェックポイントと1000iter未満は残す。)
"""

import os
from pathlib import Path

import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--min_iter", type=int, default=1000)
    parser.add_argument("--ckpt_iter", type=int, default=1000)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = load_args()

    if not args.dryrun:
        os.makedirs(args.dst_dir, exist_ok=True)

    # 各iterのckptディレクトリを取得
    src_dir = Path(args.src_dir)
    src_ckpt_dirs = sorted(src_dir.glob("iter_*/"))

    # 最後尾のckptは残す
    src_ckpt_dirs = src_ckpt_dirs[:-1]
    
    for src_ckpt_dir in src_ckpt_dirs:
        iter_num = int(src_ckpt_dir.name.split("_")[1])

        # min_iter未満は移動の対象外とする
        if iter_num < args.min_iter:
            continue

        # ckpt_iter(1000iter)の倍数は移動の対象外とする
        if iter_num % args.ckpt_iter == 0:
            continue

        mv_command = f"mv {src_ckpt_dir} {args.dst_dir}"

        if args.dryrun:
            # dryrunの場合は実行コマンドを表示する(実際の移動はしない。)
            print(mv_command)
            continue
        else:
            # 実際にckptディレクトリを移動する
            os.system(mv_command)


if __name__ == "__main__":
    main()

