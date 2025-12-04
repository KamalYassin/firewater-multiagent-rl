import os
import argparse
import random
import shutil
from typing import List, Dict

DIFFICULTIES = ["easy", "medium", "hard"]


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def list_level_files(root: str, difficulty: str) -> List[str]:
    """Return list of `.txt` level files under root/difficulty/."""
    diff_dir = os.path.join(root, difficulty)
    if not os.path.isdir(diff_dir):
        print(f"[WARN] Directory missing for difficulty '{difficulty}': {diff_dir}")
        return []
    return [
        os.path.join(diff_dir, f)
        for f in os.listdir(diff_dir)
        if f.lower().endswith(".txt")
    ]


def stratified_split(
    files: List[str],
    train_pct: float,
    val_pct: float,
    test_pct: float,
    rng: random.Random,
) -> Dict[str, List[str]]:
    n = len(files)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    rng.shuffle(files)

    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    n_test = n - n_train - n_val

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def copy_split(
    split_map: Dict[str, Dict[str, List[str]]],
    out_root: str,
) -> None:
    """
    Copy files into out_root/train|val|test/easy|medium|hard.

    split_map has shape:
      split_map[split][difficulty] = list of file paths
    where split in {"train","val","test"} and difficulty in DIFFICULTIES.
    """
    for split in ["train", "val", "test"]:
        for diff in DIFFICULTIES:
            files = split_map[split].get(diff, [])
            if not files:
                continue

            target_dir = os.path.join(out_root, split, diff)
            os.makedirs(target_dir, exist_ok=True)

            for src in files:
                dst = os.path.join(target_dir, os.path.basename(src))
                shutil.copy2(src, dst)


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split levels into train/val/test stratified by difficulty."
    )
    parser.add_argument(
        "--in-root",
        required=True,
        type=str,
        help="Directory containing easy/, medium/, hard/ subfolders.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=str,
        help="Output directory root.",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.8,
        help="Fraction of levels per difficulty for train (default: 0.8).",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.1,
        help="Fraction of levels per difficulty for val (default: 0.1).",
    )
    parser.add_argument(
        "--test-pct",
        type=float,
        default=0.1,
        help="Fraction of levels per difficulty for test (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=188,
        help="Random seed for shuffling (default: 188).",
    )

    args = parser.parse_args()

    total_pct = args.train_pct + args.val_pct + args.test_pct
    if abs(total_pct - 1.0) > 1e-6:
        raise ValueError("train_pct + val_pct + test_pct must sum to 1.0")

    rng = random.Random(args.seed)

    print(f"\nLoading levels from: {args.in_root}\n")

    # list files per difficulty
    available: Dict[str, List[str]] = {}
    for diff in DIFFICULTIES:
        files = list_level_files(args.in_root, diff)
        available[diff] = files
        print(f"{diff.capitalize()}: {len(files)} levels")

    # stratified split per difficulty
    split: Dict[str, Dict[str, List[str]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for diff in DIFFICULTIES:
        files = available[diff]
        parts = stratified_split(files, args.train_pct, args.val_pct, args.test_pct, rng)
        split["train"][diff] = parts["train"]
        split["val"][diff] = parts["val"]
        split["test"][diff] = parts["test"]

    # copy to out_root/train|val|test/easy|medium|hard
    copy_split(split, args.out_root)

    print(f"\n✓ Split complete! Output saved to: {args.out_root}\n")
    for split_name in ["train", "val", "test"]:
        print(f"{split_name}:")
        for diff in DIFFICULTIES:
            count = len(split[split_name].get(diff, []))
            print(f"  {diff}: {count} levels")
    print("")


if __name__ == "__main__":
    main()
