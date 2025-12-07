import argparse
import glob
import os
from importlib import import_module

from .ui_viewer import run_viewer


def main():
    parser = argparse.ArgumentParser(
        description="Run trained policy on all easy levels one by one."
    )
    parser.add_argument(
        "--pattern",
        default="env/levels/dataset/train/easy/*.txt",
        help="Glob pattern for levels (default: all easy train levels).",
    )
    parser.add_argument(
        "--policy-module",
        required=True,
        help="Module path with policy_fn(obs)->(a_fire,a_water), e.g. 'RL.easy_demo_policy'.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per level before moving on.",
    )
    args = parser.parse_args()

    mod = import_module(args.policy_module)
    if not hasattr(mod, "policy_fn"):
        raise RuntimeError(f"Module {args.policy_module!r} has no function 'policy_fn'")
    policy_fn = getattr(mod, "policy_fn")

    level_paths = sorted(glob.glob(args.pattern))
    if not level_paths:
        raise RuntimeError(f"No levels found for pattern: {args.pattern}")

    print(f"Found {len(level_paths)} levels.")
    for idx, lvl_path in enumerate(level_paths, start=1):
        print(f"\n=== Level {idx}/{len(level_paths)}: {os.path.basename(lvl_path)} ===")
        run_viewer(lvl_path, max_steps=args.max_steps, policy_fn=policy_fn)


if __name__ == "__main__":
    main()
