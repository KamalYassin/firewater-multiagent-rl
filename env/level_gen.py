import os
import argparse
import copy
from collections import deque
from typing import List, Tuple, Optional, Set

from .firewater_env import FireWaterEnv, parse_level_from_string, NUM_ACTIONS

TEMPLATE_SUFFIX = "_templates.txt"


# --------------------------------------------------------------------------------
# Utilities to load templates
# --------------------------------------------------------------------------------

def discover_template_files(templates_dir: str) -> dict:
    """
    Scan templates_dir for files named '<name>_templates.txt' and
    return a mapping: {name: full_path}.

    Example:
      supereasy_templates.txt -> {'supereasy': '/.../supereasy_templates.txt'}
    """
    files = {}
    if not os.path.isdir(templates_dir):
        print(f"[WARN] Templates dir does not exist: {templates_dir}")
        return files

    for fname in os.listdir(templates_dir):
        if not fname.endswith(TEMPLATE_SUFFIX):
            continue
        diff_name = fname[: -len(TEMPLATE_SUFFIX)]  # strip suffix
        full_path = os.path.join(templates_dir, fname)
        files[diff_name] = full_path

    return files


def load_templates(path: str) -> List[str]:
    """
    Load ASCII templates from a text file.

    - Templates are separated by one or more blank lines.
    - Each template is a rectangular ASCII grid.
    """
    templates: List[str] = []
    current: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")

            if line.strip() == "":
                if current:
                    templates.append("\n".join(current))
                    current = []
            else:
                current.append(line)

    if current:
        templates.append("\n".join(current))

    return templates


def str_to_grid(level_str: str) -> List[List[str]]:
    return [list(row) for row in level_str.splitlines() if row]


def grid_to_str(grid: List[List[str]]) -> str:
    return "\n".join("".join(row) for row in grid)


# --------------------------------------------------------------------------------
# Transformations: rotate & swap roles
# --------------------------------------------------------------------------------

def rotate_grid(grid: List[List[str]], k: int) -> List[List[str]]:
    """
    Rotate grid by k * 90 degrees clockwise.
    k in {0,1,2,3}.
    Works for any rectangular grid (H x W).
    """
    k = k % 4
    if k == 0:
        return [row[:] for row in grid]

    H = len(grid)
    W = len(grid[0])

    def rot90(g: List[List[str]]) -> List[List[str]]:
        # (r, c) -> (c, W-1-r)
        return [[g[H - 1 - r][c] for r in range(H)] for c in range(W)]

    out = [row[:] for row in grid]
    for _ in range(k):
        out = rot90(out)
        H, W = len(out), len(out[0])
    return out


ROLE_SWAP_MAP = {
    "F": "G",
    "G": "F",
    "f": "g",
    "g": "f",
    "L": "W",
    "W": "L",
}


def swap_roles_in_grid(grid: List[List[str]]) -> List[List[str]]:
    """
    Swap Fire/Water roles in-place:
      F <-> G, f <-> g, L <-> W
    Everything else unchanged.
    """
    H = len(grid)
    W = len(grid[0])
    new_grid: List[List[str]] = [[ch for ch in row] for row in grid]

    for y in range(H):
        for x in range(W):
            ch = grid[y][x]
            if ch in ROLE_SWAP_MAP:
                new_grid[y][x] = ROLE_SWAP_MAP[ch]
    return new_grid


# --------------------------------------------------------------------------------
# Solvability
# --------------------------------------------------------------------------------

def _state_key_from_env(env: FireWaterEnv) -> Tuple:
    """
    Produce a hashable key representing the logical game state,
    so BFS can avoid revisiting states.

    We include:
      - fire position
      - water position
      - sorted block positions

    The base grid is static for a given level, so we don't need it in the key.
    """
    st = env.state
    if st is None:
        raise RuntimeError("FireWaterEnv.state is None during solvability check")

    fire_pos = st.fire_pos
    water_pos = st.water_pos
    blocks_sorted = tuple(sorted(st.blocks))

    return (fire_pos, water_pos, blocks_sorted)


def is_level_solvable(
    level_str: str,
    max_depth: int = 80,
    max_states: int = 50_000,
) -> Tuple[bool, Optional[int]]:
    """
    BFS over joint actions (fire, water) to see if both can reach exits.
    Returns (solvable, min_steps).

    - max_depth: cutoff on number of steps in a solution path
    - max_states: safety bound on number of visited states
    """
    # Parse ASCII into LevelSpec and build env
    lvl = parse_level_from_string(level_str)
    env = FireWaterEnv(lvl, max_steps=max_depth + 5)

    env.reset()
    start_key = _state_key_from_env(env)

    visited: Set[Tuple] = {start_key}
    q = deque([(copy.deepcopy(env), 0)])
    best: Optional[int] = None

    while q:
        cur_env, depth = q.popleft()
        if depth >= max_depth:
            continue

        for a_fire in range(NUM_ACTIONS):
            for a_water in range(NUM_ACTIONS):
                new_env = copy.deepcopy(cur_env)
                _, _, done, info = new_env.step(a_fire, a_water)

                key = _state_key_from_env(new_env)
                if key in visited:
                    continue
                visited.add(key)

                if done and info.get("success", False):
                    steps = depth + 1
                    if best is None or steps < best:
                        best = steps
                    return True, best

                if len(visited) >= max_states:
                    # Give up – treat as unsolvable under the given bounds
                    return False, best

                q.append((new_env, depth + 1))

    return False, best


def check_solvable(level_str: str,
                   max_depth: int = 80,
                   max_states: int = 50_000) -> bool:
    """
    Simple boolean wrapper used by the generator.

    Returns True if a solution is found within (max_depth, max_states), else False.
    """
    solvable, _ = is_level_solvable(
        level_str,
        max_depth=max_depth,
        max_states=max_states,
    )
    return solvable

# --------------------------------------------------------------------------------
# Main generation logic
# --------------------------------------------------------------------------------

def generate_variants_for_template(level_str: str) -> List[str]:
    """
    Given one ASCII template, produce up to 8 variants:
      - 4 rotations (0, 90, 180, 270)
      - For each rotation: original roles + swapped roles
    """
    base_grid = str_to_grid(level_str)

    variants: List[str] = []
    for rot_k in range(4):
        rotated = rotate_grid(base_grid, rot_k)

        # Original roles
        variants.append(grid_to_str(rotated))

        # Swapped roles
        swapped = swap_roles_in_grid(rotated)
        variants.append(grid_to_str(swapped))

    return variants


def generate_from_templates(
    templates_path: str,
    difficulty: str,
    out_dir: str,
    check: bool = True,
) -> None:
    """
    For a given difficulty and template file:
      - load templates
      - generate 8 variants each
      - check solvability
      - save as .txt files under out_dir/<difficulty>/
    """
    templates = load_templates(templates_path)
    print(f"[{difficulty.upper()}] Loaded {len(templates)} base templates from {templates_path}")

    diff_dir = os.path.join(out_dir, difficulty)
    os.makedirs(diff_dir, exist_ok=True)

    total_written = 0
    skipped = 0

    for idx, tmpl in enumerate(templates):
        tmpl_variants = generate_variants_for_template(tmpl)

        for v_ix, variant in enumerate(tmpl_variants):
            if check:
                if not check_solvable(variant):
                    skipped += 1
                    continue

            fname = f"{difficulty}_t{idx:02d}_v{v_ix:02d}.txt"
            fpath = os.path.join(diff_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(variant + "\n")

            total_written += 1

    print(
        f"[{difficulty.upper()}] Wrote {total_written} variants"
        + (f" (skipped {skipped} unsolvable)" if check else "")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Template-based level generator (rotations + role swaps)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "levels/generated"),
        help="Output directory root for generated levels.",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "templates"),
        help="Directory containing templates.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Disable solvability checks (trust templates).",
    )

    args = parser.parse_args()
    out_root = args.out
    templates_dir = args.templates_dir
    do_check = not args.no_check

    os.makedirs(out_root, exist_ok=True)

    files = discover_template_files(templates_dir)

    if not files:
        print(f"[WARN] No '*{TEMPLATE_SUFFIX}' files found in {templates_dir}. Nothing to do.")
        return
    
    print("[INFO] Found template sets:")
    for difficulty, path in files.items():
        print(f"  - {difficulty}: {path}")

    for difficulty, path in files.items():
        if not os.path.exists(path):
            print(f"[WARN] Template file not found for {difficulty}: {path}. Skipping.")
            continue

        generate_from_templates(
            templates_path=path,
            difficulty=difficulty,
            out_dir=out_root,
            check=do_check,
        )

    print("\n✓ Template-based level generation complete.")


if __name__ == "__main__":
    main()
