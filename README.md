# Project Group 188

**Members:**
- Aidan Casselman (101277801)
- Kamal Yassin (101265070)

This repository contains the implementation of our Fireboy & Watergirl–inspired multi-agent reinforcement learning environment, along with training scaffolding for MAPPO.

Two agents (Fireboy and Watergirl) must co-operate in a grid-world of walls, switches, doors, hazards, and exits. Each agent sees only its own local observation, but they are trained jointly with a shared policy and a centralized critic.

---

## Repository Structure

- `env/`
    - `firewater_env.py`: core single-episode environment with grid logic and reward function.
    - `env_wrapper.py`: wraps multiple level files into a Gym-style vector env for training and evaluation.
    - `level_gen.py`: generates solvable levels from simple ASCII templates.
    - `level_split.py`: splits generated levels into `train/`, `val/`, `test/` splits.
    - `levels/`
        - `templates/`: template levels for input to the level generator.
        - `generated/`: auto-generated per-difficulty levels (output of `level_gen.py`).
        - `dataset/`: final split into `train/`, `val/`, `test/` (output of `level_split.py`)
- `RL/`
    - `networks.py`: convolutional encoder, actor heads, centralized critic.
    - `mappo_agent.py`: MAPPO logic (PPO update, advantage computation, etc.).
    - `buffer.py`: rollout buffer.
    - `train_mappo.py`: main training script.
    - `eval_policy.py`: evaluation + policy visualization.
- `checkpoints/`: saved PyTorch checkpoints.

---

## Setup

### Install dependencies (recommended to use a Python environment, we used Python 3.10.*):

General dependency install instructions (for any Python version):
```
pip install torch numpy pygame
```

Recommended install (using requirements.txt for Python 3.10.*):
```
pip install -r requirements.txt
```

> *Note:* If you have a compatible NVIDIA GPU, include your compatible CUDA version to the torch install.

---

## Level Generation

Templates are simple ASCII layouts with one level per block, stored under:
```
env/levels/templates
```

Naming convention:
- Any file named `X_templates.txt` becomes a difficulty called `X`.
Examples:
- `easy_templates.txt` -> `easy`
- `medium_templates.txt` -> `medium`
- `hard_templates.txt` -> `hard`
- etc.

Within each file:
- Levels are separated by one or more blank lines.
- Each level is a rectangular ASCII grid (however, levels must be square for the level generator to work properly).

For each templates level, the generator create 8 variants:
- 4 rotations (90 degrees each)
- For each rotation: original roles + swapped roles (F <-> G, f <-> g, L <-> W)

Run (from project root):
```
python -m env.level_gen --out env/levels/generated --templates-dir env/levels/templates --no-check
```

#### `level_gen.py` Arguments:
- `--out`: Root directory where per-difficulty folders will be created. Defaults to `env/levels/generated`.
- `--templates-dir`: Directory containing `*_templates.txt`. Defaults to `env/templates`.
- `--no-check`: If given, skips solvability checking (by default each variant is checked by a BFS that simulates both agents), it is recommended unless you are not confident your templates are valid. 
> **NOTE:** It is very slow if you leave default BFS checking on.

---

## Train/Val/Test Split of Levels

Once you have generated `env/levels/generated/<difficulty>/*.txt`, create the dataset split:
```
python -m env.level_split --in-root env/levels/generated --out-root env/levels/dataset --train-pct 0.8 --val-pct 0.1 --test-pct 0.1 --seed 188
```

What this does:
- Automatically discovers difficulties as any subdirectory of `--in-root` that:
    - is a directory, and
    - contains at least one `.txt` file.
- For each difficulty, it:
    - Lists all .txt files under `--in-root`.
    - Randomly shuffles them with the given `--seed`.
    - Splits into train/val/test according to the percentages.
    - Copies them into `--out-root`.

#### `level_split.py` Arguments:
- `--in-root`: Input directory containing generated difficulty subfolders.
- `--out-root`: Output directory for `train/`, `val/`, `test/`, splits.
- `--train-pct`: Fraction of levels per difficulty for train (default: 0.8).
- `--val-pct`: Fraction of levels per difficulty for val (default: 0.1).
- `--test-pct`: Fraction of levels per difficulty for test (default: 0.1).
- `--seed`: Random seed for shuffling (default: 188).

---

## Training
From the repo root:
```
python -m RL.train_mappo --out-name mappo_easy
```

### Resuming Training
To continue training from an existing checkpoint:
```
python -m RL.train_mappo --ckpt checkpoints/mappo_easy.pt
```

Behaviour:
- Loads encoder, both actors, critic, and optimizer state from `--ckpt`.
- Saves the new checkpoint with an incremented suffix:
    - `mappo_easy.py` -> `mappo_easy_0.pt`
    - next run: `mappo_easy_0.pt` -> `mappo_easy_1.pt`, etc.

### Optional Experimental Curriculum Training
If you want multi-stage curriculum not tied to easy/medium/hard difficulties, create folders like:
```
env/levels/curriculum/stage0/
env/levels/curriculum/stage1/
env/levels/curriculum/stage2/
```

Each `stageX` (or any name ending with a digit), must contain `.txt` levels.

`train_mappo.py` discovers these stages by the trailing digit, sorts by that digit, and schedules them evenly across training updates.

To use curriculum:
```
python -m RL.train_mappo --training-mode curriculum --out-name mappo_curriculum
```

#### `train_mappo.py` Arguments:
- `--training-mode`: Choose between default training, and experimental curriculum based training.
- `--ckpt`: Path to checkpoint `.pt` file to resume from. Otherwise, training starts from scratch
- `--updates`: Number of training updates to run in this call (default: 500).
- `--out-name`: Name of output ckpt (default: "mappo").

### Automatic Evaluation

After each `train_mappo` run, it will automatically evaluate it for 200 episodes in both the `train/` and `test/` levels, using a greedy policy evaluation method.

---

## Evaluating a Checkpoint
If you wish to manually evaluate an existing checkpoint, run this from the project root:
```
python -m RL.eval_policy --ckpt checkpoints/mappo.pt --difficulties easy --episodes 200 --max-steps 50 --test-split test
```

### Visualizing Episodes
To render a few episodes step-by-step, run this from the project root:
```
python -m RL.eval_policy --ckpt checkpoints/mappo.pt --difficulties easy --max-steps 25 --test-split test --visualize-episodes 3
```

This prints, for each of the `visualize-episodes` runs:
- The grid at each step.
- Chosen actions for Fire & Water.
- Reward and done flag.
- Final state and total return.

#### `eval_policy.py` Arguments:
- `--ckpt`: Path to checkpoint `.pt` file to evaluate.
- `--difficulties`: Which difficulties to evaluate with (default: ["easy"]).
- `--episodes`: Number of evaluation episodes (default: 200).
- `--max-steps`: Max steps per eval episode (default: 50).
- `--visualize-episodes`: If >0, runs this many greedy episodes with ASCII rendering.
- `--test-split`: Which split to test on (train/test/val) (default: "test").

---

## Additional Testing of Environment & Levels
`firewater_env.py` itself can be called to test the environment/levels with manual and scripted modes.

### Manual Testing
Run this from project root:
```
python -m env.firewater_env env/levels/testing/switch_door_block.txt --mode manual
```

You can then control each agent manually, using the following keyboard controls:

Fire (F):
    W A S D : up, left, down, right

Water (G):
    I J K L : up, left, down, right

You must press enter after entering a character to move to the next step. Leaving blank will make both agents *STAY*.

### Scripted Testing
Run this from project root:
```
python -m env.firewater_env env/levels/testing/switch_door_block.txt --mode scripted --script env/scripts/test_script.txt
```

This will cause the agents to act according to each line in the script, with the format being:
`fire_action water_action` for each line.

where actions are ints in [0,4]:
    `0=UP, 1=LEFT, 2=DOWN, 3=RIGHT, 4=STAY`

---

## Pygame Viewer (GUI Visualization)

In addition to ASCII visualization, we provide a simple Pygame-based viewer to inspect levels and watch trained policies interact with the environment.

> **Note:** This requires `pygame` (included in `requirements.txt`). If needed, install manually via:
```
pip install pygame
```

### 1. Single-Level Viewer (`env.ui_viewer`)

You can run a single level either **manually** (keyboard control) or using a **trained policy**.

#### Manual mode

From the project root:
```
python -m env.ui_viewer env/levels/dataset/train/easy/some_level.txt --mode manual
```

**Controls:**
- **Fire (F)**: `W` (up), `A` (left), `S` (down), `D` (right)  
- **Water (G)**: `I` (up), `J` (left), `K` (down), `L` (right)  
- **Quit**: `Q` or `Esc`  

The grid is drawn with coloured tiles for walls, agents, exits, switches, doors, etc., and updates every step.

#### Policy mode

To let a trained MAPPO policy play the level:
```
python -m env.ui_viewer env/levels/dataset/test/easy/some_level.txt --mode policy --policy-module RL.easy_demo_policy
```

Here:
- `--mode policy` tells the viewer to use a function `policy_fn(obs) -> (a_fire, a_water)`.
- `--policy-module` is a Python module path that exposes that `policy_fn` (see below).

### 2. Demo Over Multiple Levels (`env.demo_easy`)

To run a trained policy over **all** easy levels (or any glob pattern) one by one:
```
python -m env.demo_easy --policy-module RL.easy_demo_policy --pattern "env/levels/dataset/test/easy/*.txt" --max-steps 200
```

- `--pattern` is a glob for level paths (default: `env/levels/dataset/train/easy/*.txt`).
- For each level, a Pygame window opens, runs up to `max-steps`, then closes and moves on.

### 3. Example Policy Module (`RL/easy_demo_policy.py`)

We provide an example policy module that loads a checkpoint and exposes the required `policy_fn`:

- **File:** `RL/easy_demo_policy.py`  
- **Default checkpoint path:**

      CKPT_PATH = "checkpoints/mappo_easy_curriculum.pt"

- Internally, it:
  - Infers the observation shape from the first call to `policy_fn`.
  - Reconstructs the encoder, actors, and critic.
  - Loads weights from `CKPT_PATH`.
  - Runs a **greedy** action selection for both agents.

To use it with your own model, either:

- Save your best checkpoint as `checkpoints/mappo_easy_curriculum.pt`, **or**
- Edit `CKPT_PATH` in `RL/easy_demo_policy.py` to point to your chosen `.pt` file.

This setup lets you quickly **visually inspect** how the learned policy behaves on individual levels or across a whole batch of easy levels.

---

## Final Notes and Limitations

This codebase is still a work in progress and there are a few important caveats to keep in mind:

#### Curriculum scheduling is manual when not using curriculum mode
The function that decides which difficulties to train on over time is currently hard-coded (`difficulties_for_update` in `train_mappo.py`).

For different experiments (e.g., only easy, or easy → medium → hard), you may want to manually edit this function to adjust when new difficulties are introduced, or to restrict training to a subset of levels. A more principled, automated curriculum (e.g., based on success rate thresholds or level-specific statistics) is left as future work.

#### Navigation vs. true puzzle solving.
The current best models are primarily solving navigation-style easy levels: short corridors, simple paths, and basic coordination to reach exits. While the environment supports richer mechanics (switches, doors, blocks, hazards), our submitted models do not yet reliably solve the more complex “puzzle-like” medium and hard templates that require multi-step planning and cooperation (e.g., one agent holding a switch while the other passes through a door).

In other words, the project demonstrates that MAPPO can learn robust cooperative navigation policies in this environment, but full puzzle-solving behaviour (with long sequences of interdependent actions) remains an open direction. Extending the reward shaping, curriculum design, and possibly the network architecture to handle these harder levels is a natural next step beyond this submission.