from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable
import numpy as np
import sys
import argparse
from importlib import import_module


# --------------------------------------------------------------------------------
# Tile / object conventions
# --------------------------------------------------------------------------------

WALL = '#'
FLOOR = '.'
LAVA = 'L'
WATER = 'W'
BLOCK = 'X'

FIRE_SPAWN = 'F'
WATER_SPAWN = 'G'
FIRE_EXIT = 'f'
WATER_EXIT = 'g'

SWITCH_CHARS = ['1', '2', '3', '4', '5']
DOOR_CHARS = ['A', 'B', 'C', 'D', 'E']
SWITCH_TO_DOOR = {s: d for s, d in zip(SWITCH_CHARS, DOOR_CHARS)}
DOOR_TO_SWITCH = {d: s for s, d in SWITCH_TO_DOOR.items()}


# --------------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------------

# 0: UP, 1: LEFT, 2: DOWN, 3: RIGHT, 4: STAY
ACTION_UP = 0
ACTION_LEFT = 1
ACTION_DOWN = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_LEFT: (-1, 0),
    ACTION_DOWN: (0, 1),
    ACTION_RIGHT: (1, 0),
    ACTION_STAY: (0, 0),
}

NUM_ACTIONS = len(ACTION_DELTAS)


# --------------------------------------------------------------------------------
# Level parsing
# --------------------------------------------------------------------------------

@dataclass
class LevelSpec:
    width: int
    height: int
    base_grid: np.ndarray  # W x H of chars (static tiles: walls, floor, hazards, switches, doors)
    fire_spawn: Optional[Tuple[int, int]]
    water_spawn: Optional[Tuple[int, int]]
    fire_exit: Optional[Tuple[int, int]]
    water_exit: Optional[Tuple[int, int]]
    blocks: List[Tuple[int, int]]


def _normalize_lines(lines: Iterable[str]) -> List[str]:
    stripped = [ln.rstrip('\n') for ln in lines]
    # drop leading/trailing, completely empty lines
    while stripped and stripped[0] == '':
        stripped.pop(0)
    while stripped and stripped[-1] == '':
        stripped.pop()
    return stripped


def parse_level_from_string(level_str: str) -> LevelSpec:
    """
    Parse a level from a multi-line string.

    Uses the conventions:
      F, G, f, g, X, 1-5, A-E, #, ., L, W
    """
    lines = _normalize_lines(level_str.splitlines())
    if not lines:
        raise ValueError("Empty level string")

    height = len(lines)
    width = max(len(ln) for ln in lines)

    # pad lines to same width with walls
    padded_lines = [ln.ljust(width, WALL) for ln in lines]

    base_grid = np.full((height, width), FLOOR, dtype='<U1')

    fire_spawn = None
    water_spawn = None
    fire_exit = None
    water_exit = None
    blocks: List[Tuple[int, int]] = []

    for y, row in enumerate(padded_lines):
        for x, ch in enumerate(row):
            if ch == FIRE_SPAWN:
                if fire_spawn is not None:
                    raise ValueError("Multiple Fire spawns not supported")
                fire_spawn = (x, y)
                base_grid[y, x] = FLOOR
            elif ch == WATER_SPAWN:
                if water_spawn is not None:
                    raise ValueError("Multiple Water spawns not supported")
                water_spawn = (x, y)
                base_grid[y, x] = FLOOR
            elif ch == FIRE_EXIT:
                fire_exit = (x, y)
                base_grid[y, x] = FLOOR
            elif ch == WATER_EXIT:
                water_exit = (x, y)
                base_grid[y, x] = FLOOR
            elif ch == BLOCK:
                blocks.append((x, y))
                base_grid[y, x] = FLOOR
            elif ch in (WALL, FLOOR, LAVA, WATER) or ch in SWITCH_CHARS or ch in DOOR_CHARS:
                base_grid[y, x] = ch
            else:
                raise ValueError(f"Unknown char in level: {repr(ch)} at ({x},{y})")

    return LevelSpec(
        width=width,
        height=height,
        base_grid=base_grid,
        fire_spawn=fire_spawn,
        water_spawn=water_spawn,
        fire_exit=fire_exit,
        water_exit=water_exit,
        blocks=blocks,
    )


def parse_level_from_file(path: str) -> LevelSpec:
    with open(path, 'r') as f:
        text = f.read()
    return parse_level_from_string(text)


# --------------------------------------------------------------------------------
# Game state + core rules engine
# --------------------------------------------------------------------------------

@dataclass
class GameState:
    base_grid: np.ndarray          # static tiles (H x W)
    fire_pos: Optional[Tuple[int, int]]
    water_pos: Optional[Tuple[int, int]]
    fire_exit: Optional[Tuple[int, int]]
    water_exit: Optional[Tuple[int, int]]
    blocks: List[Tuple[int, int]]  # list of (x,y)
    steps_taken: int
    max_steps: int


class FireWaterEnv:
    """
    Core environment.

    Multi-agent convention:

      - obs = env.reset(level_spec)
      - for each step:
          obs, reward, done, info = env.step(action_fire, action_water)

      - obs is dict with keys 'fire' and 'water' (if that agent exists),
        each a (C, H, W) float32 tensor.
    """

    def __init__(self, level: LevelSpec, max_steps: int = 50,
                 step_penalty: float = -0.01,
                 success_reward: float = 5.0,
                 death_penalty: float = -1.0,
                 hazards_mode: str = "wall",
                 exit_partial_reward: float = 0.0,
                 switch_reward: float = 0.0,
                 push_block_reward: float = 0.0,
                 dist_coef: float = 0.1,
                 move_reward: float = 0.0,
                 blocked_move_penalty: float = -0.1,
                 stagnation_penalty: float = -0.02,
                 stay_penalty: float = -0.02):
        self.level = level
        self.height = level.height
        self.width = level.width
        self.max_steps = max_steps

        # reward config
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.death_penalty = death_penalty

        # shaping params
        self.exit_partial_reward = exit_partial_reward
        self.switch_reward = switch_reward
        self.push_block_reward = push_block_reward
        self.dist_coef = dist_coef
        self.move_reward = move_reward
        self.blocked_move_penalty = blocked_move_penalty
        self.stagnation_penalty = stagnation_penalty
        self.stay_penalty = stay_penalty

        # distance shaping state
        self.last_fire_dist = None
        self.last_water_dist = None

        self.hazards_mode = hazards_mode

        # state
        self.state: Optional[GameState] = None

        # does each agent exist?
        self.has_fire = level.fire_spawn is not None
        self.has_water = level.water_spawn is not None

        # tracking for one-time rewards
        self.fire_reached_exit_once = False
        self.water_reached_exit_once = False
        self.fire_switches_seen = set()
        self.water_switches_seen = set()
        self.block_switches_seen = set()

        self.reset()

    # ---------------------------------- public API for RL & testing -------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        # reset to initial state for this level
        st = GameState(
            base_grid=self.level.base_grid.copy(),
            fire_pos=self.level.fire_spawn,
            water_pos=self.level.water_spawn,
            fire_exit=self.level.fire_exit,
            water_exit=self.level.water_exit,
            blocks=list(self.level.blocks),
            steps_taken=0,
            max_steps=self.max_steps,
        )

        # initialize distance shaping baselines
        self.last_fire_dist = None
        self.last_water_dist = None
        if self.has_fire and st.fire_exit is not None:
            self.last_fire_dist = self._manhattan_dist(st.fire_pos, st.fire_exit)
        if self.has_water and st.water_exit is not None:
            self.last_water_dist = self._manhattan_dist(st.water_pos, st.water_exit)

        self.fire_reached_exit_once = False
        self.water_reached_exit_once = False
        self.fire_switches_seen = set()
        self.water_switches_seen = set()
        self.block_switches_seen = set()

        self.state = st
        return self._build_observation()

    def step(self, action_fire: Optional[int], action_water: Optional[int]):
        """
        One environment step.

        action_fire / action_water are ints 0-4 or None if agent doesn't exist.
        Returns obs, reward (shared scalar), done, info.
        """
        assert self.state is not None, "Call reset() before step()"
        st = self.state

        # normalize actions for missing agents
        if not self.has_fire:
            action_fire = ACTION_STAY
        if not self.has_water:
            action_water = ACTION_STAY

        # apply actions and compute next state
        next_state, events = self._apply_actions(st, action_fire, action_water)

        done, success, death = self._compute_done_and_flags(next_state, events)

        reward = self._compute_reward(
            st=next_state,
            done=done,
            success=success,
            death=death,
            events=events
        )

        next_state.steps_taken += 1
        if next_state.steps_taken >= next_state.max_steps:
            done = True

        self.state = next_state
        obs = self._build_observation()

        info = {
            "success": success,
            "events": events,
        }
        if self.hazards_mode == "death":
            info["death"] = death

        return obs, reward, done, info

    # ---------------------------------- core logic -------------------------------------

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _get_static_tile(self, st: GameState, pos: Tuple[int, int]) -> str:
        x, y = pos
        if not self._in_bounds(x, y):
            return WALL
        return st.base_grid[y, x]

    def _blocks_set(self, st: GameState) -> set:
        return set(st.blocks)

    def _active_switch_ids(self, st: GameState) -> set:
        """
        Return set of switch chars '1'..'5' that currently have an agent or block on them.
        Doors for these switches are considered open.
        """
        active = set()
        positions_to_check: List[Tuple[int, int]] = []

        if st.fire_pos is not None:
            positions_to_check.append(st.fire_pos)
        if st.water_pos is not None:
            positions_to_check.append(st.water_pos)
        positions_to_check.extend(st.blocks)

        for (x, y) in positions_to_check:
            tile = st.base_grid[y, x]
            if tile in SWITCH_CHARS:
                active.add(tile)
        return active

    def _door_is_open(self, st: GameState, pos: Tuple[int, int]) -> bool:
        tile = self._get_static_tile(st, pos)
        if tile not in DOOR_CHARS:
            return False
        switch_char = DOOR_TO_SWITCH[tile]
        return switch_char in self._active_switch_ids(st)
    
    def _manhattan_dist(self, pos: Optional[Tuple[int, int]],
                    exit_pos: Optional[Tuple[int, int]]) -> Optional[int]:
        """Manhattan distance between pos and exit_pos, or None if missing."""
        if pos is None or exit_pos is None:
            return None
        return abs(pos[0] - exit_pos[0]) + abs(pos[1] - exit_pos[1])

    def _is_passable_for_agent(self, st: GameState, pos: Tuple[int, int], agent_is_fire: bool) -> bool:
        if not self._in_bounds(*pos):
            return False

        x, y = pos
        tile = st.base_grid[y, x]

        # walls
        if tile == WALL:
            return False

        # closed doors behave as walls
        if tile in DOOR_CHARS and not self._door_is_open(st, pos):
            return False

        # hazards
        if tile == LAVA and not agent_is_fire:
            return False
        if tile == WATER and agent_is_fire:
            return False

        # blocks and other agent considered impassable here;
        # pushing is handled separately in movement function.
        return True

    def _is_passable_for_block(self, st: GameState, pos: Tuple[int, int]) -> bool:
        """
        Decide where a block can be pushed.
        I.e.: floor, exits, switches, open doors.
        Hazards and walls are forbidden for blocks.
        """
        if not self._in_bounds(*pos):
            return False

        x, y = pos
        tile = st.base_grid[y, x]

        if tile == WALL:
            return False

        if tile in (LAVA, WATER):
            return True

        # doors: only passable if open
        if tile in DOOR_CHARS and not self._door_is_open(st, pos):
            return False

        # otherwise OK (floor, switch, exit locations are just floor in base grid)
        return True

    def _apply_single_agent_move(
        self,
        st: GameState,
        pos: Optional[Tuple[int, int]],
        action: int,
        other_agent_pos: Optional[Tuple[int, int]],
        blocks: List[Tuple[int, int]],
        agent_is_fire: bool
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]], Dict]:
        """
        Apply movement for one agent (Fire or Water), possibly pushing blocks.
        Returns: new_pos, new_blocks, events
        """
        events: Dict = {}
        if pos is None:
            # agent absent in this level
            return None, blocks, events

        dx, dy = ACTION_DELTAS.get(action, (0, 0))
        target = (pos[0] + dx, pos[1] + dy)

        # staying in place
        if action == ACTION_STAY:
            events["move"] = "stay"
            return pos, blocks, events

        # can't move out of bounds
        if not self._in_bounds(*target):
            events["move"] = "blocked_wall_or_oob"
            return pos, blocks, events

        blocks_set = set(blocks)

        # if target is occupied by other agent, block movement
        if other_agent_pos is not None and target == other_agent_pos:
            events["move"] = "blocked_other_agent"
            return pos, blocks, events

        # if target has a block, try pushing
        if target in blocks_set:
            # position behind the block
            push_target = (target[0] + dx, target[1] + dy)

            # can't push out of bounds or into another block or into other agent
            if (not self._in_bounds(*push_target)
                or push_target in blocks_set
                or (other_agent_pos is not None and push_target == other_agent_pos)
            ):
                events["move"] = "blocked_block_cannot_push"
                return pos, blocks, events

            # check block passability at push target
            if not self._is_passable_for_block(st, push_target):
                events["move"] = "blocked_block_impassable"
                return pos, blocks, events

            # check agent passability at block's current target
            if not self._is_passable_for_agent(st, target, agent_is_fire):
                events["move"] = "blocked_agent_impassable"
                return pos, blocks, events

            # perform push: move block to push_target, agent to target
            new_blocks = []
            for b in blocks:
                if b == target:
                    new_blocks.append(push_target)
                else:
                    new_blocks.append(b)
            events["move"] = "pushed_block"
            return target, new_blocks, events

        # no block; check tile passability
        if not self._is_passable_for_agent(st, target, agent_is_fire):
            events["move"] = "blocked_tile"
            return pos, blocks, events

        # plain move
        events["move"] = "moved"
        return target, blocks, events

    def _apply_actions(self, st: GameState, action_fire: int, action_water: int):
        """
        Apply both agents' actions, sequentially (Fire then Water),
        updating positions and blocks.

        Returns: next_state, events dict
        """
        # copy state
        next_st = GameState(
            base_grid=st.base_grid,
            fire_pos=st.fire_pos,
            water_pos=st.water_pos,
            fire_exit=st.fire_exit,
            water_exit=st.water_exit,
            blocks=list(st.blocks),
            steps_taken=st.steps_taken,
            max_steps=st.max_steps,
        )

        events = {
            "fire": {},
            "water": {},
        }

        # Fire moves first
        next_fire_pos, next_blocks, fire_events = self._apply_single_agent_move(
            next_st,
            next_st.fire_pos,
            action_fire,
            next_st.water_pos,
            next_st.blocks,
            agent_is_fire=True
        )
        next_st.fire_pos = next_fire_pos
        next_st.blocks = next_blocks
        events["fire"] = fire_events

        # Water moves second, seeing Fire's updated position and blocks
        next_water_pos, next_blocks2, water_events = self._apply_single_agent_move(
            next_st,
            next_st.water_pos,
            action_water,
            next_st.fire_pos,
            next_st.blocks,
            agent_is_fire=False
        )
        next_st.water_pos = next_water_pos
        next_st.blocks = next_blocks2
        events["water"] = water_events

        # After movement, check for hazard deaths (optional, most likely not going to be used)
        if self.hazards_mode == "death":
            death_events = self._check_hazards(next_st)
            events.update(death_events)

        return next_st, events

    def _check_hazards(self, st: GameState) -> Dict:
        # check hazards and mark death if any agent stands on its bad tile
        events = {}
        fire_dead = False
        water_dead = False

        if st.fire_pos is not None:
            x, y = st.fire_pos
            tile = st.base_grid[y, x]
            if tile == WATER:  # Fire dies on water
                fire_dead = True

        if st.water_pos is not None:
            x, y = st.water_pos
            tile = st.base_grid[y, x]
            if tile == LAVA:  # Water dies on lava
                water_dead = True

        events["fire_dead"] = fire_dead
        events["water_dead"] = water_dead
        return events

    def _compute_done_and_flags(self, st: GameState, events: Dict) -> Tuple[bool, bool, bool]:
        """
        Return (done, success, death).

        success:
            - For every existing agent:
                - that agent has an exit, AND
                - its position equals its exit.
        death:
            - Any agent died on a hazard (probably not going to be used).
        """
        death = events.get("fire_dead", False) or events.get("water_dead", False)

        def agent_success(has_agent: bool,
                            pos: Optional[Tuple[int, int]],
                            exit_pos: Optional[Tuple[int, int]]) -> bool:
            # If agent doesn't exist in this level, it's not required for success
            if not has_agent:
                return True
            # If an agent exists but has no exit, level can never be successfully completed
            if exit_pos is None:
                return False
            # Normal case: require agent to be standing on its exit
            return pos == exit_pos

        fire_success = agent_success(self.has_fire, st.fire_pos, st.fire_exit)
        water_success = agent_success(self.has_water, st.water_pos, st.water_exit)

        any_agent = self.has_fire or self.has_water
        success = any_agent and fire_success and water_success

        done = success or death

        if not any_agent:
            done = True
            success = True

        return done, success, death
    
    def _is_switch_tile(self, tile: str) -> bool:
        return tile.isdigit()

    def _compute_reward(self,
                        st: GameState,
                        done: bool,
                        success: bool,
                        death: bool,
                        events: Dict) -> float:
        """
        Compute reward with:
          - base step penalty
          - optional distance-based shaping toward exits
          - terminal success/death bonuses
        """
        # base step cost
        r = self.step_penalty

        shaping_coef = self.dist_coef

        # --------- partial rewards for exit progress ---------
        # One-time bonus when each agent reaches its exit for the first time
        for agent_key, pos, exit_pos, flag_name in [
            ("fire", st.fire_pos, st.fire_exit, "fire_reached_exit_once"),
            ("water", st.water_pos, st.water_exit, "water_reached_exit_once"),
        ]:
            if pos is None or exit_pos is None:
                continue
            if getattr(self, flag_name):
                continue  # already got bonus this episode
            if pos == exit_pos:
                r += self.exit_partial_reward
                setattr(self, flag_name, True)

        # --------- reward switches & blocks ---------
        # Reward an agent the first time it stands on each switch tile
        for agent_key, pos, seen_attr in [
            ("fire", st.fire_pos, "fire_switches_seen"),
            ("water", st.water_pos, "water_switches_seen"),
        ]:
            if pos is None:
                continue
            x, y = pos
            tile = st.base_grid[y, x]
            if tile in SWITCH_CHARS:
                seen = getattr(self, seen_attr)
                if pos not in seen:
                    r += self.switch_reward
                    seen.add(pos)

        # Reward blocks being pushed onto a new switch tile
        for (bx, by) in st.blocks:
            tile = st.base_grid[by, bx]
            if tile in SWITCH_CHARS:
                if (bx, by) not in self.block_switches_seen:
                    r += 0.5 * self.switch_reward
                    self.block_switches_seen.add((bx, by))

        # Reward actually pushing blocks at all
        for agent_key in ["fire", "water"]:
            move_evt = events.get(agent_key, {}).get("move", "")
            if move_evt == "pushed_block":
                r += self.push_block_reward

        # FIRE
        if self.has_fire and st.fire_exit is not None:
            d_new = self._manhattan_dist(st.fire_pos, st.fire_exit)
            if d_new is not None and self.last_fire_dist is not None:
                delta = self.last_fire_dist - d_new
                move_evt = events.get("fire", {}).get("move", "")
                if d_new == self.last_fire_dist and move_evt != "stay" and st.fire_pos != st.fire_exit:
                    r += self.stagnation_penalty
                elif delta > 0:
                    r += shaping_coef * delta
            self.last_fire_dist = d_new

        # WATER
        if self.has_water and st.water_exit is not None:
            d_new = self._manhattan_dist(st.water_pos, st.water_exit)
            if d_new is not None and self.last_water_dist is not None:
                delta = self.last_water_dist - d_new
                move_evt = events.get("water", {}).get("move", "")
                if d_new == self.last_water_dist and move_evt != "stay" and st.water_pos != st.water_exit:
                    r += self.stagnation_penalty
                elif delta > 0:
                    r += shaping_coef * delta
            self.last_water_dist = d_new

        # --------- penalize bad moves ---------
        for agent_key in ["fire", "water"]:
            move_evt = events.get(agent_key, {}).get("move", "")
            if move_evt.startswith("blocked"):
                r += self.blocked_move_penalty  # e.g. -0.1

        # --------- terminal success/death bonuses ---------
        if done:
            if success:
                r += self.success_reward
            if death:
                r += self.death_penalty

        # ----- STAY penalty only if not on exit or switch -----
        for agent_key, pos, exit_pos in [
            ("fire", st.fire_pos, st.fire_exit),
            ("water", st.water_pos, st.water_exit)
        ]:
            move_evt = events.get(agent_key, {}).get("move", "")

            # Agent exists AND tried to stay still
            if move_evt == "stay" and pos is not None:

                tile = st.base_grid[pos[1], pos[0]]
                on_exit = (exit_pos is not None and pos == exit_pos)
                on_switch = tile.isdigit()

                # Penalize only if they are NOT doing something useful
                if not on_exit and not on_switch:
                    r += self.stay_penalty

        return r


    # ---------------------------------- Observation builder -------------------------------------

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """
        Build full-grid observations for each agent.

        Returns dict:
          {
            "fire":  (C,H,W) float32  if fire exists,
            "water": (C,H,W) float32  if water exists,
          }

        Channels (C=11):
          0: walls
          1: floor
          2: lava
          3: water
          4: switches (any 1-5)
          5: doors (all, regardless open/closed)
          6: blocks
          7: fire position
          8: water position
          9: fire exit
          10: water exit
        """
        assert self.state is not None
        st = self.state
        H, W = st.base_grid.shape
        C = 11
        base = np.zeros((C, H, W), dtype=np.float32)

        # static tiles
        for y in range(H):
            for x in range(W):
                ch = st.base_grid[y, x]
                if ch == WALL:
                    base[0, y, x] = 1.0
                elif ch == FLOOR:
                    base[1, y, x] = 1.0
                elif ch == LAVA:
                    base[2, y, x] = 1.0
                elif ch == WATER:
                    base[3, y, x] = 1.0
                elif ch in SWITCH_CHARS:
                    base[4, y, x] = 1.0
                elif ch in DOOR_CHARS:
                    base[5, y, x] = 1.0
                else:
                    pass

        # blocks
        for (x, y) in st.blocks:
            base[6, y, x] = 1.0

        # fire pos
        if st.fire_pos is not None:
            x, y = st.fire_pos
            base[7, y, x] = 1.0

        # water pos
        if st.water_pos is not None:
            x, y = st.water_pos
            base[8, y, x] = 1.0

        # exits
        if st.fire_exit is not None:
            x, y = st.fire_exit
            base[9, y, x] = 1.0
        if st.water_exit is not None:
            x, y = st.water_exit
            base[10, y, x] = 1.0

        obs = {}
        if self.has_fire:
            obs["fire"] = base.copy()
        if self.has_water:
            obs["water"] = base.copy()
        return obs


# --------------------------------------------------------------------------------
# ASCII renderer
# --------------------------------------------------------------------------------

def render_ascii(env: FireWaterEnv, file=sys.stdout):
    """
    Render the current state of the env as ASCII.

    Priority (topmost last):
      1) static base_grid
      2) open/closed doors (visualized differently if open)
      3) exits (f,g)
      4) switches (1-5) if not overwritten
      5) blocks (X)
      6) agents (F,G)
    """
    st = env.state
    if st is None:
        print("[Environment not initialized]", file=file)
        return

    H, W = st.base_grid.shape
    # start with static
    canvas = np.array(st.base_grid, copy=True)

    # mark doors as open/closed visually
    active_switches = env._active_switch_ids(st)
    for y in range(H):
        for x in range(W):
            ch = st.base_grid[y, x]
            if ch in DOOR_CHARS:
                if DOOR_TO_SWITCH[ch] in active_switches:
                    # open door: lowercase letter
                    canvas[y, x] = ch.lower()
                else:
                    canvas[y, x] = ch  # closed door: uppercase

    # exits (just for visualization, underlying base is technically floor)
    if st.fire_exit is not None:
        fx, fy = st.fire_exit
        if canvas[fy, fx] == FLOOR:
            canvas[fy, fx] = FIRE_EXIT
    if st.water_exit is not None:
        gx, gy = st.water_exit
        if canvas[gy, gx] == FLOOR:
            canvas[gy, gx] = WATER_EXIT

    # blocks
    for (x, y) in st.blocks:
        canvas[y, x] = BLOCK

    # agents override everything visually
    if st.fire_pos is not None:
        x, y = st.fire_pos
        canvas[y, x] = FIRE_SPAWN
    if st.water_pos is not None:
        x, y = st.water_pos
        canvas[y, x] = WATER_SPAWN

    for y in range(H):
        row = ''.join(canvas[y, :])
        print(row, file=file)
    print(file=file)


# --------------------------------------------------------------------------------
# Manual control + scripted + policy
# --------------------------------------------------------------------------------

def _key_to_actions(key: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Map keyboard input to (action_fire, action_water).

    Fire (F):
      W A S D : up, left, down, right

    Water (G):
      I J K L : up, left, down, right

    Returns None for agents that should not move from that key.
    """
    key = key.strip()

    # Fire controls (WASD)
    fire_action: Optional[int] = None
    water_action: Optional[int] = None

    if key.lower() == 'w':    # up
        fire_action = ACTION_UP
    elif key.lower() == 'a':  # left
        fire_action = ACTION_LEFT
    elif key.lower() == 's':  # down
        fire_action = ACTION_DOWN
    elif key.lower() == 'd':  # right
        fire_action = ACTION_RIGHT

    # Water controls (IJKL)
    if key.lower() == 'i':    # up
        water_action = ACTION_UP
    elif key.lower() == 'j':  # left
        water_action = ACTION_LEFT
    elif key.lower() == 'k':  # down
        water_action = ACTION_DOWN
    elif key.lower() == 'l':  # right
        water_action = ACTION_RIGHT

    return fire_action, water_action


def play_manual(env: FireWaterEnv):
    """
    Simple manual control loop using stdin.

    - Prints ASCII each step.
    - You type a key and press Enter.
    - Supports:
        Fire: WASD
        Water: IJKL
    """
    obs = env.reset()
    done = False

    print("Manual control. Controls:")
    print("  Fire  (F): W A S D  (up/left/down/right)")
    print("  Water (G): I J K L  (up/left/down/right)")
    print("  q: quit")
    print()

    step = 0
    while not done:
        print(f"Step {step}")
        render_ascii(env)
        key = input("Move (WASD/IJKL, q to quit): ")
        if key.lower().startswith('q'):
            print("Quitting.")
            break

        fire_action, water_action = _key_to_actions(key)

        # default to STAY if no action for that agent from this key
        if env.has_fire and fire_action is None:
            fire_action = ACTION_STAY
        if env.has_water and water_action is None:
            water_action = ACTION_STAY

        obs, reward, done, info = env.step(
            fire_action if fire_action is not None else ACTION_STAY,
            water_action if water_action is not None else ACTION_STAY,
        )
        print(f"Reward: {reward:.3f}, done={done}, info={info}")
        step += 1

    print("Final state:")
    render_ascii(env)


def run_scripted_episode(
    env: FireWaterEnv,
    fire_actions: List[int],
    water_actions: Optional[List[int]] = None,
    render: bool = True
):
    """
    Run an episode with scripted actions for testing.

    fire_actions: list of ints (0-4)
    water_actions: list of ints (0-4) or None (STAY)
    """
    obs = env.reset()
    done = False
    t = 0
    total_reward = 0.0

    if water_actions is None:
        water_actions = [ACTION_STAY] * len(fire_actions)

    assert len(fire_actions) == len(water_actions)

    while not done and t < len(fire_actions):
        if render:
            print(f"Step {t}")
            render_ascii(env)
        af = fire_actions[t]
        aw = water_actions[t]
        obs, r, done, info = env.step(af, aw)
        total_reward += r
        t += 1

    if render:
        print("Final state:")
        render_ascii(env)
        print(f"Total reward: {total_reward:.3f}, done={done}, steps={t}, info={info}")


def load_script_file(path: str) -> Tuple[List[int], List[int]]:
    """
    Load a scripted action sequence from a text file.

    Each non-empty, non-comment line should be:
      fire_action water_action

    where actions are ints in [0,4]:
      0=UP, 1=LEFT, 2=DOWN, 3=RIGHT, 4=STAY

    If only one int is provided, water defaults to STAY.
    """
    fire_actions: List[int] = []
    water_actions: List[int] = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                try:
                    af = int(parts[0])
                except ValueError:
                    raise ValueError(f"Invalid action on line {lineno} in {path}: {line!r}")
                aw = ACTION_STAY
            elif len(parts) == 2:
                try:
                    af = int(parts[0])
                    aw = int(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid actions on line {lineno} in {path}: {line!r}")
            else:
                raise ValueError(f"Expected 1 or 2 ints on line {lineno} in {path}, got: {line!r}")

            if not (0 <= af < NUM_ACTIONS and 0 <= aw < NUM_ACTIONS):
                raise ValueError(f"Action out of range on line {lineno} in {path}: {line!r}")

            fire_actions.append(af)
            water_actions.append(aw)

    if not fire_actions:
        raise ValueError(f"No actions found in script file: {path}")
    return fire_actions, water_actions


def run_policy_episode(
    env: FireWaterEnv,
    policy_fn,
    max_steps: int = 200,
    render: bool = True,
):
    """
    Run an episode where actions are chosen by a policy function.

    policy_fn: callable taking obs dict and returning (action_fire, action_water)
               each an int in [0,4].

    Example signature
        def policy_fn(obs: Dict[str, np.ndarray]) -> Tuple[int, int]:
            ...
    """
    obs = env.reset()
    done = False
    t = 0
    total_reward = 0.0

    while not done and t < max_steps:
        if render:
            print(f"Step {t}")
            render_ascii(env)

        a_fire, a_water = policy_fn(obs)
        obs, r, done, info = env.step(a_fire, a_water)
        total_reward += r
        t += 1

    if render:
        print("Final state:")
        render_ascii(env)
        print(f"Total reward: {total_reward:.3f}, steps={t}, info={info}")


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FireWaterEnv CLI: run manual, scripted, or policy episodes."
    )
    parser.add_argument(
        "level",
        help="Path to level .txt file (ASCII map).",
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "scripted", "policy"],
        default="manual",
        help="How to drive the environment: manual controls, scripted sequence, or policy.",
    )
    parser.add_argument(
        "--script",
        help="Path to script file (for --mode scripted).",
    )
    parser.add_argument(
        "--policy-module",
        help=(
            "Python module path providing a `policy_fn(obs) -> (a_fire, a_water)` function "
            "for --mode policy, e.g. 'myproject.my_policy'."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode.",
    )

    args = parser.parse_args()

    # load level
    lvl = parse_level_from_file(args.level)
    env = FireWaterEnv(lvl, max_steps=args.max_steps)

    if args.mode == "manual":
        # manual keyboard control
        play_manual(env)

    elif args.mode == "scripted":
        if not args.script:
            parser.error("--mode scripted requires --script PATH")
        fire_actions, water_actions = load_script_file(args.script)
        run_scripted_episode(env, fire_actions, water_actions, render=True)

    elif args.mode == "policy":
        if not args.policy_module:
            parser.error("--mode policy requires --policy-module MODULE_PATH")

        # dynamically import the module and get policy_fn
        mod = import_module(args.policy_module)
        if not hasattr(mod, "policy_fn"):
            parser.error(f"Module {args.policy_module!r} has no function 'policy_fn'")

        policy_fn = getattr(mod, "policy_fn")
        run_policy_episode(env, policy_fn, max_steps=args.max_steps, render=True)


if __name__ == "__main__":
    main()
