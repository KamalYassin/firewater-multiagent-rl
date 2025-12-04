from typing import Dict, List, Union
import numpy as np
import random
from env.firewater_env import FireWaterEnv, parse_level_from_file, NUM_ACTIONS

LevelPaths = Union[str, List[str]]


class MultiAgentFireWaterEnv:
    def __init__(self, level_paths: LevelPaths, max_steps: int = 200):
        """
        level_paths can be:
          - a single string (one level)
          - a list of strings (multiple levels). A random one will be chosen each reset().
        """
        if isinstance(level_paths, str):
            self.level_paths = [level_paths]
        else:
            self.level_paths = list(level_paths)

        if not self.level_paths:
            raise ValueError("MultiAgentFireWaterEnv needs at least one level path.")

        self.max_steps = max_steps

        self._build_env(random.choice(self.level_paths))

        self.agent_ids = []
        if self.env.has_fire:
            self.agent_ids.append("fire")
        if self.env.has_water:
            self.agent_ids.append("water")

        self.num_actions = NUM_ACTIONS

    def _build_env(self, level_path: str):
        lvl = parse_level_from_file(level_path)
        self.env = FireWaterEnv(lvl, max_steps=self.max_steps)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        For training on a dataset, each episode can start on a random level.
        """
        level_path = random.choice(self.level_paths)
        self._build_env(level_path)
        obs = self.env.reset()
        return obs

    def step(self, actions: Dict[str, int]):
        a_fire = actions.get("fire", 4)
        a_water = actions.get("water", 4)

        obs, reward, done, info = self.env.step(a_fire, a_water)

        rewards = {agent: reward for agent in obs.keys()}
        dones = {agent: done for agent in obs.keys()}
        dones["__all__"] = done

        return obs, rewards, dones, info
