from typing import Dict
import numpy as np
from firewater_env import FireWaterEnv, parse_level_from_file, NUM_ACTIONS


class MultiAgentFireWaterEnv:
    def __init__(self, level_path: str, max_steps: int = 200):
        lvl = parse_level_from_file(level_path)
        self.env = FireWaterEnv(lvl, max_steps=max_steps)

        self.agent_ids = []
        if self.env.has_fire:
            self.agent_ids.append("fire")
        if self.env.has_water:
            self.agent_ids.append("water")

        self.num_actions = NUM_ACTIONS

    def reset(self) -> Dict[str, np.ndarray]:
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
