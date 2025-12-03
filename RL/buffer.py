import torch


class RolloutBuffer:
    def __init__(self, num_steps: int, obs_shape, device: torch.device):
        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.device = device

        self.reset()

    def reset(self):
        T = self.num_steps
        C, H, W = self.obs_shape

        self.step = 0

        self.obs_fire = torch.zeros((T, C, H, W), device=self.device)
        self.obs_water = torch.zeros((T, C, H, W), device=self.device)

        self.actions_fire = torch.zeros(T, dtype=torch.long, device=self.device)
        self.actions_water = torch.zeros(T, dtype=torch.long, device=self.device)

        self.rewards = torch.zeros(T, device=self.device)
        self.dones = torch.zeros(T, device=self.device)  

        self.values = torch.zeros(T, device=self.device)

        self.logp_fire = torch.zeros(T, device=self.device)
        self.logp_water = torch.zeros(T, device=self.device)

    def add(self,
            obs_fire, obs_water,
            action_fire, action_water,
            reward, done,
            value,
            logp_fire, logp_water):

        t = self.step
        self.obs_fire[t].copy_(obs_fire)
        self.obs_water[t].copy_(obs_water)

        self.actions_fire[t] = action_fire
        self.actions_water[t] = action_water

        self.rewards[t] = reward
        self.dones[t] = done

        self.values[t] = value

        self.logp_fire[t] = logp_fire
        self.logp_water[t] = logp_water

        self.step += 1

    @property
    def full(self) -> bool:
        return self.step >= self.num_steps
