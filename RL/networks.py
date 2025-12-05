import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_dim: int = 128,
                 height: int = 12, width: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # -> (B, 64 * H * W)
        )
        self.fc = nn.Linear(64 * height * width, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.fc(x)
        return torch.relu(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.policy(x)
        return logits


class CentralCritic(nn.Module):
    def __init__(self, joint_dim: int):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        # returns shape (B,)
        return self.value(joint_obs).squeeze(-1)
