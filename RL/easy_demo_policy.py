import os
import numpy as np
import torch

from .networks import ConvEncoder, Actor, CentralCritic
from .mappo_agent import MAPPOAgent
from env.firewater_env import NUM_ACTIONS

CKPT_PATH = "checkpoints/mappo_easy.pt"
HIDDEN_DIM = 128                               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_agent = None
_initialized = False


def _init_agent_from_obs(sample_obs_fire: np.ndarray):
    global _agent, _initialized

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    C, H, W = sample_obs_fire.shape

    encoder = ConvEncoder(
        in_channels=C,
        hidden_dim=HIDDEN_DIM,
        height=H,
        width=W,
    ).to(DEVICE)

    actor_fire = Actor(obs_dim=HIDDEN_DIM, num_actions=NUM_ACTIONS).to(DEVICE)
    actor_water = Actor(obs_dim=HIDDEN_DIM, num_actions=NUM_ACTIONS).to(DEVICE)
    critic = CentralCritic(joint_dim=2 * HIDDEN_DIM).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    actor_fire.load_state_dict(ckpt["actor_fire"])
    actor_water.load_state_dict(ckpt["actor_water"])
    critic.load_state_dict(ckpt["critic"])

    _agent = MAPPOAgent(
        encoder=encoder,
        actor_fire=actor_fire,
        actor_water=actor_water,
        critic=critic,
        optimizer=None,
    )

    _agent.encoder.eval()
    _agent.actor_fire.eval()
    _agent.actor_water.eval()
    _agent.critic.eval()

    _initialized = True
    print(f"[easy_demo_policy] Loaded checkpoint from {CKPT_PATH} on {DEVICE}", flush=True)


def _act_greedy(agent: MAPPOAgent,
                obs_fire_t: torch.Tensor,
                obs_water_t: torch.Tensor):
    with torch.no_grad():
        feat_fire = agent.encoder(obs_fire_t)
        feat_water = agent.encoder(obs_water_t)

        logits_f = agent.actor_fire(feat_fire)
        logits_w = agent.actor_water(feat_water)

        a_f = torch.argmax(logits_f, dim=-1)
        a_w = torch.argmax(logits_w, dim=-1)

    return int(a_f.item()), int(a_w.item())


def policy_fn(obs):
    global _initialized, _agent

    fire_np = obs["fire"]
    water_np = obs["water"]

    if not _initialized:
        _init_agent_from_obs(fire_np)

    fire_t = torch.from_numpy(fire_np).unsqueeze(0).to(DEVICE)
    water_t = torch.from_numpy(water_np).unsqueeze(0).to(DEVICE)

    a_f, a_w = _act_greedy(_agent, fire_t, water_t)
    return a_f, a_w
