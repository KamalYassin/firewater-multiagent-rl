import argparse
import glob
import os
import random

import numpy as np
import torch

from .env_wrapper import MultiAgentFireWaterEnv
from .networks import ConvEncoder, Actor, CentralCritic
from .mappo_agent import MAPPOAgent
from env.firewater_env import render_ascii


TRAIN_ROOT = "env/levels/dataset/train"


def collect_level_paths(train_root: str, difficulties=("easy",)):
    paths = []
    for diff in difficulties:
        pattern = os.path.join(train_root, diff, "*.txt")
        found = glob.glob(pattern)
        paths.extend(found)
    if not paths:
        raise ValueError(f"No level files found under {train_root} for {difficulties}")
    print(f"Found {len(paths)} eval levels for {difficulties}", flush=True)
    return paths


def make_env(difficulties=("easy",)):
    level_paths = collect_level_paths(TRAIN_ROOT, difficulties=difficulties)
    return MultiAgentFireWaterEnv(level_paths)


def act_greedy(agent: MAPPOAgent,
               obs_fire: torch.Tensor,
               obs_water: torch.Tensor,
               device: torch.device):
    with torch.no_grad():
        feat_fire = agent.encoder(obs_fire)
        feat_water = agent.encoder(obs_water)

        logits_f = agent.actor_fire(feat_fire)
        logits_w = agent.actor_water(feat_water)

        a_f = torch.argmax(logits_f, dim=-1)
        a_w = torch.argmax(logits_w, dim=-1)

    return int(a_f.item()), int(a_w.item())


def evaluate_policy(env: MultiAgentFireWaterEnv,
                    agent: MAPPOAgent,
                    device: torch.device,
                    num_episodes: int = 200,
                    max_steps: int = 50) -> float:
    successes = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = {"__all__": False}
        steps = 0
        info = {}

        while not done["__all__"] and steps < max_steps:
            fire_np = obs["fire"]
            water_np = obs["water"]

            fire_t = torch.from_numpy(fire_np).unsqueeze(0).to(device)
            water_t = torch.from_numpy(water_np).unsqueeze(0).to(device)

            a_f, a_w = act_greedy(agent, fire_t, water_t, device)

            obs, rewards, dones, info = env.step({"fire": a_f, "water": a_w})
            done = dones
            steps += 1

        if info.get("success", False):
            successes += 1

    return successes / float(num_episodes)


def random_policy_baseline(env: MultiAgentFireWaterEnv,
                           num_episodes: int = 200,
                           max_steps: int = 50) -> float:
    successes = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = {"__all__": False}
        steps = 0
        info = {}

        while not done["__all__"] and steps < max_steps:
            a_f = random.randint(0, env.num_actions - 1)
            a_w = random.randint(0, env.num_actions - 1)

            obs, rewards, dones, info = env.step({"fire": a_f, "water": a_w})
            done = dones
            steps += 1

        if info.get("success", False):
            successes += 1

    return successes / float(num_episodes)


def visualize_policy(env: MultiAgentFireWaterEnv,
                     agent: MAPPOAgent,
                     device: torch.device,
                     num_episodes: int = 5,
                     max_steps: int = 50):
    """
    Run a few greedy episodes and print ASCII frames.
    """
    success_count = 0

    for ep in range(1, num_episodes + 1):
        print("\n" + "=" * 60)
        print(f"VISUAL EPISODE {ep}/{num_episodes}")
        print("=" * 60)

        obs = env.reset()
        done = {"__all__": False}
        steps = 0
        ep_return = 0.0
        ep_success = False
        info = {}

        while not done["__all__"] and steps < max_steps:
            print(f"\nStep {steps}")
            render_ascii(env.env)

            fire_np = obs["fire"]
            water_np = obs["water"]

            fire_t = torch.from_numpy(fire_np).unsqueeze(0).to(device)
            water_t = torch.from_numpy(water_np).unsqueeze(0).to(device)

            a_f, a_w = act_greedy(agent, fire_t, water_t, device)

            obs, rewards, dones, info = env.step({"fire": a_f, "water": a_w})
            done = dones
            r = float(rewards["fire"])
            ep_return += r
            steps += 1

            print(f"Actions: fire={a_f}, water={a_w}")
            print(f"Reward: {r:.3f}, done={done['__all__']}, info={info}")

            if info.get("success", False):
                ep_success = True

        print("\nFINAL STATE:")
        render_ascii(env.env)
        print(f"Episode return: {ep_return:.3f}, steps: {steps}, success: {ep_success}")

        if ep_success:
            success_count += 1

    sr = success_count / float(num_episodes) if num_episodes > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"[VISUAL] Greedy success rate over {num_episodes} episodes = {sr:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MAPPO policy on FireWater levels."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint .pt file saved from train_mappo.py",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy"],
        help="Which difficulties to eval on (subset of easy medium hard)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of eval episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per eval episode",
    )
    parser.add_argument(
        "--visualize-episodes",
        type=int,
        default=0,
        help="If >0, also run this many greedy episodes with ASCII rendering.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(difficulties=tuple(args.difficulties))
    obs = env.reset()
    sample_obs = obs["fire"]
    C, H, W = sample_obs.shape
    print(f"Eval obs shape: (C,H,W)=({C},{H},{W})")

    hidden_dim = 128  # MUST match training

    encoder = ConvEncoder(in_channels=C, hidden_dim=hidden_dim,
                          height=H, width=W).to(device)
    actor_fire = Actor(obs_dim=hidden_dim, num_actions=env.num_actions).to(device)
    actor_water = Actor(obs_dim=hidden_dim, num_actions=env.num_actions).to(device)
    critic = CentralCritic(joint_dim=2 * hidden_dim).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    actor_fire.load_state_dict(ckpt["actor_fire"])
    actor_water.load_state_dict(ckpt["actor_water"])
    critic.load_state_dict(ckpt["critic"])

    agent = MAPPOAgent(
        encoder=encoder,
        actor_fire=actor_fire,
        actor_water=actor_water,
        critic=critic,
        optimizer=None,
    )

    greedy_sr = evaluate_policy(env, agent, device,
                                num_episodes=args.episodes,
                                max_steps=args.max_steps)
    random_sr = random_policy_baseline(env,
                                       num_episodes=args.episodes,
                                       max_steps=args.max_steps)

    print(
        f"[EVAL] difficulties={args.difficulties} "
        f"| Greedy SR={greedy_sr:.3f} | Random SR={random_sr:.3f}"
    )

    if args.visualize_episodes > 0:
        visualize_policy(env, agent, device,
                            num_episodes=args.visualize_episodes,
                            max_steps=args.max_steps)


if __name__ == "__main__":
    main()
