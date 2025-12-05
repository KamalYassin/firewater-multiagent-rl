import torch
import numpy as np
import glob
import os
import random

from .env_wrapper import MultiAgentFireWaterEnv
from .networks import ConvEncoder, Actor, CentralCritic
from .mappo_agent import MAPPOAgent
from .buffer import RolloutBuffer

TRAIN_ROOT = "env/levels/dataset/train"


def collect_level_paths(train_root: str, difficulties=("easy", "medium", "hard")):
    paths = []
    for diff in difficulties:
        pattern = os.path.join(train_root, diff, "*.txt")
        found = glob.glob(pattern)
        paths.extend(found)
    if not paths:
        raise ValueError(f"No level files found under {train_root} for {difficulties}")
    print(f"Found {len(paths)} train levels:", flush=True)
    return paths


def make_env(difficulties=("easy", "medium", "hard")):
    # Collect all train levels across easy/medium/hard
    level_paths = collect_level_paths(TRAIN_ROOT, difficulties=difficulties)
    return MultiAgentFireWaterEnv(level_paths)


def difficulties_for_update(update: int, total_updates: int):
    # """Simple 3-stage curriculum based on update index."""
    # if update <= total_updates // 3:
    #     return ("easy",)
    # elif update <= (2 * total_updates) // 3:
    #     return ("easy", "medium")
    # else:
    #     return ("easy", "medium", "hard")
    return ("easy",)


def act_greedy(agent: MAPPOAgent,
               obs_fire: torch.Tensor,
               obs_water: torch.Tensor,
               device: torch.device):
    """
    Deterministic policy: pick argmax action for both agents.
    obs_fire/obs_water: (1, C, H, W) tensors on device.
    """
    with torch.no_grad():
        feat_fire = agent.encoder(obs_fire)
        feat_water = agent.encoder(obs_water)

        logits_f = agent.actor_fire(feat_fire)
        logits_w = agent.actor_water(feat_water)

        a_f = torch.argmax(logits_f, dim=-1)  # (1,)
        a_w = torch.argmax(logits_w, dim=-1)  # (1,)

    return int(a_f.item()), int(a_w.item())


def evaluate_policy(env: MultiAgentFireWaterEnv,
                    agent: MAPPOAgent,
                    device: torch.device,
                    num_episodes: int = 200,
                    max_steps: int = 50) -> float:
    """
    Run num_episodes with greedy (argmax) policy and compute success rate.
    """
    successes = 0

    for ep in range(num_episodes):
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
    """
    Compare against a pure random policy.
    """
    successes = 0

    for ep in range(num_episodes):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()

    tmp_env = make_env(difficulties=("easy", "medium", "hard"))
    tmp_obs = tmp_env.reset()
    sample_obs = tmp_obs["fire"]
    C, H, W = sample_obs.shape
    print(f"Obs shape (C,H,W) = ({C},{H},{W})", flush=True)

    hidden_dim = 128
    encoder = ConvEncoder(in_channels=C, hidden_dim=hidden_dim,
                          height=H, width=W).to(device)
    actor_fire = Actor(obs_dim=hidden_dim, num_actions=env.num_actions).to(device)
    actor_water = Actor(obs_dim=hidden_dim, num_actions=env.num_actions).to(device)
    critic = CentralCritic(joint_dim=2 * hidden_dim).to(device)

    params = (
        list(encoder.parameters())
        + list(actor_fire.parameters())
        + list(actor_water.parameters())
        + list(critic.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=3e-4)

    agent = MAPPOAgent(encoder, actor_fire, actor_water, critic, optimizer)

    rollout_steps = 256      
    num_updates = 500          
    max_episode_steps = 50   

    buffer = RolloutBuffer(rollout_steps, obs_shape=(C, H, W), device=device)

    episode_returns = []
    episode_lengths = []
    episode_successes = []

    current_diffs = None
    env = None
    obs = None
    done = {"__all__": True}
    ep_return = 0.0
    ep_length = 0
    ep_success = False

    for update in range(1, num_updates + 1):
        diffs = difficulties_for_update(update, num_updates)
        if diffs != current_diffs:
            current_diffs = diffs
            print(f"\n[Stage] Update {update}: training on difficulties={current_diffs}", flush=True)
            env = make_env(difficulties=current_diffs)
            obs = env.reset()
            done = {"__all__": False}
            ep_return = 0.0
            ep_length = 0
            ep_success = False

        buffer.reset()

        for t in range(rollout_steps):
            if done["__all__"] or ep_length >= max_episode_steps:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_length)
                episode_successes.append(1.0 if ep_success else 0.0)

                obs = env.reset()
                ep_return = 0.0
                ep_length = 0
                ep_success = False
                done = {"__all__": False}

            fire_np = obs["fire"]   
            water_np = obs["water"]

            fire_t = torch.from_numpy(fire_np).unsqueeze(0).to(device)
            water_t = torch.from_numpy(water_np).unsqueeze(0).to(device)

            (a_f, a_w), (logp_f, logp_w), values = agent.act(fire_t, water_t)

            actions = {
                "fire": int(a_f.item()),
                "water": int(a_w.item()),
            }

            next_obs, rewards, dones, info = env.step(actions)

            reward = float(rewards["fire"])          
            done_all = float(dones["__all__"])      

            buffer.add(
                obs_fire=fire_t.squeeze(0),
                obs_water=water_t.squeeze(0),
                action_fire=a_f.squeeze(0),
                action_water=a_w.squeeze(0),
                reward=torch.tensor(reward, device=device),
                done=torch.tensor(done_all, device=device),
                value=values.squeeze(0),
                logp_fire=logp_f.squeeze(0),
                logp_water=logp_w.squeeze(0),
            )

            ep_return += reward
            ep_length += 1
            if info.get("success", False):
                ep_success = True

            obs = next_obs
            done = dones

        fire_np = obs["fire"]
        water_np = obs["water"]
        last_fire_t = torch.from_numpy(fire_np).to(device)
        last_water_t = torch.from_numpy(water_np).to(device)

        agent.update(buffer, last_fire_t, last_water_t, num_epochs=4)

        if len(episode_returns) > 0:
            mean_return = np.mean(episode_returns[-10:])
            mean_length = np.mean(episode_lengths[-10:])
            mean_success = np.mean(episode_successes[-10:])
        else:
            mean_return = 0.0
            mean_length = 0.0
            mean_success = 0.0

        print(
            f"Update {update:04d} | "
            f"MeanReturn(last10)={mean_return:.3f} | "
            f"MeanLen(last10)={mean_length:.1f}",
            f"SuccessRate(last10)={mean_success:.2f}",
            flush=True,
        )

    print("Training complete.")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/mappo_easy_curriculum.pt"

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "actor_fire": actor_fire.state_dict(),
            "actor_water": actor_water.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "hidden_dim": hidden_dim,
                "obs_shape": (C, H, W),
                "num_actions": env.num_actions,
            },
        },
        ckpt_path,
    )
    print(f"Saved checkpoint → {ckpt_path}")

    # Post-training evaluation on EASY levels only
    eval_env = make_env(difficulties=("easy",))
    easy_greedy_sr = evaluate_policy(eval_env, agent, device,
                                     num_episodes=200,
                                     max_steps=50)
    easy_random_sr = random_policy_baseline(eval_env,
                                            num_episodes=200,
                                            max_steps=50)

    print(
        f"[EVAL EASY] Greedy success rate = {easy_greedy_sr:.3f}, "
        f"Random success rate = {easy_random_sr:.3f}"
    )


if __name__ == "__main__":
    main()
