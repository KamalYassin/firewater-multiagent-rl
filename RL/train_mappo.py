import random
import torch
import numpy as np

from .env_wrapper import MultiAgentFireWaterEnv
from .networks import ConvEncoder, Actor, CentralCritic
from .mappo_agent import MAPPOAgent
from .buffer import RolloutBuffer


TRAIN_LEVEL = "levels/testing/switch_door_block.txt"


def make_env():
    return MultiAgentFireWaterEnv(TRAIN_LEVEL)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()

    obs = env.reset()
    sample_obs = obs["fire"]
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

    rollout_steps = 128      
    num_updates = 200          
    max_episode_steps = 200   

    buffer = RolloutBuffer(rollout_steps, obs_shape=(C, H, W), device=device)

    episode_returns = []
    episode_lengths = []

    obs = env.reset()
    ep_return = 0.0
    ep_length = 0
    done = {"__all__": False}

    for update in range(1, num_updates + 1):
        buffer.reset()

        for t in range(rollout_steps):
            if done["__all__"] or ep_length >= max_episode_steps:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_length)
                obs = env.reset()
                ep_return = 0.0
                ep_length = 0
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
        else:
            mean_return = 0.0
            mean_length = 0.0

        print(
            f"Update {update:04d} | "
            f"MeanReturn(last10)={mean_return:.3f} | "
            f"MeanLen(last10)={mean_length:.1f}",
            flush=True,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
