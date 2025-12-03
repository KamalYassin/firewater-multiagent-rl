import torch
from torch.distributions import Categorical

from .buffer import RolloutBuffer


class MAPPOAgent:
    def __init__(self, encoder, actor_fire, actor_water, critic, optimizer,
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.encoder = encoder
        self.actor_fire = actor_fire
        self.actor_water = actor_water
        self.critic = critic
        self.optimizer = optimizer

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def act(self, obs_fire: torch.Tensor, obs_water: torch.Tensor):
        feat_fire = self.encoder(obs_fire)    
        feat_water = self.encoder(obs_water)  

        logits_f = self.actor_fire(feat_fire)
        logits_w = self.actor_water(feat_water)

        dist_f = Categorical(logits=logits_f)
        dist_w = Categorical(logits=logits_w)

        a_f = dist_f.sample()
        a_w = dist_w.sample()

        logp_f = dist_f.log_prob(a_f)
        logp_w = dist_w.log_prob(a_w)

        joint = torch.cat([feat_fire, feat_water], dim=-1)
        values = self.critic(joint)  # (B,)

        return (a_f, a_w), (logp_f, logp_w), values

    def _compute_gae(self, rewards, dones, values, last_value):
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=rewards.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self,
               buffer: RolloutBuffer,
               last_obs_fire: torch.Tensor,
               last_obs_water: torch.Tensor,
               num_epochs: int = 4):
    
        device = next(self.encoder.parameters()).device

        with torch.no_grad():
            last_fire = last_obs_fire.unsqueeze(0).to(device)
            last_water = last_obs_water.unsqueeze(0).to(device)

            feat_fire_last = self.encoder(last_fire)
            feat_water_last = self.encoder(last_water)
            joint_last = torch.cat([feat_fire_last, feat_water_last], dim=-1)
            last_value = self.critic(joint_last).squeeze(0)

        rewards = buffer.rewards.detach()
        dones = buffer.dones.detach()
        values = buffer.values.detach()

        advantages, returns = self._compute_gae(rewards, dones, values, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = advantages.detach()
        returns = returns.detach()

        old_logp_f = buffer.logp_fire.detach()
        old_logp_w = buffer.logp_water.detach()

        actions_f = buffer.actions_fire.detach()
        actions_w = buffer.actions_water.detach()

        obs_fire = buffer.obs_fire.detach()
        obs_water = buffer.obs_water.detach()

        for _ in range(num_epochs):
            feat_fire = self.encoder(obs_fire)
            feat_water = self.encoder(obs_water)

            logits_f = self.actor_fire(feat_fire)
            logits_w = self.actor_water(feat_water)

            dist_f = Categorical(logits=logits_f)
            dist_w = Categorical(logits=logits_w)

            logp_f = dist_f.log_prob(actions_f)
            logp_w = dist_w.log_prob(actions_w)

            entropy_f = dist_f.entropy().mean()
            entropy_w = dist_w.entropy().mean()

            joint = torch.cat([feat_fire, feat_water], dim=-1)
            values_pred = self.critic(joint)

            ratio_f = torch.exp(logp_f - old_logp_f)
            ratio_w = torch.exp(logp_w - old_logp_w)

            adv = advantages
            ret = returns

            def ppo_loss(ratio, adv_):
                unclipped = ratio * adv_
                clipped = torch.clamp(ratio,
                                      1.0 - self.clip_eps,
                                      1.0 + self.clip_eps) * adv_
                return -torch.min(unclipped, clipped).mean()

            policy_loss_f = ppo_loss(ratio_f, adv)
            policy_loss_w = ppo_loss(ratio_w, adv)

            value_loss = (values_pred - ret).pow(2).mean()
            entropy_bonus = (entropy_f + entropy_w) * 0.5

            loss = (policy_loss_f + policy_loss_w) \
                   + self.value_coef * value_loss \
                   - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
                + list(self.actor_fire.parameters())
                + list(self.actor_water.parameters())
                + list(self.critic.parameters()),
                max_norm=0.5,
            )
            self.optimizer.step()
