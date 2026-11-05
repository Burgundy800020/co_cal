"""TD3+BC (Fujimoto & Gu, 2021): TD3 with lambda-normalised Q term plus BC loss in the actor update.

Actor loss: -lmbda * Q(s, pi(s)) + MSE(pi(s), a),  lmbda = alpha / mean|Q(s, pi(s))|.
Optionally normalises states with dataset mean/std (as in the paper); `normalize_state` returns stats.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim])
        self.max_action = max_action

    def forward(self, obs):
        return self.max_action * torch.tanh(self.net(obs))


class Critic(nn.Module):
    """Twin Q networks."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs, act):
        return self.q1(torch.cat([obs, act], -1)).squeeze(-1)


def normalize_state(buffer, eps=1e-3):
    """In-place normalise buffer obs/next_obs; returns (mean, std)."""
    mean = buffer.obs[: buffer.size].mean(0, keepdims=True)
    std = buffer.obs[: buffer.size].std(0, keepdims=True) + eps
    buffer.obs[: buffer.size] = (buffer.obs[: buffer.size] - mean) / std
    buffer.next_obs[: buffer.size] = (buffer.next_obs[: buffer.size] - mean) / std
    return mean, std


class TD3BC:
    def __init__(self, obs_dim, act_dim, max_action, device="cpu", hidden=(256, 256),
                 lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, alpha=2.5):
        self.device = torch.device(device)
        self.max_action = max_action
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise * max_action, noise_clip * max_action
        self.policy_delay, self.alpha = policy_delay, alpha

        self.actor = Actor(obs_dim, act_dim, max_action, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.total_it = 0

    @torch.no_grad()
    def act(self, obs):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        return self.actor(o.unsqueeze(0)).squeeze(0).cpu().numpy()

    def update(self, buffer, batch_size=256):
        self.total_it += 1
        b = buffer.sample(batch_size)
        obs, act, rew, nobs, done = b["obs"], b["actions"], b["rewards"], b["next_obs"], b["dones"]

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            nact = (self.actor_target(nobs) + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_target(nobs, nact)
            target_q = rew + (1.0 - done) * self.gamma * torch.min(tq1, tq2)

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        stats = {"critic_loss": critic_loss.item(), "q": q1.mean().item()}
        if self.total_it % self.policy_delay == 0:
            pi = self.actor(obs)
            q = self.critic.q1_only(obs, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            bc = F.mse_loss(pi, act)
            actor_loss = -lmbda * q.mean() + bc
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            for net, tgt in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            stats.update(actor_loss=actor_loss.item(), bc_loss=bc.item())
        return stats

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
