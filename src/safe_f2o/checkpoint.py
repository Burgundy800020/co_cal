"""Agent construction and run-dir checkpoints (shared by offline, online and expert drivers).

A run dir is loadable if it holds `config.yaml` (with `algo` + `normalize_state`), `model.pt`
and, when normalising, `state_norm.npz`. Offline and online drivers both write this layout.
"""
import os

import numpy as np
import yaml

from .td3_bc import TD3BC


class StateNorm:
    """Fixed observation normaliser (dataset mean/std). Identity when mean/std are None."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = None if std is None else np.asarray(std, dtype=np.float32).reshape(1, -1)

    @property
    def active(self):
        return self.mean is not None

    def __call__(self, obs):
        if not self.active:
            return np.asarray(obs, dtype=np.float32)
        out = (np.asarray(obs, dtype=np.float32) - self.mean) / self.std
        return out[0] if np.ndim(obs) == 1 else out

    def as_tuple(self):
        """`(mean, std)` in the form `evaluation.evaluate(normalize=...)` expects, or None."""
        return (self.mean, self.std) if self.active else None

    def apply_to_buffer(self, buffer):
        """In-place normalise the filled part of a buffer (e.g. offline data added before online)."""
        if self.active:
            n = buffer.size
            buffer.obs[:n] = (buffer.obs[:n] - self.mean) / self.std
            buffer.next_obs[:n] = (buffer.next_obs[:n] - self.mean) / self.std

    def save(self, run_dir):
        if self.active:
            np.savez(os.path.join(run_dir, "state_norm.npz"), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, run_dir):
        path = os.path.join(run_dir, "state_norm.npz")
        if not os.path.exists(path):
            return cls()
        d = np.load(path)
        return cls(d["mean"], d["std"])


def build_agent(obs_dim, act_dim, max_action, algo, device="cpu"):
    """TD3BC from an `algo` config block."""
    return TD3BC(obs_dim, act_dim, max_action, device, hidden=tuple(algo["hidden"]), lr=algo["lr"],
                 gamma=algo["gamma"], tau=algo["tau"], policy_noise=algo["policy_noise"],
                 noise_clip=algo["noise_clip"], policy_delay=algo["policy_delay"], alpha=algo["alpha"],
                 bc_weight=algo.get("bc_weight", 1.0))


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def load_agent(run_dir, env, device="cpu", algo_overrides=None, load_optim=True):
    """Rebuild the agent saved in `run_dir` for `env`. Returns (agent, StateNorm, run_cfg).

    `algo_overrides` (e.g. new lr / bc_weight) are applied on top of the run's algo config;
    network sizes must match the checkpoint.
    """
    cfg = load_run_config(run_dir)
    algo = {**cfg["algo"], **(algo_overrides or {})}
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = build_agent(obs_dim, act_dim, float(env.action_space.high[0]), algo, device)
    agent.load(os.path.join(run_dir, "model.pt"), load_optim=load_optim)
    norm = StateNorm.load(run_dir) if cfg.get("normalize_state", False) else StateNorm()
    return agent, norm, {**cfg, "algo": algo}
