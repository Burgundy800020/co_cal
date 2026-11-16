"""Minari MuJoCo dataset loading -> ReplayBuffer."""
import minari
import numpy as np

from .replay_buffer import ReplayBuffer


def make_env(dataset_name):
    """The gymnasium env a Minari dataset was collected in (same version/kwargs)."""
    return minari.load_dataset(dataset_name, download=True).recover_environment()


def load_offline(dataset_name, device="cpu", extra_capacity=0):
    """Returns (buffer, eval_env, dataset). `dataset_name` e.g. 'mujoco/hopper/medium-v0'.
    `extra_capacity` leaves room in the buffer for online transitions appended later."""
    dataset = minari.load_dataset(dataset_name, download=True)
    eval_env = dataset.recover_environment()
    obs_dim = int(np.prod(dataset.observation_space.shape))
    act_dim = int(np.prod(dataset.action_space.shape))
    buf = ReplayBuffer(obs_dim, act_dim, max_size=dataset.total_steps + extra_capacity, device=device)
    for ep in dataset.iterate_episodes():
        buf.add_batch(
            obs=ep.observations[:-1],
            actions=ep.actions,
            rewards=ep.rewards,
            next_obs=ep.observations[1:],
            dones=ep.terminations,  # truncations do not cut off bootstrapping
        )
    return buf, eval_env, dataset


def episode_boundaries(dataset):
    """Per-episode (start, end) indices into the buffer, in iteration order (for episode-level splits)."""
    bounds, start = [], 0
    for ep in dataset.iterate_episodes():
        n = len(ep.actions)
        bounds.append((start, start + n))
        start += n
    return bounds
