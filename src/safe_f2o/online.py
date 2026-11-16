"""Env-agnostic online RL loop for fine-tuning an (offline-pretrained) off-policy agent.

Tracks online terminations (true `terminated`, not time-limit truncations), our cost axis.
Hooks for later phases:
  - `reset_fn(env, rng) -> obs`: custom start-state distribution (e.g. expert handover states).
  - `on_episode_end(info_dict)`: per-episode callback.
The agent only needs `act(obs)`, `update(buffer, batch_size, update_actor)` and a `bc_weight` attr.
"""
import time
from dataclasses import dataclass

import numpy as np

from .checkpoint import StateNorm
from .evaluation import evaluate


def linear_schedule(start, end, duration, offset=0):
    """t -> value, linear from `start` (t<=offset) to `end` (t>=offset+duration)."""
    def f(t):
        if duration <= 0:
            return end
        frac = min(max((t - offset) / duration, 0.0), 1.0)
        return start + frac * (end - start)
    return f


@dataclass
class OnlineConfig:
    total_steps: int = 1_000_000
    batch_size: int = 256
    utd: int = 1                      # gradient updates per env step
    learning_starts: int = 0          # env steps before updates (0 when buffer already has offline data)
    random_steps: int = 0             # uniform-random actions at the start (0 when fine-tuning)
    critic_warmup: int = 0            # env steps with critic-only updates (offline critic adapts first)
    expl_noise: float = 0.1           # Gaussian action noise std, as a fraction of max_action
    bc_weight_start: float = 1.0      # BC weight annealed start -> end over bc_anneal_steps
    bc_weight_end: float = 0.0
    bc_anneal_steps: int = 100_000
    eval_every: int = 10_000
    eval_episodes: int = 10
    save_every: int = 50_000
    log_every: int = 1000

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def default_reset(env, rng):
    return env.reset(seed=int(rng.integers(2**31)))[0]


def run_online(agent, env, eval_env, buffer, cfg: OnlineConfig, norm: StateNorm = None, logger=None,
               save_fn=None, seed=0, reset_fn=default_reset, on_episode_end=None):
    """Online interaction + training. Buffer stores *normalised* observations.

    Logs (via `logger.log(**kw)`) rows with kind in {"episode", "train", "eval"}, each carrying
    `step` (env steps) and `n_terminations` (cumulative online terminations).
    `save_fn(tag)` is called periodically ("latest") and at the end ("final").
    """
    norm = norm or StateNorm()
    rng = np.random.default_rng(seed)
    max_action = float(env.action_space.high[0])
    act_dim = env.action_space.shape[0]
    bc_sched = linear_schedule(cfg.bc_weight_start, cfg.bc_weight_end, cfg.bc_anneal_steps,
                               offset=cfg.critic_warmup)
    log = logger.log if logger is not None else (lambda **kw: None)
    eval_seed = seed + 10_000

    def do_eval(step, n_term):
        res, _ = evaluate(eval_env, agent.act, cfg.eval_episodes, seed=eval_seed, normalize=norm.as_tuple())
        log(kind="eval", step=step, n_terminations=n_term, **res)
        print(f"[{step:>8}] eval {res['mean_return']:8.1f} +- {res['std_return']:7.1f} | "
              f"terminations {n_term} | bc_w {agent.bc_weight:.2f}", flush=True)
        return res

    agent.bc_weight = bc_sched(0)  # so the step-0 eval reports the scheduled weight, not the ckpt default
    do_eval(0, 0)
    obs = reset_fn(env, rng)
    ep_ret, ep_len, n_episodes, n_term = 0.0, 0, 0, 0
    window, t0 = {}, time.time()  # update stats accumulated over the current log window
    for step in range(1, cfg.total_steps + 1):
        o = norm(obs)
        if step <= cfg.random_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o) + rng.normal(0.0, cfg.expl_noise * max_action, size=act_dim)
            a = np.clip(a, -max_action, max_action).astype(np.float32)
        next_obs, r, term, trunc, _ = env.step(a)
        buffer.add(o, a, r, norm(next_obs), term)  # truncation keeps bootstrapping
        ep_ret += r
        ep_len += 1
        obs = next_obs

        if term or trunc:
            n_episodes += 1
            n_term += int(term)
            ep = dict(kind="episode", step=step, n_terminations=n_term, episode=n_episodes,
                      ret=float(ep_ret), length=ep_len, terminated=bool(term))
            log(**ep)
            if on_episode_end is not None:
                on_episode_end(ep)
            obs = reset_fn(env, rng)
            ep_ret, ep_len = 0.0, 0

        if step > cfg.learning_starts:
            agent.bc_weight = bc_sched(step)
            for _ in range(cfg.utd):
                stats = agent.update(buffer, cfg.batch_size, update_actor=step > cfg.critic_warmup)
                for k, v in stats.items():
                    window.setdefault(k, []).append(v)
        if step % cfg.log_every == 0 and window:
            # window means; actor stats average over the (delayed) actor updates only
            log(kind="train", step=step, n_terminations=n_term, bc_weight=agent.bc_weight,
                sps=cfg.log_every / (time.time() - t0), **{k: float(np.mean(v)) for k, v in window.items()})
            window, t0 = {}, time.time()
        if step % cfg.eval_every == 0:
            do_eval(step, n_term)
        if save_fn is not None and step % cfg.save_every == 0:
            save_fn("latest")
    if save_fn is not None:
        save_fn("final")
    return {"n_terminations": n_term, "n_episodes": n_episodes}
