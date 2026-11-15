"""Policy rollouts. Returns raw per-episode / per-step data so downstream code can calibrate & plot."""
import numpy as np


def rollout(env, policy_fn, seed=None, normalize=None):
    """One episode. `policy_fn(obs)->action`. Returns dict with per-step arrays and the return."""
    obs, _ = env.reset(seed=seed)
    traj = {"obs": [], "actions": [], "rewards": []}
    done = False
    while not done:
        o = obs if normalize is None else (obs - normalize[0][0]) / normalize[1][0]
        a = policy_fn(o)
        traj["obs"].append(obs)
        traj["actions"].append(a)
        obs, r, term, trunc, _ = env.step(a)
        traj["rewards"].append(r)
        done = term or trunc
    out = {k: np.asarray(v) for k, v in traj.items()}
    out["return"] = float(out["rewards"].sum())
    out["length"] = len(out["rewards"])
    return out


def evaluate(env, policy_fn, n_episodes=10, seed=0, normalize=None):
    eps = [rollout(env, policy_fn, seed=seed + i, normalize=normalize) for i in range(n_episodes)]
    rets = np.array([e["return"] for e in eps])
    return {"mean_return": float(rets.mean()), "std_return": float(rets.std()), "returns": rets.tolist()}, eps
