"""Train TD3+BC on a Minari dataset. Artefacts (incl. figures/) -> runs/<name>/<timestamp>/ (git-ignored).

Usage: python drivers/train_offline.py configs/td3bc_hopper_medium.yaml [train.steps=1000 seed=1 ...]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from cocal.config import load_config
from cocal.data import load_offline
from cocal.evaluation import evaluate
from cocal.logging_utils import JsonlLogger, make_run_dir, save_config, write_json
from cocal.td3_bc import TD3BC, normalize_state
from plot_training import main as plot_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--runs_root", default="runs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    name = cfg["dataset"].replace("/", "_")
    run_dir = make_run_dir(args.runs_root, name)
    save_config(run_dir, cfg)
    log = JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))
    print("run dir:", run_dir)

    buf, env, _ = load_offline(cfg["dataset"], cfg["device"])
    norm = normalize_state(buf) if cfg["normalize_state"] else None
    if norm is not None:
        np.savez(os.path.join(run_dir, "state_norm.npz"), mean=norm[0], std=norm[1])

    obs_dim, act_dim = buf.obs.shape[1], buf.actions.shape[1]
    max_action = float(env.action_space.high[0])
    a = cfg["algo"]
    agent = TD3BC(obs_dim, act_dim, max_action, cfg["device"], hidden=tuple(a["hidden"]), lr=a["lr"],
                  gamma=a["gamma"], tau=a["tau"], policy_noise=a["policy_noise"],
                  noise_clip=a["noise_clip"], policy_delay=a["policy_delay"], alpha=a["alpha"])

    t = cfg["train"]
    for step in range(1, t["steps"] + 1):
        stats = agent.update(buf, t["batch_size"])
        if step % 1000 == 0:
            log.log(step=step, kind="train", **stats)
        if step % t["eval_every"] == 0:
            res, _ = evaluate(env, agent.act, t["eval_episodes"], seed=cfg["seed"] + 100, normalize=norm)
            log.log(step=step, kind="eval", **res)
            print(f"step {step} eval return {res['mean_return']:.1f} +- {res['std_return']:.1f}")
        if step % t["save_every"] == 0 or step == t["steps"]:
            agent.save(os.path.join(run_dir, "model.pt"))
    write_json(os.path.join(run_dir, "done.json"), {"steps": t["steps"]})
    plot_training(run_dir, os.path.join(run_dir, "figures"))  # untracked: figures live in the run dir


if __name__ == "__main__":
    main()
