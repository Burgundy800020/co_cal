"""Online RL fine-tuning (TD3+BC -> TD3, BC weight annealed) from an offline checkpoint, unlimited budget.

Phase 1 of expert training; also the "standard offline-to-online TD3" baseline.
Artefacts -> runs/<name>/<timestamp>/ in the same layout as offline runs (so the result is loadable
by `checkpoint.load_agent` as the next phase's init).

Usage: python drivers/train_online.py configs/online_td3_ant_medium.yaml [online.total_steps=5000 seed=1 ...]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from safe_f2o.checkpoint import load_agent, load_run_config
from safe_f2o.config import load_config
from safe_f2o.data import load_offline, make_env
from safe_f2o.logging_utils import JsonlLogger, make_run_dir, save_config, write_json
from safe_f2o.online import OnlineConfig, run_online
from safe_f2o.replay_buffer import MixedBuffer, ReplayBuffer
from plot_online import main as plot_online


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--runs_root", default="runs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    init_cfg = load_run_config(cfg["init_from"])
    dataset = cfg.get("dataset") or init_cfg["dataset"]
    ocfg = OnlineConfig.from_dict(cfg["online"])

    # buffer: online only | offline + online in one buffer (uniform sampling, online_frac: null)
    #         | separate buffers with balanced sampling (online_frac in (0, 1])
    online_frac = cfg.get("online_frac")
    if cfg["offline_data"] and online_frac is None:
        buf, env, _ = load_offline(dataset, cfg["device"], extra_capacity=ocfg.total_steps)
        offline_buf = buf
    else:
        env = make_env(dataset)
        buf = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0],
                           max_size=ocfg.total_steps, device=cfg["device"])
        offline_buf = None
        if cfg["offline_data"]:
            offline_buf, _, _ = load_offline(dataset, cfg["device"])
            buf = MixedBuffer(offline_buf, buf, online_frac)
    eval_env = make_env(dataset)

    agent, norm, run_cfg = load_agent(cfg["init_from"], env, cfg["device"],
                                      algo_overrides=cfg.get("algo_overrides"), load_optim=cfg["load_optim"])
    if offline_buf is not None:
        norm.apply_to_buffer(offline_buf)  # offline data normalised with the checkpoint's stats

    # saved config: the merged algo block + normalisation, so this run dir is itself a checkpoint
    cfg.update(dataset=dataset, algo=run_cfg["algo"], normalize_state=run_cfg.get("normalize_state", False))
    run_dir = make_run_dir(args.runs_root, cfg.get("name") or "online_" + dataset.replace("/", "_"))
    save_config(run_dir, cfg)
    norm.save(run_dir)
    print("run dir:", run_dir, "| offline transitions:", 0 if offline_buf is None else offline_buf.size,
          "| online_frac:", online_frac)

    log = JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))
    out = run_online(agent, env, eval_env, buf, ocfg, norm=norm, logger=log, seed=cfg["seed"],
                     save_fn=lambda tag: agent.save(os.path.join(run_dir, "model.pt")))
    log.close()
    write_json(os.path.join(run_dir, "done.json"), {"steps": ocfg.total_steps, **out})
    plot_online(run_dir, os.path.join(run_dir, "figures"))


if __name__ == "__main__":
    main()
