"""Aggregate an online experiment over seeds: mean ± 1 std across seeds (not across eval episodes).

Panels: (1) eval return vs env steps, (2) eval return vs online terminations,
(3) cumulative terminations vs env steps.
Return-vs-terminations: each seed reaches a given termination count at a different step, so each seed's
eval-return curve is linearly interpolated onto a common termination grid, up to the smallest final
count across seeds (so every grid point averages all seeds).

Usage: python analysis/plot_seeds.py <exp_dir with seed*/ run dirs> [out_dir]   (default: results/figures)
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from safe_f2o.logging_utils import read_jsonl

COLOR = "#2a6fdb"


def load_seed(run_dir):
    log = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    ev = [r for r in log if r["kind"] == "eval"]
    ep = [r for r in log if r["kind"] == "episode"]
    return {
        "eval_step": np.array([r["step"] for r in ev]),
        "eval_term": np.array([r["n_terminations"] for r in ev], dtype=float),
        "eval_ret": np.array([r["mean_return"] for r in ev]),
        # cumulative terminations as a step function of env steps (starts at 0)
        "ep_step": np.array([0] + [r["step"] for r in ep]),
        "ep_term": np.array([0] + [r["n_terminations"] for r in ep], dtype=float),
    }


def band(ax, x, ys, xlabel, ylabel, title):
    m, s = ys.mean(0), ys.std(0)
    ax.fill_between(x, m - s, m + s, color=COLOR, alpha=0.2, lw=0)
    ax.plot(x, m, color=COLOR, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    if xlabel == "env steps":
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main(exp_dir, out=os.path.join("results", "figures")):
    runs = sorted(d for d in glob.glob(os.path.join(exp_dir, "seed*")) if os.path.isdir(d))
    seeds = [load_seed(d) for d in runs]
    n = len(seeds)
    name = os.path.basename(os.path.normpath(exp_dir))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1) eval return vs env steps (all seeds share eval steps)
    steps = seeds[0]["eval_step"]
    assert all(np.array_equal(s["eval_step"], steps) for s in seeds), "seeds have different eval steps"
    band(axes[0], steps, np.stack([s["eval_ret"] for s in seeds]),
         "env steps", "eval return", f"Eval return vs env steps (mean ± 1 std, {n} seeds)")

    # 2) eval return vs terminations (interpolated onto a common grid)
    t_max = min(s["eval_term"][-1] for s in seeds)
    grid = np.linspace(0, t_max, 200)
    ys = []
    for s in seeds:
        t, idx = np.unique(s["eval_term"], return_index=True)  # np.interp needs increasing x
        ys.append(np.interp(grid, t, s["eval_ret"][idx]))
    band(axes[1], grid, np.stack(ys), "online terminations", "eval return",
         f"Eval return vs online terminations ({n} seeds)")

    # 3) cumulative terminations vs env steps
    total = max(s["ep_step"][-1] for s in seeds)
    sgrid = np.linspace(0, total, 400)
    ys = [s["ep_term"][np.searchsorted(s["ep_step"], sgrid, side="right") - 1] for s in seeds]
    band(axes[2], sgrid, np.stack(ys), "env steps", "cumulative terminations",
         f"Online terminations vs env steps ({n} seeds)")

    fig.suptitle(name, x=0.01, ha="left", fontsize=11)
    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}_seeds.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:3])
