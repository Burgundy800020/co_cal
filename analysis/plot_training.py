"""Plot eval-return curves. Reads golden artefacts from results/, writes figures to results/figures/.

Usage: python analysis/plot_training.py <experiment_dir> [out_dir]   (out_dir default: results/figures)
"""
import json
import os
import sys

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cocal.logging_utils import read_jsonl


def main(exp_dir, out=os.path.join("results", "figures")):
    log = list(read_jsonl(os.path.join(exp_dir, "train_log.jsonl")))
    ev = [r for r in log if r["kind"] == "eval"]
    tr = [r for r in log if r["kind"] == "train"]

    def series(rows, key):
        pts = [(r["step"], r[key]) for r in rows if key in r]
        return [p[0] for p in pts], [p[1] for p in pts]

    # One panel per quantity "space"; metrics sharing a space share an axis.
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 9), sharex=True)
    ax = axes[0]
    st = [r["step"] for r in ev]
    m = [r["mean_return"] for r in ev]
    lo = [min(r["returns"]) for r in ev]
    hi = [max(r["returns"]) for r in ev]
    ax.plot(st, m, marker="o", ms=3, label="mean")
    ax.fill_between(st, lo, hi, alpha=0.2, label="min–max")
    ax.legend(fontsize=8)
    ax.set_ylabel("eval return")

    for key, c in (("q1", "C2"), ("q2", "C5"), ("q", "C2")):  # "q" = legacy logs (Q1 only)
        if any(key in r for r in tr):
            axes[1].plot(*series(tr, key), color=c, label="q1" if key == "q" else key)
    axes[1].set_ylabel("mean Q(s, a_data)")
    axes[1].legend(fontsize=8)

    axes[2].plot(*series(tr, "critic_loss"), color="C3")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("critic loss\n(return$^2$, log)")

    # actor_loss = -lambda*Q + bc, with lambda*Q ~= alpha, so actor_loss + alpha ~= bc (action space)
    with open(os.path.join(exp_dir, "config.yaml")) as f:
        alpha = yaml.safe_load(f)["algo"]["alpha"]
    sx, sy = series(tr, "actor_loss")
    axes[3].plot(sx, [v + alpha for v in sy], color="C1", label=f"actor_loss + alpha ({alpha})")
    axes[3].plot(*series(tr, "bc_loss"), color="C4", ls="--", label="bc_loss")
    axes[3].set_ylabel("action-space MSE")
    axes[3].legend(fontsize=8)
    axes[3].set_xlabel("step")
    for a_ in axes:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, os.path.basename(os.path.normpath(exp_dir)) + "_training.png")
    plt.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:3])
