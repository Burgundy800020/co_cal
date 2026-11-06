"""Plot eval-return curves. Reads golden artefacts from results/, writes figures to results/figures/.

Usage: python analysis/plot_training.py results/<experiment_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cocal.logging_utils import read_jsonl


def main(exp_dir):
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
    sd = [r["std_return"] for r in ev]
    ax.plot(st, m, marker="o", ms=3)
    ax.fill_between(st, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)], alpha=0.2)
    ax.set_ylabel("eval return")

    axes[1].plot(*series(tr, "q"), color="C2")
    axes[1].set_ylabel("mean Q1")

    for key, c in (("critic_loss", "C3"), ("bc_loss", "C4")):  # both MSE-type, same space
        axes[2].plot(*series(tr, key), label=key, color=c)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("MSE losses (log)")
    axes[2].legend(fontsize=8)

    axes[3].plot(*series(tr, "actor_loss"), color="C1")
    axes[3].set_ylabel("actor loss")
    axes[3].set_xlabel("step")
    for a_ in axes:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join("results", "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, os.path.basename(os.path.normpath(exp_dir)) + "_training.png")
    plt.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
