"""Plot an online run: eval & episode return vs env steps and vs cumulative online terminations.

Usage: python analysis/plot_online.py <run_dir> [out_dir] [ema_alpha]   (defaults: results/figures, 0.05)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from safe_f2o.logging_utils import read_jsonl


def ema(xs, alpha):
    """Exponential moving average over episodes, bias-corrected so early values aren't pulled to 0."""
    out, m, w = [], 0.0, 0.0
    for x in xs:
        m = (1 - alpha) * m + alpha * x
        w = (1 - alpha) * w + alpha
        out.append(m / w)
    return out


def main(run_dir, out=os.path.join("results", "figures"), ema_alpha=0.05):
    ema_alpha = float(ema_alpha)
    log = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    ev = [r for r in log if r["kind"] == "eval"]
    ep = [r for r in log if r["kind"] == "episode"]
    tr = [r for r in log if r["kind"] == "train"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, x in ((axes[0, 0], "step"), (axes[0, 1], "n_terminations")):
        m = np.array([r["mean_return"] for r in ev])
        s = np.array([r["std_return"] for r in ev])
        xs = [r[x] for r in ev]
        ax.plot(xs, m, marker="o", ms=3, label="eval mean")
        ax.fill_between(xs, m - s, m + s, alpha=0.2, label="eval ±1 std")
        if ep:
            ax.scatter([r[x] for r in ep], [r["ret"] for r in ep], s=2, c="0.6", label="train episode")
            ax.plot([r[x] for r in ep], ema([r["ret"] for r in ep], ema_alpha), color="C3",
                    label=f"train EMA (α={ema_alpha})")
        ax.set_xlabel("env steps" if x == "step" else "online terminations")
        ax.set_ylabel("return")
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot([r["step"] for r in ep], [r["n_terminations"] for r in ep])
    ax.set_xlabel("env steps"); ax.set_ylabel("cumulative terminations")

    ax = axes[1, 1]
    st = [r["step"] for r in tr]
    if tr:
        ax.plot(st, [r["q1"] for r in tr], color="C2", label="mean Q1(s,a)")
        ax.set_ylabel("Q"); ax.set_xlabel("env steps")
        ax2 = ax.twinx()
        ax2.plot(st, [r["bc_weight"] for r in tr], color="C1", ls="--", label="bc_weight")
        ax2.set_ylabel("bc_weight")
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

    def series(key):
        pts = [(r["step"], r[key]) for r in tr if key in r]
        return [p[0] for p in pts], [p[1] for p in pts]

    ax = axes[2, 0]
    ax.plot(*series("critic_loss"), color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("env steps"); ax.set_ylabel("critic loss (log)")

    # actor_loss = -q_term + bc_weight * bc_loss  (q_term = lmbda * Q ~= alpha)
    ax = axes[2, 1]
    ax.plot(*series("actor_loss"), color="C1", label="actor_loss")
    ax.plot(*series("q_term"), color="C2", label="q_term = λ·Q (enters as −)")
    ax.set_xlabel("env steps"); ax.set_ylabel("actor loss terms")
    ax2 = ax.twinx()
    ax2.plot(*series("bc_loss"), color="C4", ls="--", label="bc_loss (MSE to batch actions)")
    ax2.set_ylabel("bc_loss")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, os.path.basename(os.path.normpath(run_dir)) + "_online.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
