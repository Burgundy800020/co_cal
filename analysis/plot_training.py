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
    rows = [r for r in read_jsonl(os.path.join(exp_dir, "train_log.jsonl")) if r["kind"] == "eval"]
    steps = [r["step"] for r in rows]
    m = [r["mean_return"] for r in rows]
    s = [r["std_return"] for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.plot(steps, m)
    plt.fill_between(steps, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], alpha=0.2)
    plt.xlabel("step"); plt.ylabel("eval return"); plt.tight_layout()
    out = os.path.join("results", "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, os.path.basename(os.path.normpath(exp_dir)) + "_training.png")
    plt.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1])
