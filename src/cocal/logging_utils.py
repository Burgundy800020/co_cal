"""Run directories and JSON-lines logging. Core + driver code write here; plotting reads it."""
import json
import os
import time

import yaml


def make_run_dir(root, name):
    run_dir = os.path.join(root, name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(run_dir, cfg):
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)


class JsonlLogger:
    """Append-only metrics log: one JSON dict per line."""

    def __init__(self, path):
        self.f = open(path, "a")

    def log(self, **kw):
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
