"""YAML config loading with CLI overrides (`key=value`, dotted keys allowed)."""
import copy
import yaml


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
