"""Configuration, paths, seeds and small I/O helpers shared by all experiments."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("ZSH_CONFIG", REPO / "configs" / "zsh_v2.json"))


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key, env in (("d1_parquet", "ZSH_D1"), ("elliptic_dir", "ZSH_ELLIPTIC"),
                     ("tagmap_json", "ZSH_TAGMAP"), ("out_root", "ZSH_OUT")):
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]
    return cfg


CFG = load_config()
OUT = Path(CFG["paths"]["out_root"])
RESULTS = REPO / "results"


def out_dir(*parts):
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(*parts):
    p = RESULTS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_for(*names):
    """Deterministic child seed for a named purpose, derived from the master seed."""
    h = hashlib.sha256("/".join(map(str, names)).encode()).digest()
    entropy = int.from_bytes(h[:8], "little")
    ss = np.random.SeedSequence([CFG["master_seed"], entropy])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def rng_for(*names):
    return np.random.default_rng(seed_for(*names))


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class Log:
    """Timestamped logger that prints and appends to a file."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
