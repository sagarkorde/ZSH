"""Data loading for experiments, with a SMOKE mode that never touches evaluation data.

ZSH_SMOKE=1: DEV rows only. Rows from 2023-07-01 onward play the role of TEST,
the rest play DEV, both subsampled; Elliptic uses timesteps 1-34 only, with
25-34 standing in for the test period. Used to test code before `v2-frozen`.
"""
import json
import os

import numpy as np
import pandas as pd

from .config import CFG, OUT, RESULTS

SMOKE = os.environ.get("ZSH_SMOKE") == "1"
SMOKE_ROWS = int(os.environ.get("ZSH_SMOKE_ROWS", "150000"))
BASE_PATH = OUT / "base" / "base.parquet"
FUTURE_PATH = OUT / "future" / "d4_future.parquet"


def selected_features():
    if CFG["features"].get("selected"):
        return list(CFG["features"]["selected"])
    with open(RESULTS / "E0" / "feature_selection.json", encoding="utf-8") as fh:
        return json.load(fh)["selected"]


def _smoke_split(df):
    cut = int(pd.Timestamp("2023-07-01", tz="UTC").timestamp())
    df = df[df.split == 0].copy()
    df["split"] = np.where(df.block_time >= cut, 1, 0).astype(np.int8)
    rng = np.random.default_rng(0)
    parts = []
    for s in (0, 1):
        g = df[df.split == s]
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=min(SMOKE_ROWS, len(g)), replace=False))])
    return pd.concat(parts, ignore_index=True)


def load_base(columns=None):
    df = pd.read_parquet(BASE_PATH, columns=columns)
    if SMOKE:
        return _smoke_split(df)
    return df[df.split.isin([0, 1])].reset_index(drop=True)


def dev_test(columns=None):
    df = load_base(columns)
    dev = df[df.split == 0].reset_index(drop=True)
    test = df[df.split == 1].reset_index(drop=True)
    return dev, test


def load_future():
    if SMOKE:
        _, test = dev_test()
        fut = test.sample(frac=0.5, random_state=1).reset_index(drop=True)
        fut["eocj"] = (fut.L1 == 1).astype(int)
        fut["design_weight"] = 1.0
        fut["month"] = pd.to_datetime(fut.block_time, unit="s", utc=True).dt.strftime("%Y-%m")
        return fut
    return pd.read_parquet(FUTURE_PATH)


def load_elliptic():
    """Returns features frame (all rows), timestep, label (1 illicit, 0 licit, -1 unknown)."""
    d = CFG["paths"]["elliptic_dir"]
    feats = pd.read_csv(os.path.join(d, "elliptic_txs_features.csv"), header=None)
    cls = pd.read_csv(os.path.join(d, "elliptic_txs_classes.csv"))
    feats.columns = ["txId", "timestep"] + [f"f{i:03d}" for i in range(1, feats.shape[1] - 1)]
    df = feats.merge(cls, on="txId", how="left").copy()
    df["label"] = df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0}).fillna(-1).astype(int)
    if SMOKE:
        df = df[df.timestep <= CFG["elliptic"]["train_last_timestep"]].copy()
        df["timestep_eval"] = np.where(df.timestep >= 25, 35, 1)
    else:
        df["timestep_eval"] = df["timestep"]
    return df


def elliptic_train_mask(df):
    return df["timestep_eval"].to_numpy() <= CFG["elliptic"]["train_last_timestep"]
