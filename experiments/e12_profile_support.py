"""E12 (added after the freeze): support intervals for each profile.

Reviewer 2 asked for profile-support intervals next to the assignment and centroid
stability of E4. For every profile this script reports its share of transactions in
each period with a block-bootstrap interval (blocks are the resampling unit; the
prospective sample uses its design weights), and the spread of its cluster-wise
Jaccard similarity across the 30 block-bootstrap refits of E4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, load_future  # noqa: E402


def share_ci(labels, blocks, k, weights=None, B=1000, seed=0):
    """Share of each profile with a percentile interval from resampling blocks."""
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    ub, inv = np.unique(blocks, return_inverse=True)
    # per-block totals: rows = blocks, columns = profiles
    tot = np.zeros((len(ub), k), dtype=np.float64)
    np.add.at(tot, (inv, labels), w)
    block_sum = tot.sum(axis=1)
    point = tot.sum(axis=0) / block_sum.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty((B, k), dtype=np.float64)
    for b in range(B):
        pick = rng.integers(len(ub), size=len(ub))
        s = tot[pick].sum(axis=0)
        draws[b] = s / s.sum()
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return point, lo, hi


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e12_profile_support{sfx}.log")
    res = results_dir("E12" + sfx)
    ldir = out_dir("labels" + sfx)
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    dev, test = dev_test()
    dev_lab = np.load(ldir / "dev_zsh.npy")
    k = int(dev_lab.max()) + 1

    def add(name, lab, blocks, wts, out):
        if len(lab) != len(blocks):
            log(f"skip {name}: {len(lab):,} labels but {len(blocks):,} rows (stale labels)")
            return
        out.append((name, lab, blocks, wts))

    periods = []
    add("Development", dev_lab, dev.block_height.to_numpy(), None, periods)
    add("Test", np.load(ldir / "test_zsh.npy"), test.block_height.to_numpy(), None, periods)
    if SMOKE or FUTURE_PATH.exists():
        fut = load_future()
        add("Prospective", np.load(ldir / "future_transfer_zsh.npy"), fut.block_height.to_numpy(),
            fut.design_weight.to_numpy(), periods)

    rows = {}
    for name, lab, blocks, wts in periods:
        point, lo, hi = share_ci(lab, blocks, k, wts, B, seed_for("E12", "share", name))
        rows[name] = (point, lo, hi)
        log(f"{name}: {len(lab):,} rows, {len(np.unique(blocks)):,} blocks, "
            f"widest interval {np.max(hi - lo) * 100:.2f} points")

    # spread of the cluster-wise Jaccard across the E4 bootstrap refits
    jac = pd.read_csv(results_dir("E4") / "clusterwise_jaccard_long.csv",
                      keep_default_na=False, na_values=[""])
    jz = jac[(jac.kind == "bootstrap") & (jac.method == "ZSH")]
    js = jz.groupby("cluster").jaccard.agg(["mean", lambda s: s.quantile(0.05),
                                            lambda s: s.quantile(0.95), "count"])
    js.columns = ["jaccard_mean", "jaccard_p05", "jaccard_p95", "replicates"]

    out = []
    for c in range(k):
        row = {"profile": c}
        for name, (point, lo, hi) in rows.items():
            key = name.lower()[:4]
            row[f"{key}_share"] = point[c]
            row[f"{key}_lo"] = lo[c]
            row[f"{key}_hi"] = hi[c]
        if c in js.index:
            row.update({m: float(js.loc[c, m]) for m in js.columns})
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(res / "profile_support.csv", index=False)
    write_json({"profiles": k, "bootstrap_B": B,
                "periods": {n: {"rows": int(len(l)), "blocks": int(len(np.unique(b)))}
                            for n, l, b, _ in periods},
                "widest_share_interval_points": {n: float(np.max(hi - lo) * 100)
                                                 for n, (p, lo, hi) in rows.items()}},
               res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
