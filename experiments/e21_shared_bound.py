"""E21 (exploratory, for discussion): how much of the supervised bound is lost by sharing
one partition across all annotations?

Section 6.8 compares the profiles with a supervised model fitted for one annotation at a
time. That comparison is unfair in one direction: the profiles are a single partition
that has to serve eleven annotations at once, while each supervised bound serves one.
This script measures the price of that constraint.

Three partitions of the test period are compared on the same eleven annotations with the
same cross-fitted protocol:

  profiles          the frozen ZSH model (unsupervised, one partition for all annotations)
  shared bound      one partition built with full label knowledge of all eleven: a
                    gradient-boosted tree is cross-fitted for each annotation, the eleven
                    out-of-fold scores of a transaction form its coordinates, and K-means
                    cuts that space into K* clusters
  single bound      the per-annotation bound of Section 6.8, repeated here for reference

Every score is out of fold: a transaction's coordinates come from models fitted on the
other block-parity half, and the cluster ranking is cross-fitted as everywhere else, so
no transaction is ranked or scored by a model that saw its own label. The shared bound is
still an upper bound rather than an achievable method, because building it needs labels
for every annotation.

Nothing here changes the frozen pipeline; it is an additional analysis.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402

MAX_TRAIN = 1_500_000


def bins_from_scores(score, k):
    qs = np.unique(np.quantile(score, np.linspace(0, 1, k + 1)[1:-1]))
    lab = np.searchsorted(qs, score, side="right").astype(np.int32)
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int32)


def oof_scores(X, y, parity, seed, rng):
    """Out-of-fold probabilities: fit on one block-parity half, score the other."""
    sc = np.empty(len(X), dtype=np.float64)
    for half in (0, 1):
        tr, ev = np.flatnonzero(parity == half), np.flatnonzero(parity == 1 - half)
        if len(tr) > MAX_TRAIN:
            tr = np.sort(rng.choice(tr, size=MAX_TRAIN, replace=False))
        m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, early_stopping=True,
                                           validation_fraction=0.1, random_state=seed + half)
        m.fit(X[tr], y[tr])
        sc[ev] = m.predict_proba(X[ev])[:, 1]
    return sc


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e21_shared_bound{sfx}.log")
    res = results_dir("E21" + sfx)
    mdir = out_dir("models" + sfx)
    feats = selected_features()
    _, test = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    K = primary.k
    X = test[feats].to_numpy(np.float32)
    parity = (test["block_height"].to_numpy() % 2).astype(np.int8)
    targets = independent_targets(test, include_eocj=True)
    targets = {k: v for k, v in targets.items() if v.sum() >= 200}
    rng = np.random.default_rng(seed_for("E21", "rows"))
    log(f"K* = {K}; TEST {len(test):,}; {len(targets)} annotations")

    with threadpool_limits(CFG["evaluation"]["threads"]):
        S, names = [], []
        for name, y in targets.items():
            t = time.time()
            S.append(oof_scores(X, y.astype(np.int8), parity, seed_for("E21", "oof", name), rng))
            names.append(name)
            log(f"  out-of-fold scores for {name} ({time.time() - t:.0f}s)")
        S = np.vstack(S).T
        np.save(out_dir("labels" + sfx) / "e21_oof_scores.npy", S.astype(np.float32))

        Z = StandardScaler().fit_transform(S)
        t = time.time()
        shared = KMeans(K, n_init=10, random_state=seed_for("E21", "kmeans")).fit_predict(Z)
        log(f"  shared supervised partition: K={len(np.unique(shared))} ({time.time() - t:.0f}s)")

        arms_common = {"profiles (ZSH)": primary.predict(test), "shared bound": shared}
        parts = []
        B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
        for j, name in enumerate(names):
            arms = dict(arms_common)
            arms["single bound"] = bins_from_scores(S[:, j], K)
            tab, _, _ = evaluate_targets(arms, test, {name: targets[name]}, B,
                                         seed_for("E21", "boot", name), reference="profiles (ZSH)", log=log)
            parts.append(tab)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out):
        out["attained"] = out["ap"]
        out["max_lift"] = 1.0 / out["base_rate"]
    out.to_csv(res / "shared_bound.csv", index=False)
    write_json({"K": K, "annotations": names, "rows": int(len(test)),
                "construction": "eleven out-of-fold gradient-boosted scores per transaction, "
                                "standardised, partitioned by K-means into K* clusters"},
               res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
