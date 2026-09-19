"""E18 (added after the freeze): how much of each annotation do the twelve features carry?

Sections 6.4 and 6.7 show that the profiles reach a small share of the attainable
concentration for rare annotations, and Section 6.6 shows that weights computed from
the annotation itself barely help. Two explanations remain: the twelve features do not
carry the annotation at all, or they carry it and the clustering objective does not
find it. This script separates them by replacing the clustering with a supervised
model on exactly the same twelve features and scoring it with exactly the same
protocol.

For every annotation a gradient-boosted tree is fitted on the features and its score
is cut into K* quantile bins. Those bins are then treated as a partition and passed
through the same cross-fitted ranking, block bootstrap and average-precision machinery
as the profiles, so the numbers are directly comparable with Tables 8 and 12. Two
variants are fitted:

  transfer   trained on the development period, evaluated on the test period; this is
             what a supervised model would actually deliver a year later.
  in-period  trained on one block-parity half of the test period and scored on the
             other, the same split the cluster ranking uses; this is the upper bound
             for what the twelve features can express about the annotation at all.

A high in-period number with a low profile number means the features carry the
annotation and the clustering objective misses it. A low in-period number means the
features do not carry it, and no clustering of them could.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402

MAX_TRAIN = 1_500_000        # rows used to fit each supervised model


def bins_from_scores(score, k, rng):
    """Cut a score into k quantile bins, returned as a partition label array."""
    qs = np.quantile(score, np.linspace(0, 1, k + 1)[1:-1])
    qs = np.unique(qs)
    lab = np.searchsorted(qs, score, side="right").astype(np.int32)
    # a degenerate score (many ties) can leave empty bins; relabel to a dense range
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int32)


def fit_predict(X_tr, y_tr, X_ev, seed):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                       early_stopping=True, validation_fraction=0.1,
                                       random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_ev)[:, 1]


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e18_feature_ceiling{sfx}.log")
    res = results_dir("E18" + sfx)
    mdir = out_dir("models" + sfx)
    feats = selected_features()
    dev, test = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    K = primary.k
    saved = out_dir("labels" + sfx) / "test_zsh.npy"
    zsh_test = np.load(saved) if saved.exists() else None
    if zsh_test is None or len(zsh_test) != len(test):   # stale labels from another run
        zsh_test = primary.predict(test)
    log(f"K* = {K}; DEV {len(dev):,}; TEST {len(test):,}")

    Xd = dev[feats].to_numpy(np.float32)
    Xt = test[feats].to_numpy(np.float32)
    parity = (test["block_height"].to_numpy() % 2).astype(np.int8)
    targets = independent_targets(test, include_eocj=True)
    dev_targets = independent_targets(dev, include_eocj=True)
    rng = np.random.default_rng(seed_for("E18", "rows"))

    label_sets = {"ZSH": zsh_test}
    rows = []
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for name, y_test in targets.items():
            if name not in dev_targets:
                log(f"  {name}: not available on development data, skipped")
                continue
            y_dev = dev_targets[name].astype(np.int8)
            if y_dev.sum() < 200 or y_test.sum() < 200:
                log(f"  {name}: too few positives, skipped")
                continue
            row = {"target": name, "positives_dev": int(y_dev.sum()), "positives_test": int(y_test.sum())}

            idx = np.sort(rng.choice(len(dev), size=min(MAX_TRAIN, len(dev)), replace=False))
            t = time.time()
            score = fit_predict(Xd[idx], y_dev[idx], Xt, seed_for("E18", "transfer", name))
            row["transfer_seconds"] = time.time() - t
            label_sets[f"supervised transfer: {name}"] = bins_from_scores(score, K, rng)

            # in-period: fit on one parity half, score the other, so no row scores its own model
            t = time.time()
            sc = np.empty(len(test), dtype=np.float64)
            for half in (0, 1):
                tr = np.flatnonzero(parity == half)
                ev = np.flatnonzero(parity == 1 - half)
                if len(tr) > MAX_TRAIN:
                    tr = np.sort(rng.choice(tr, size=MAX_TRAIN, replace=False))
                sc[ev] = fit_predict(Xt[tr], y_test.astype(np.int8)[tr], Xt[ev],
                                     seed_for("E18", "inperiod", name, half))
            row["inperiod_seconds"] = time.time() - t
            label_sets[f"supervised in-period: {name}"] = bins_from_scores(sc, K, rng)
            rows.append(row)
            log(f"  {name}: models fitted ({row['transfer_seconds']:.0f}s / {row['inperiod_seconds']:.0f}s)")

    pd.DataFrame(rows).to_csv(res / "models.csv", index=False)
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    # score each annotation against the profiles and its own two supervised arms only
    parts, skipped = [], {}
    for name, y in targets.items():
        arms = {"ZSH": label_sets["ZSH"]}
        for kind in ("transfer", "in-period"):
            key = f"supervised {kind}: {name}"
            if key in label_sets:
                arms[f"supervised {kind}"] = label_sets[key]
        if len(arms) == 1:
            continue
        t, _, sk = evaluate_targets(arms, test, {name: y}, B, seed_for("E18", "boot", name),
                                    reference="ZSH", log=log)
        skipped.update(sk)
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(tab):
        tab["max_lift"] = 1.0 / tab["base_rate"]
        tab["attained"] = tab["ap"]
    tab.to_csv(res / "feature_ceiling.csv", index=False)
    write_json({"K": K, "train_rows": int(min(MAX_TRAIN, len(dev))), "skipped": skipped,
                "model": "HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31)"},
               res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
