"""E15 (added after the freeze): how much of the concentration could better weights buy?

The weights of ZSH are learned without labels, from the mutual information between
each feature and a proxy partition. This experiment replaces that ranking with one
computed from the annotation itself on development data — an oracle that no
unsupervised method could have — and refits the pipeline. The resulting
concentration is an upper bound for what any rank-power weighting of these twelve
features can reach for that annotation, and the gap to ZSH says whether the limit is
the weighting or the feature set.

Two oracle variants per annotation: the same rank-power curve applied to the oracle
ranking (`oracle rank-power`), and weights proportional to the oracle mutual
information (`oracle MI-direct`). Everything else follows E10: the same 1,000,000-row
development sample, evaluation on the test period.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402
from zsh.weighting import rank_power, ranks_from_scores  # noqa: E402

# annotations with enough positives in the development period, chosen to span the range of
# attainment seen in Section 6.4: two that ZSH captures well and two that it does not
TARGETS = ["L2:P2PKH", "L2:P2SH", "L2:P2WSH", "L4:exchange"]
MI_ROWS = 200_000


def oracle_scores(X, y, is_binary, seed):
    """Mutual information between each feature and the annotation, on a balanced sample."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    take = min(len(pos), len(neg), MI_ROWS // 2)
    idx = np.sort(np.concatenate([rng.choice(pos, take, replace=False),
                                  rng.choice(neg, take, replace=False)]))
    return mutual_info_classif(X[idx], y[idx], discrete_features=np.asarray(is_binary),
                               n_neighbors=CFG["weighting"]["mi_neighbors"], random_state=seed)


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e15_oracle_weights{sfx}.log")
    res = results_dir("E15" + sfx)
    feats = selected_features()
    dev, test = dev_test()
    rng = rng_for("E10", "rows")                      # the same working sample as E10 and E11
    rows_n = min(CFG["e10"]["rows"], len(dev))
    sample = dev.iloc[np.sort(rng.choice(len(dev), size=rows_n, replace=False))].reset_index(drop=True)
    seed = seed_for("E10", "fit")
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    s = CFG["weighting"]["s"]

    prep = Preprocessor(feats).fit(sample)
    X = prep.transform(sample)
    dev_targets = independent_targets(sample)
    labels = {"reference (unsupervised)": None}
    info = []
    with threadpool_limits(CFG["evaluation"]["threads"]):
        t = time.time()
        ref = ZSH(feats, name="reference").fit(sample, seed)
        labels["reference (unsupervised)"] = ref.predict(test)
        log(f"reference: K={ref.k} ({time.time() - t:.0f}s)")
        info.append({"variant": "reference (unsupervised)", "target": "", "k": ref.k,
                     "top_feature": feats[int(np.argmax(ref.w))],
                     "weights": ";".join(f"{f}={w:.4f}" for f, w in zip(feats, ref.w))})

        for t_name in TARGETS:
            if t_name not in dev_targets:
                log(f"  {t_name}: not available in the development period; skipped")
                continue
            y = np.asarray(dev_targets[t_name], dtype=bool)
            sc = oracle_scores(X, y, prep.is_binary, seed_for("E15", "mi", t_name))
            for scheme in ("rank-power", "MI-direct"):
                w = (rank_power(ranks_from_scores(sc), s) if scheme == "rank-power"
                     else sc / sc.sum() if sc.sum() > 0 else np.full(len(sc), 1 / len(sc)))
                name = f"oracle {scheme}: {t_name}"
                t0 = time.time()
                m = ZSH(feats, fixed_w=w, name=name).fit(sample, seed)
                labels[name] = m.predict(test)
                info.append({"variant": name, "target": t_name, "k": m.k,
                             "top_feature": feats[int(np.argmax(w))],
                             "weights": ";".join(f"{f}={x:.4f}" for f, x in zip(feats, w))})
                log(f"  {name}: K={m.k}, top feature {feats[int(np.argmax(w))]} "
                    f"({time.time() - t0:.0f}s)")

        tab, _, skipped = evaluate_targets(labels, test, {k: v for k, v in independent_targets(test).items()
                                                          if k in TARGETS},
                                           B, seed_for("E15", "eval"), "reference (unsupervised)", log)
    tab.to_csv(res / "oracle_concentration.csv", index=False)
    pd.DataFrame(info).to_csv(res / "oracle_weights.csv", index=False)
    write_json({"rows": rows_n, "targets": TARGETS, "skipped": skipped, "s": s,
                "mi_rows": MI_ROWS}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
