"""E11 (added after the freeze): sensitivity to the size of the proxy partition.

The rank-power weights come from the mutual information between each feature and a
proxy partition of K_p = 10 mini-batch K-means clusters. Reviewer 2 asked how the
result depends on K_p. Same 1,000,000-row development sample and the same evaluation
protocol as E10: K_p in {5, 10 (reference), 20, 50}, evaluated on the test period.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402
from zsh.metrics import intrinsic  # noqa: E402

KP_VALUES = [5, 20, 50]


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e11_proxy_k{sfx}.log")
    res = results_dir("E11" + sfx)
    feats = selected_features()
    dev, test = dev_test()
    # the same working sample as E10
    rng = rng_for("E10", "rows")
    rows_n = min(CFG["e10"]["rows"], len(dev))
    sample = dev.iloc[np.sort(rng.choice(len(dev), size=rows_n, replace=False))].reset_index(drop=True)
    seed = seed_for("E10", "fit")
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    kp_ref = CFG["weighting"]["proxy_k"]

    geo_idx = rng.choice(len(dev), size=min(CFG["e2"]["eval_rows"], len(dev)), replace=False)
    common = Preprocessor(feats).fit(dev)
    Xg = common.transform(dev.iloc[geo_idx])

    rows, labels, ranks = [], {}, {}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for kp in [kp_ref] + KP_VALUES:
            name = f"K_p={kp}" + (" (reference)" if kp == kp_ref else "")
            t = time.time()
            m = ZSH(feats, proxy_k=(None if kp == kp_ref else kp), name=name).fit(sample, seed)
            labels[name] = m.predict(test)
            ranks[name] = np.asarray(m.winfo["ranks"])
            g = intrinsic(Xg, m.predict(dev.iloc[geo_idx]), seed_for("E11", "geo", name))
            rows.append({"variant": name, "proxy_k": kp, "k": m.k,
                         "max_share_fit": float(m.shares_.max()),
                         "max_share_test": float(np.bincount(labels[name]).max() / len(test)),
                         "fit_seconds": time.time() - t,
                         **{f"common_{k}": v for k, v in g.items() if k in ("silhouette", "dbi", "chi")},
                         "weights": ";".join(f"{f}={w:.4f}" for f, w in zip(feats, m.w))})
            log(f"{name}: K={m.k} ({time.time() - t:.0f}s)")

        ref_name = f"K_p={kp_ref} (reference)"
        for r in rows:
            tau = kendalltau(ranks[ref_name], ranks[r["variant"]])
            r["kendall_tau_vs_reference"] = float(tau.statistic)
            r["top_feature"] = feats[int(np.argmin(ranks[r["variant"]]))]
        pd.DataFrame(rows).to_csv(res / "variants_geometry.csv", index=False)
        pd.DataFrame({"feature": feats, **{r["variant"]: ranks[r["variant"]] for r in rows}}).to_csv(
            res / "ranks.csv", index=False)

        ind, _, skipped = evaluate_targets(labels, test, independent_targets(test), B,
                                           seed_for("E11", "ind"), ref_name, log)
        ind.to_csv(res / "variants_independent.csv", index=False)
    write_json({"rows": rows_n, "proxy_k_reference": kp_ref, "proxy_k_values": KP_VALUES,
                "skipped": skipped}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
