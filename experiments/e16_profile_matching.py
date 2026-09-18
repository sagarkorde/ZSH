"""E16 (added after the freeze): can profiles be followed across refits?

Section 6.3 shows that a model fitted in 2023 no longer describes 2026 activity. The
practical answer is to refit, but then the profiles of the new model have to be
matched to the old ones or the series breaks. This script tests how well that works
for the refits we already have (test period and prospective sample, E5): profiles are
matched by minimum-cost assignment on the transactions they share, and the match is
scored by the Jaccard similarity of the two member sets and by the distance between
the centroids in the original feature units.

Reported per period: how many of the 31 profiles find a match above 0.5 and above
0.75, what share of the period's transactions those matched profiles hold, and the
same for K-means++ as a reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from zsh.cluster import PlainKMeans  # noqa: E402
from zsh.config import Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, load_future, selected_features  # noqa: E402

PAIRS = [("Test", "test", "test_transfer_zsh.npy", "test_transfer_kmu.npy", "e5_refit_test.joblib"),
         ("Prospective", "future", "future_transfer_zsh.npy", "future_transfer_kmu.npy",
          "e5_refit_future.joblib")]


def match(old_labels, new_labels, weights=None):
    """Minimum-cost matching of two partitions of the same rows; returns Jaccard per old cluster."""
    w = np.ones(len(old_labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    ko, kn = int(old_labels.max()) + 1, int(new_labels.max()) + 1
    inter = np.zeros((ko, kn))
    np.add.at(inter, (old_labels, new_labels), w)
    size_o = inter.sum(1)
    size_n = inter.sum(0)
    union = size_o[:, None] + size_n[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    r, c = linear_sum_assignment(-jac)
    return jac, r, c, size_o


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e16_profile_matching{sfx}.log")
    res = results_dir("E16" + sfx)
    ldir = out_dir("labels" + sfx)
    mdir = out_dir("models" + sfx)
    dev, test = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    rows, per_profile = [], []
    for name, key, zsh_lab, kmu_lab, refit_file in PAIRS:
        if key == "future" and not (SMOKE or FUTURE_PATH.exists()):
            continue
        df = test if key == "test" else load_future()
        w = df["design_weight"].to_numpy() if "design_weight" in df else None
        refit = joblib.load(mdir / refit_file)
        for method, lab_file in (("ZSH", zsh_lab), ("K-means++", kmu_lab)):
            old = np.load(ldir / lab_file)
            if len(old) != len(df):
                log(f"skip {name} {method}: {len(old):,} labels but {len(df):,} rows (stale labels)")
                continue
            if method == "ZSH":
                new = refit.predict(df)
            else:
                # E5 did not save the K-means++ refit; refit it with the same seed
                p = mdir / f"e5_refit_{key}_kmu.joblib"
                if p.exists():
                    km = joblib.load(p)
                else:
                    km = PlainKMeans(selected_features(), primary.k, weighting="uniform",
                                     init="kmeans++").fit(df, seed_for("E5", "refit", name.upper()
                                                                      if name != "Prospective" else "FUTURE"))
                    joblib.dump(km, p)
                new = km.predict(df)
            jac, r, c, size_o = match(old, np.asarray(new), w)
            best = jac[r, c]
            share = size_o / size_o.sum()
            rows.append({"period": name, "method": method,
                         "profiles": int(len(size_o)), "refit_clusters": int(jac.shape[1]),
                         "matched_0.5": int((best >= 0.5).sum()), "matched_0.75": int((best >= 0.75).sum()),
                         "share_matched_0.5": float(share[r][best >= 0.5].sum()),
                         "share_matched_0.75": float(share[r][best >= 0.75].sum()),
                         "median_jaccard": float(np.median(best)),
                         "mean_jaccard": float(best.mean())})
            log(f"{name} {method}: {int((best >= 0.5).sum())} of {len(size_o)} profiles matched at 0.5 "
                f"({100 * share[r][best >= 0.5].sum():.0f}% of transactions), "
                f"{int((best >= 0.75).sum())} at 0.75")
            if method == "ZSH":
                for i, (oi, ni) in enumerate(zip(r, c)):
                    per_profile.append({"period": name, "profile": int(oi), "matched_cluster": int(ni),
                                        "jaccard": float(jac[oi, ni]), "share": float(share[oi])})
    pd.DataFrame(rows).to_csv(res / "matching.csv", index=False)
    pd.DataFrame(per_profile).to_csv(res / "matched_profiles.csv", index=False)
    write_json({"pairs": [p[0] for p in PAIRS], "primary_k": primary.k}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
