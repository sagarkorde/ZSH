"""E20 (exploratory, for discussion): does a refit constrained to the previous centroids
let profiles be followed?

Section 6.3 reports that a refit on a later period produces a partition whose clusters
cannot be matched to the development profiles: no profile reaches a Jaccard similarity
of 0.5. Section 7.3 suggests constraining the refit towards the previous centroids as a
remedy but does not test it. This script tests it.

The constrained refit keeps everything the frozen model learned about the feature space
-- the scaler and the rank-power weights -- and re-estimates only the centroids on the
later period, starting Lloyd's algorithm from the development centroids. The result is
compared with the frozen model transferred unchanged and with the cold refit of
Section 6.3 on two questions: how far the partition moves (ARI against the transferred
labels) and whether the profiles can still be followed (Jaccard of the minimum-cost
match with the development profiles).

Nothing here changes the frozen pipeline; it is an additional analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from e16_profile_matching import match  # noqa: E402
from zsh.cluster import ZSH, assign  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, load_future, selected_features  # noqa: E402
from zsh.metrics import agreement  # noqa: E402

MAX_ITER = 100
TOL = 1e-6


def lloyd_from(Xw, C0, max_iter=MAX_ITER, tol=TOL):
    """Lloyd iterations started at C0; empty clusters keep their previous centroid."""
    C = C0.astype(np.float64).copy()
    for it in range(max_iter):
        lab, _ = assign(Xw, C)
        newC = C.copy()
        cnt = np.bincount(lab, minlength=len(C))
        for c in np.flatnonzero(cnt):
            newC[c] = Xw[lab == c].mean(0)
        shift = float(np.sqrt(((newC - C) ** 2).sum(1)).max())
        C = newC
        if shift <= tol:
            break
    lab, _ = assign(Xw, C)
    return C, lab, it + 1, int((np.bincount(lab, minlength=len(C)) == 0).sum())


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e20_warm_refit{sfx}.log")
    res = results_dir("E20" + sfx)
    mdir = out_dir("models" + sfx)
    feats = selected_features()
    _, test = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    K = primary.k
    periods = {"test": test}
    if SMOKE or FUTURE_PATH.exists():
        periods["future"] = load_future()
    log(f"K* = {K}")

    rows, per_profile, labels = [], [], {}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for pname, df in periods.items():
            w = df["design_weight"].to_numpy() if "design_weight" in df else None
            transferred = primary.predict(df)
            Xw = primary.transform(df)                     # frozen scaler and weights
            C1, warm, iters, empty = lloyd_from(Xw, primary.centroids)
            labels[pname] = {"transferred": transferred, "warm": warm}
            shift = np.sqrt(((C1 - primary.centroids) ** 2).sum(1))
            log(f"{pname}: {iters} Lloyd iterations, {empty} empty clusters, "
                f"mean centroid shift {shift.mean():.3f}")

            row = {"period": pname, "rows": len(df), "iterations": iters, "empty_clusters": empty,
                   "mean_centroid_shift": float(shift.mean()), "max_centroid_shift": float(shift.max())}
            row.update({f"warm_vs_transferred_{k}": v for k, v in agreement(transferred, warm).items()})
            jac, r, c, size_o = match(transferred, warm, w)
            matched = jac[r, c]
            tot = size_o.sum()
            row.update({"profiles": int(len(size_o)),
                        "median_matched_jaccard": float(np.median(matched)),
                        "n_matched_ge_0.5": int((matched >= 0.5).sum()),
                        "n_matched_ge_0.75": int((matched >= 0.75).sum()),
                        "share_matched_ge_0.5": float(size_o[r][matched >= 0.5].sum() / tot)})
            # cold refit of the same period, for reference (Section 6.3 protocol)
            cold_path = mdir / f"e5_refit_{pname}.joblib"
            if cold_path.exists():
                cold = joblib.load(cold_path)
                cl = cold.predict(df)
                jc, rc, cc, so = match(transferred, cl, w)
                mc = jc[rc, cc]
                row.update({"cold_refit_k": int(cold.k),
                            "cold_ari": agreement(transferred, cl)["ari"],
                            "cold_median_matched_jaccard": float(np.median(mc)),
                            "cold_n_matched_ge_0.5": int((mc >= 0.5).sum())})
            rows.append(row)
            for i in range(len(size_o)):
                per_profile.append({"period": pname, "profile": i, "jaccard": float(jac[i, c[list(r).index(i)]])
                                    if i in list(r) else np.nan,
                                    "share": float(size_o[i] / tot),
                                    "centroid_shift": float(shift[i]) if i < len(shift) else np.nan})

        # does the constrained refit keep or improve what the profiles concentrate?
        parts = []
        B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
        for pname, df in periods.items():
            arms = {"frozen (transferred)": labels[pname]["transferred"],
                    "constrained refit": labels[pname]["warm"]}
            tab, _, _ = evaluate_targets(arms, df, independent_targets(df, include_eocj=True), B,
                                         seed_for("E20", "boot", pname), reference="frozen (transferred)",
                                         log=log)
            if len(tab):
                tab["period"] = pname
                tab["attained"] = tab["ap"]
                tab["max_lift"] = 1.0 / tab["base_rate"]
            parts.append(tab)
    pd.DataFrame(rows).to_csv(res / "warm_refit.csv", index=False)
    pd.DataFrame(per_profile).to_csv(res / "warm_refit_profiles.csv", index=False)
    (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_csv(
        res / "warm_refit_concentration.csv", index=False)
    write_json({"K": K, "periods": {p: len(d) for p, d in periods.items()},
                "note": "scaler and weights frozen; only centroids re-estimated, started at the "
                        "development centroids"}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
