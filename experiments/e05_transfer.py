"""E5: temporal transfer DEV -> TEST and DEV -> prospective (D4); drift; Runes emergence."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH, PlainKMeans  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.data import L1_NAMES, L2_NAMES  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, load_future, selected_features  # noqa: E402
from zsh.metrics import agreement, clusterwise_jaccard  # noqa: E402


def month_of(df):
    return pd.to_datetime(df.block_time, unit="s", utc=True).dt.strftime("%Y-%m")


def composition(codes, n, weights=None):
    w = np.ones(len(codes)) if weights is None else weights
    v = np.bincount(codes, weights=w, minlength=n)
    return v / v.sum() if v.sum() > 0 else v


def runes_concentration(labels, is_runes, k):
    per = np.bincount(labels[is_runes], minlength=k)
    tot = np.bincount(labels, minlength=k)
    order = np.argsort(-per)
    cum = np.cumsum(per[order]) / max(per.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(tot > 0, per / tot, 0)
    return {"runes": int(per.sum()),
            "profiles_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
            "profiles_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
            "profiles_runes_ge_90pct": int(((purity >= 0.9) & (tot > 0)).sum()),
            "runes_in_ge_90pct_profiles": float(per[purity >= 0.9].sum() / max(per.sum(), 1)),
            "per_profile_runes_share": purity.round(4).tolist()}


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e05_transfer{sfx}.log")
    res = results_dir("E5" + sfx)
    mdir = out_dir("models" + sfx)
    ldir = out_dir("labels" + sfx)
    feats = selected_features()
    dev, test = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    kmu = joblib.load(mdir / "e3_A9.joblib")
    K = primary.k
    periods = {"TEST": test}
    if SMOKE or FUTURE_PATH.exists():
        periods["FUTURE"] = load_future()
    out = {"K": K}

    transferred = {"DEV": {"ZSH": primary.labels_, "KMeans++ K*": kmu.labels_}}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for pname, df in periods.items():
            w = df["design_weight"].to_numpy() if "design_weight" in df else None
            lab = {"ZSH": primary.predict(df), "KMeans++ K*": kmu.predict(df)}
            transferred[pname] = lab
            np.save(ldir / f"{pname.lower()}_transfer_zsh.npy", lab["ZSH"])
            np.save(ldir / f"{pname.lower()}_transfer_kmu.npy", lab["KMeans++ K*"])
            seed = seed_for("E5", "refit", pname)
            t = time.time()
            refit = {"ZSH": ZSH(feats).fit(df, seed),
                     "KMeans++ K*": PlainKMeans(feats, K, weighting="uniform", init="kmeans++").fit(df, seed)}
            log(f"{pname}: refits done ({time.time() - t:.0f}s), ZSH refit K={refit['ZSH'].k}")
            res_p = {"rows": len(df)}
            for m in lab:
                a = agreement(lab[m], refit[m].labels_)
                jac = clusterwise_jaccard(lab[m], refit[m].labels_)
                a.update({"refit_k": refit[m].k,
                          "transfer_profiles_used": int(len(np.unique(lab[m]))),
                          "mean_best_jaccard": float(np.mean(list(jac.values()))),
                          "n_best_jaccard_ge_0.75": int(sum(v >= 0.75 for v in jac.values())),
                          "n_best_jaccard_lt_0.5": int(sum(v < 0.5 for v in jac.values())),
                          "best_jaccard": jac})
                res_p[m] = a
            if "L3" in df:
                runes = df["L3"].to_numpy() == 1
                if runes.sum() > 0:
                    res_p["runes_transfer_zsh"] = runes_concentration(lab["ZSH"], runes, K)
                    res_p["runes_refit_zsh"] = runes_concentration(refit["ZSH"].labels_, runes, refit["ZSH"].k)
                    res_p["runes_transfer_kmu"] = runes_concentration(lab["KMeans++ K*"], runes, K)
            # share of each transferred profile, and composition drift vs DEV
            drift = []
            for c in range(K):
                md, mp = primary.labels_ == c, lab["ZSH"] == c
                row = {"profile": c,
                       "share_dev": float(md.mean()),
                       "share_period": float(np.average(mp, weights=w))}
                for ann, n, names in (("L2", len(L2_NAMES), L2_NAMES), ("L1", len(L1_NAMES), L1_NAMES)):
                    pd_ = composition(dev[ann].to_numpy()[md], n)
                    pp = composition(df[ann].to_numpy()[mp], n, None if w is None else w[mp])
                    row[f"js_{ann}"] = float(jensenshannon(pd_, pp, base=2)) if mp.any() and md.any() else np.nan
                    row[f"top_{ann}_period"] = names[int(np.argmax(pp))] if mp.any() else ""
                for f in ("input_count", "output_count", "total_input_value", "fee_rate_sat_per_vbyte"):
                    row[f"median_{f}_dev"] = float(np.median(dev[f].to_numpy()[md])) if md.any() else np.nan
                    row[f"median_{f}_period"] = float(np.median(df[f].to_numpy()[mp])) if mp.any() else np.nan
                drift.append(row)
            pd.DataFrame(drift).to_csv(res / f"drift_{pname.lower()}.csv", index=False)
            out[pname] = res_p
            joblib.dump(refit["ZSH"], mdir / f"e5_refit_{pname.lower()}.joblib")

    # monthly shares of transferred ZSH profiles across all periods
    frames = [pd.DataFrame({"month": month_of(dev), "profile": primary.labels_, "w": 1.0})]
    for pname, df in periods.items():
        w = df["design_weight"].to_numpy() if "design_weight" in df else np.ones(len(df))
        frames.append(pd.DataFrame({"month": month_of(df), "profile": transferred[pname]["ZSH"], "w": w}))
    mon = pd.concat(frames)
    shares = (mon.groupby(["month", "profile"]).w.sum() / mon.groupby("month").w.sum()).rename("share")
    shares.reset_index().to_csv(res / "monthly_profile_shares.csv", index=False)
    write_json(out, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
