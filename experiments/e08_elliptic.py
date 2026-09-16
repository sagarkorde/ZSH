"""E8: Elliptic temporal evaluation (H5). Fit on all transactions of timesteps 1-34,
rank clusters by labelled illicit rate in 1-34, evaluate on labelled transactions of 35-49."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.baselines import GMMU  # noqa: E402
from zsh.cluster import ZSH, PlainKMeans  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, seed_for, sha256_file, write_json  # noqa: E402
from zsh.io import SMOKE, elliptic_train_mask, load_elliptic  # noqa: E402
from zsh.metrics import ConcentrationData, concentration_many, holm  # noqa: E402

LAST = CFG["elliptic"]["train_last_timestep"]


def fold_fn(ub):
    return (ub > LAST).astype(np.int8)


def per_timestep(labels, df_lab, curve_order, train_cov):
    """Precision/recall per test timestep of the smallest train-ranked prefix covering 25% of train illicit."""
    n_top = int(np.searchsorted(train_cov, 0.25 - 1e-12)) + 1
    top = set(curve_order[:n_top])
    rows = []
    sel = np.isin(labels, list(top))
    for t, g in df_lab.assign(sel=sel).groupby("timestep_eval"):
        if t <= LAST:
            continue
        ill = g.label == 1
        rows.append({"timestep": int(t), "n": len(g), "illicit": int(ill.sum()), "flagged": int(g.sel.sum()),
                     "precision": float(ill[g.sel].mean()) if g.sel.any() else np.nan,
                     "recall": float(g.sel[ill].mean()) if ill.any() else np.nan})
    return rows, n_top


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e08_elliptic{sfx}.log")
    res = results_dir("E8" + sfx)
    mdir = out_dir("models" + sfx)
    ldir = out_dir("labels" + sfx)
    ec = CFG["elliptic"]
    if not SMOKE:
        for f in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv"):
            assert sha256_file(Path(CFG["paths"]["elliptic_dir"]) / f) == CFG["checksums"][f], f
    df = load_elliptic()
    feats_all = [c for c in df.columns if c.startswith("f")]
    feats_lf = feats_all[:ec["lf_features"]]
    tr = elliptic_train_mask(df)
    train, test = df[tr].reset_index(drop=True), df[~tr].reset_index(drop=True)
    log(f"train {len(train):,} (labelled {int((train.label >= 0).sum()):,}), test {len(test):,} "
        f"(labelled {int((test.label >= 0).sum()):,})")
    seed = seed_for("E8", "fit")
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    lab_rows = df.label.to_numpy() >= 0
    df_lab = df[lab_rows].reset_index(drop=True)
    target = df_lab.label.to_numpy() == 1
    block = df_lab.timestep_eval.to_numpy()

    summary, conc_rows, ts_rows = {}, [], []
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for setting, feats in (("AF-165", feats_all), ("LF-93", feats_lf)):
            t = time.time()
            zsh = ZSH(feats, k0=ec["k0"]).fit(train, seed, log=log)
            K = zsh.k
            log(f"{setting}: ZSH K={K} ({time.time() - t:.0f}s)")
            models = {"ZSH": zsh,
                      "K-means++ (K)": PlainKMeans(feats, K, weighting="uniform", init="kmeans++").fit(train, seed)}
            if setting == "AF-165":
                models["rank-power K-means (K)"] = PlainKMeans(feats, K, weighting="rpw", init="ward").fit(train, seed)
                models["uniform + refinement"] = ZSH(feats, weighting="uniform", k0=ec["k0"]).fit(train, seed)
                models["GMM diag (K)"] = GMMU(feats, K).fit(train, seed)
            labs = {m: np.asarray(mod.predict(df_lab)) for m, mod in models.items()}
            for m, mod in models.items():
                joblib.dump(mod, mdir / f"elliptic_{setting}_{m.split(' ')[0].replace('+', 'p')}.joblib")
            np.save(ldir / f"elliptic_{setting}_zsh_all.npy", zsh.predict(df))
            res_c, curves = concentration_many(labs, target, block, B, seed_for("E8", setting), reference="ZSH",
                                               fold_fn=fold_fn, directions=((0, 1),),
                                               min_members=ec["min_labelled_rank"])
            for m, r in res_c.items():
                row = {"setting": setting, "method": m, "k": int(getattr(models[m], "k", K)),
                       "base_rate": r["base_rate"], "ap": r["ap"], "ap_lift": r["ap_lift"]}
                for q in ("0.10", "0.25", "0.50"):
                    row[f"enrich@{q}"] = r[f"enrich@{q}"]
                    row[f"prec@{q}"] = r[f"prec@{q}"]
                    row[f"clusters@{q}"] = r[f"clusters@{q}"]
                for k2, (lo, hi) in r["ci"].items():
                    row[f"{k2}_lo"], row[f"{k2}_hi"] = lo, hi
                for k2, v in r.get("vs_ZSH", {}).items():
                    row[f"d_{k2}"], (row[f"d_{k2}_lo"], row[f"d_{k2}_hi"]), row[f"d_{k2}_p"] = v["diff"], v["ci"], v["p"]
                # per-timestep behaviour of the top clusters (25% of train illicit)
                cd = ConcentrationData(labs[m], target, block, fold_fn=fold_fn)
                n_tr, p_tr = cd.fold_counts(0)
                order = np.asarray(curves[m][0]["order"])
                train_cov = np.cumsum(p_tr[order]) / max(p_tr.sum(), 1)
                trows, n_top = per_timestep(labs[m], df_lab, order, train_cov)
                for tr_ in trows:
                    tr_.update({"setting": setting, "method": m, "top_clusters": n_top})
                ts_rows.extend(trows)
                row["top25_clusters"] = n_top
                conc_rows.append(row)
            summary[setting] = {"K": K, "weights_top10": sorted(zip(feats, zsh.w.round(5).tolist()),
                                                                 key=lambda t: -t[1])[:10],
                                "max_share": float(zsh.shares_.max()), "curves": curves,
                                "zsh_timing": zsh.timing}
            log(f"{setting}: " + ", ".join(f"{m} AP lift {r['ap_lift']:.2f}" for m, r in res_c.items()))

        # supervised reference ceiling (not an unsupervised method)
        ltr = train[train.label >= 0]
        lte = test[test.label >= 0]
        rf = RandomForestClassifier(n_estimators=ec["rf_n_estimators"],
                                    max_features=min(ec["rf_max_features"], len(feats_all)),
                                    n_jobs=CFG["evaluation"]["threads"], random_state=seed)
        rf.fit(ltr[feats_all], ltr.label)
        prob = rf.predict_proba(lte[feats_all])[:, 1]
        pred = (prob >= 0.5).astype(int)
        summary["random_forest_reference"] = {
            "precision": float(precision_score(lte.label, pred)), "recall": float(recall_score(lte.label, pred)),
            "f1": float(f1_score(lte.label, pred)), "ap": float(average_precision_score(lte.label, prob)),
            "base_rate": float(lte.label.mean())}
        log(f"RF reference: {summary['random_forest_reference']}")

    tab = pd.DataFrame(conc_rows)
    for s_, g in tab.groupby("setting"):
        sel = g.index[g.method != "ZSH"]
        if len(sel):
            tab.loc[sel, "d_ap_lift_p_holm"] = holm(tab.loc[sel, "d_ap_lift_p"].to_numpy())
    tab.to_csv(res / "concentration.csv", index=False)
    pd.DataFrame(ts_rows).to_csv(res / "per_timestep.csv", index=False)
    write_json(summary, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
