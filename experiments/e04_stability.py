"""E4: stability of the full pipeline (seeds, block bootstrap, column-permuted reference).

Methods refitted on each replicate: ZSH (primary settings), K-means++ uniform at K* (A9),
rank-power K-means at K* without refinement (A3). Each replicate partition is compared with
the same method fitted on full DEV, on a fixed 200,000-row DEV reference set.
"""
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH, PlainKMeans  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402
from zsh.metrics import agreement, clusterwise_jaccard, matched_centroid_shift  # noqa: E402


def make_models(feats, Kstar):
    return {"ZSH": ZSH(feats),
            "KMeans++ K*": PlainKMeans(feats, Kstar, weighting="uniform", init="kmeans++"),
            "RPW K* (no refinement)": PlainKMeans(feats, Kstar, weighting="rpw", init="ward")}


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e04_stability{sfx}.log")
    res = results_dir("E4" + sfx)
    mdir = out_dir("models" + sfx)
    ec = CFG["e4"]
    n_seeds, n_boot, n_null = (3, 4, 3) if SMOKE else (ec["seeds"], ec["bootstrap"], ec["null_replicates"])
    rows_rep = min(ec["rows"], 50_000) if SMOKE else ec["rows"]
    feats = selected_features()
    dev, _ = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    full = {"ZSH": primary, "KMeans++ K*": joblib.load(mdir / "e3_A9.joblib"),
            "RPW K* (no refinement)": joblib.load(mdir / "e3_A3.joblib")}
    Kstar = primary.k

    rng = np.random.default_rng(seed_for("E4", "reference"))
    ref_idx = np.sort(rng.choice(len(dev), size=min(ec["reference_rows"], len(dev) // 2), replace=False))
    ref = dev.iloc[ref_idx].reset_index(drop=True)
    ref_labels = {m: full[m].labels_[ref_idx] for m in full}
    X_common = Preprocessor(feats).fit(dev).transform(ref)
    primary_ranks = np.asarray(primary.winfo["ranks"])

    blocks = dev.block_height.to_numpy()
    ub, binv = np.unique(blocks, return_inverse=True)
    order = np.argsort(binv, kind="stable")
    starts = np.searchsorted(binv[order], np.arange(len(ub)))
    ends = np.append(starts[1:], len(order))

    rep_rows, jac_rows, partitions = [], [], {}

    def run(kind, i, sample_idx, data=None, reference=None, models=None):
        data = dev if data is None else data
        reference = ref if reference is None else reference
        fit_df = data.iloc[sample_idx].reset_index(drop=True)
        seed = seed_for("E4", kind, i)
        for m, model in (models or make_models(feats, Kstar)).items():
            t = time.time()
            model.fit(fit_df, seed)
            lab = model.predict(reference)
            row = {"kind": kind, "replicate": i, "method": m, "k": model.k,
                   "fit_seconds": time.time() - t,
                   "max_share": float(np.bincount(model.labels_).max() / len(fit_df))}
            if kind != "null":
                row.update(agreement(ref_labels[m], lab))
                row["centroid_shift"] = matched_centroid_shift(X_common, ref_labels[m], lab)
                if "ranks" in model.winfo:
                    row["mi_rank_tau"] = float(kendalltau(primary_ranks, model.winfo["ranks"])[0])
                for c, j in clusterwise_jaccard(ref_labels[m], lab).items():
                    jac_rows.append({"kind": kind, "replicate": i, "method": m, "cluster": c, "jaccard": j})
            partitions.setdefault((kind, m), []).append(lab)
            rep_rows.append(row)
            log(f"{kind} {i} {m}: K={model.k} ARI={row.get('ari', np.nan):.3f} ({row['fit_seconds']:.0f}s)")

    with threadpool_limits(CFG["evaluation"]["threads"]):
        fixed = np.sort(np.random.default_rng(seed_for("E4", "seed_rows")).choice(len(dev), rows_rep, replace=False))
        for i in range(n_seeds):
            run("seed", i, fixed)
        for i in range(n_boot):
            r = np.random.default_rng(seed_for("E4", "boot_rows", i))
            pick = r.integers(len(ub), size=len(ub))
            pool = np.concatenate([order[starts[b]:ends[b]] for b in pick])
            run("bootstrap", i, np.sort(r.choice(pool, size=min(rows_rep, len(pool)), replace=False)))
        # no-structure reference: every feature column permuted independently
        rperm = np.random.default_rng(seed_for("E4", "null"))
        null_dev = dev.iloc[fixed][feats].reset_index(drop=True).copy()
        null_ref = ref[feats].copy()
        for f in feats:
            null_dev[f] = rperm.permutation(null_dev[f].to_numpy())
            null_ref[f] = rperm.permutation(null_ref[f].to_numpy())
        for i in range(n_null):
            run("null", i, np.arange(len(null_dev)), data=null_dev, reference=null_ref,
                models={"ZSH": ZSH(feats)})

    reps = pd.DataFrame(rep_rows)
    reps.to_csv(res / "replicates.csv", index=False)
    jac = pd.DataFrame(jac_rows)
    jac.to_csv(res / "clusterwise_jaccard_long.csv", index=False)

    pair_rows = []
    for (kind, m), labs in partitions.items():
        aris = [adjusted_rand_score(a, b) for a, b in combinations(labs, 2)]
        pair_rows.append({"kind": kind, "method": m, "pairs": len(aris),
                          "pairwise_ari_mean": float(np.mean(aris)) if aris else np.nan,
                          "pairwise_ari_sd": float(np.std(aris)) if aris else np.nan,
                          "pairwise_ari_min": float(np.min(aris)) if aris else np.nan})
    pd.DataFrame(pair_rows).to_csv(res / "pairwise.csv", index=False)

    per_cluster = (jac[jac.kind == "bootstrap"].groupby(["method", "cluster"]).jaccard
                   .agg(["mean", "min"]).reset_index())
    per_cluster.to_csv(res / "clusterwise_jaccard_bootstrap.csv", index=False)
    summ = {}
    for m, g in per_cluster.groupby("method"):
        summ[m] = {"clusters": int(len(g)), "mean_jaccard": float(g["mean"].mean()),
                   "n_ge_0.75": int((g["mean"] >= 0.75).sum()),
                   "n_0.5_to_0.75": int(((g["mean"] >= 0.5) & (g["mean"] < 0.75)).sum()),
                   "n_lt_0.5": int((g["mean"] < 0.5).sum())}
    agg = reps.groupby(["kind", "method"]).agg({c: ["mean", "std"] for c in
                                                ("ari", "ami", "vi_bits", "centroid_shift", "k", "mi_rank_tau")
                                                if c in reps}).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(res / "summary_by_method.csv", index=False)
    write_json({"clusterwise": summ, "K_star": Kstar, "rows_per_replicate": rows_rep,
                "reference_rows": len(ref)}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
