"""E3: weighting x refinement factorial and weighting curves on full DEV; TEST evaluation (H1, H2).

Arms (all share the primary seed, so weights and initialisation procedure are identical
except for the factor that is varied):
  A1  rank-power (s=1.5) + refinement            = primary model (from E1)
  A2  uniform + refinement                        -> K_u
  A3  rank-power, no refinement, K = K*           (hierarchical init at K*)
  A4  uniform,    no refinement, K = K*           (hierarchical init at K*)
  A5  uniform,    no refinement, K = K_u          (pairs with A2)
  A6  MI-direct weights + refinement
  A7  Laplacian-score rank-power + refinement
  A9  uniform K-means++ (n_init=10), K = K*       (standard baseline, used in E6)
H1 primary contrast: A3 vs A4. H1 secondary: A1 vs A2.
H2 primary contrast: A1 vs A3. H2 secondary: A2 vs A5.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH, PlainKMeans  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets, structural_targets  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402
from zsh.metrics import intrinsic  # noqa: E402


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e03_factorial{sfx}.log")
    res = results_dir("E3" + sfx)
    mdir = out_dir("models" + sfx)
    ldir = out_dir("labels" + sfx)
    feats = selected_features()
    dev, test = dev_test()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    Kstar = primary.k
    seed = seed_for("E1", "primary")
    log(f"K* = {Kstar}")

    arms = {"A1": primary}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        def fit(name, model):
            t = time.time()
            model.fit(dev, seed, log=log)
            log(f"{name}: K={model.k} ({time.time() - t:.0f}s)")
            joblib.dump(model, mdir / f"e3_{name}.joblib")
            arms[name] = model
        fit("A2", ZSH(feats, weighting="uniform", name="uniform+refine"))
        Ku = arms["A2"].k
        fit("A3", PlainKMeans(feats, Kstar, weighting="rpw", init="ward", name="rpw K*"))
        fit("A4", PlainKMeans(feats, Kstar, weighting="uniform", init="ward", name="uniform K*"))
        fit("A5", PlainKMeans(feats, Ku, weighting="uniform", init="ward", name="uniform K_u"))
        fit("A6", ZSH(feats, weighting="mi_direct", name="MI-direct+refine"))
        fit("A7", ZSH(feats, weighting="laplacian_rpw", name="Laplacian rank-power+refine"))
        fit("A9", PlainKMeans(feats, Kstar, weighting="uniform", init="kmeans++", name="K-means++ K*"))

        common = Preprocessor(feats).fit(dev)
        rng = np.random.default_rng(seed_for("E3", "eval_rows"))
        idx = rng.choice(len(dev), size=min(CFG["e2"]["eval_rows"], len(dev)), replace=False)
        X_c = common.transform(dev.iloc[idx])
        rows, test_labels = [], {}
        for name, m in arms.items():
            lab_dev = m.labels_
            g = intrinsic(X_c, lab_dev[idx], seed_for("E3", "intr", name))
            g_own = intrinsic(m.transform(dev.iloc[idx]), lab_dev[idx], seed_for("E3", "own", name))
            test_labels[name] = m.predict(test)
            np.save(ldir / f"test_e3_{name}.npy", test_labels[name])
            np.save(ldir / f"dev_e3_{name}.npy", lab_dev)
            row = {"arm": name, "k": m.k, "max_share_dev": float(np.bincount(lab_dev).max() / len(dev)),
                   "max_share_test": float(np.bincount(test_labels[name]).max() / len(test)),
                   "fit_seconds": m.timing.get("total")}
            row.update({f"common_{k}": v for k, v in g.items()})
            row.update({f"own_{k}": v for k, v in g_own.items() if k in ("silhouette", "dbi", "chi")})
            rows.append(row)
    pd.DataFrame(rows).to_csv(res / "arms_geometry.csv", index=False)

    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    ind = independent_targets(test)
    struct = structural_targets(test)
    contrasts = {"all_vs_A1": (list(arms), "A1"),
                 "H1_primary_A3_vs_A4": (["A3", "A4"], "A4"),
                 "H1_secondary_A1_vs_A2": (["A1", "A2"], "A2"),
                 "H2_primary_A1_vs_A3": (["A1", "A3"], "A3"),
                 "H2_secondary_A2_vs_A5": (["A2", "A5"], "A5")}
    for cname, (members, ref) in contrasts.items():
        labs = {a: test_labels[a] for a in members}
        tab, curves, skipped = evaluate_targets(labs, test, ind, B, seed_for("E3", cname), ref, log)
        tab.to_csv(res / f"{cname}_independent.csv", index=False)
        tab2, _, _ = evaluate_targets(labs, test, struct, B, seed_for("E3", cname, "s"), ref, log)
        tab2.to_csv(res / f"{cname}_structural.csv", index=False)
        if cname == "all_vs_A1":
            write_json({"curves": curves, "skipped": skipped}, res / "curves_all.json")
    write_json({"K_star": Kstar, "K_u": Ku,
                "weights": {a: dict(zip(feats, np.asarray(m.winfo["w"]).round(6).tolist()))
                            for a, m in arms.items()},
                "ranks": {a: (dict(zip(feats, np.asarray(m.winfo["ranks"]).tolist()))
                              if "ranks" in m.winfo else None) for a, m in arms.items()}},
               res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
