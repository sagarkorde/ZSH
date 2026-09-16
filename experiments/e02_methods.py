"""E2: method comparison at matched K on identical DEV samples; TEST concentration by transfer."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.baselines import BirchU, GMMU, HDBSCANU, KMeansU, MiniBatchU, VKVPartial, WardU  # noqa: E402
from zsh.cluster import ZSH  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets, structural_targets  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402
from zsh.metrics import intrinsic  # noqa: E402


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e02_methods{sfx}.log")
    res = results_dir("E2" + sfx)
    feats = selected_features()
    dev, test = dev_test()
    rng = rng_for("E2", "samples")
    perm = rng.permutation(len(dev))
    n_fit = min(CFG["e2"]["fit_rows"], len(dev) // 2)
    fit_df = dev.iloc[np.sort(perm[:n_fit])].reset_index(drop=True)
    ev_df = dev.iloc[np.sort(perm[n_fit:n_fit + min(CFG["e2"]["eval_rows"], len(dev) - n_fit)])].reset_index(drop=True)
    log(f"fit {len(fit_df):,}  eval {len(ev_df):,}  TEST {len(test):,}")

    common = Preprocessor(feats).fit(fit_df)
    X_ev = common.transform(ev_df)
    seed = seed_for("E2", "fit")
    threads = CFG["evaluation"]["threads"]

    rows, test_labels, models = [], {}, {}
    with threadpool_limits(threads):
        t = time.time()
        zsh = ZSH(feats).fit(fit_df, seed)
        zsh_time = time.time() - t
        K = zsh.k
        log(f"ZSH on fit sample: K={K} ({zsh_time:.1f}s)")
        methods = [("ZSH", zsh, zsh_time)]
        for cls in (KMeansU, MiniBatchU, GMMU, BirchU, WardU):
            m = cls(feats, K)
            t = time.time()
            m.fit(fit_df, seed)
            methods.append((m.name, m, time.time() - t))
            log(f"  {m.name}: {time.time() - t:.1f}s")
        for k in (CFG["e12"]["k_paper"], K):
            m = VKVPartial(k)
            t = time.time()
            m.fit(fit_df, seed)
            methods.append((m.name, m, time.time() - t))
            log(f"  {m.name}: {time.time() - t:.1f}s, PCA explained {np.round(m.explained, 3)}")

        for name, m, fit_s in methods:
            t = time.time()
            lab_ev = m.predict(ev_df)
            pred_s = time.time() - t
            row = {"method": name, "k_fit": getattr(m, "k", getattr(m, "kk", None)),
                   "fit_seconds": fit_s, "predict_eval_seconds": pred_s}
            g = intrinsic(X_ev, lab_ev, seed_for("E2", "intr", name))
            row.update({f"common_{k}": v for k, v in g.items()})
            own = m.transform(ev_df) if hasattr(m, "transform") else m.space(ev_df)
            g2 = intrinsic(own, lab_ev, seed_for("E2", "intr_own", name))
            row.update({f"own_{k}": v for k, v in g2.items() if k in ("silhouette", "dbi", "chi")})
            t = time.time()
            test_labels[name] = m.predict(test)
            row["predict_test_seconds"] = time.time() - t
            row["test_k_used"] = int(len(np.unique(test_labels[name])))
            row["test_max_share"] = float(np.bincount(test_labels[name]).max() / len(test))
            rows.append(row)
            log(f"  {name}: sil={g.get('silhouette', np.nan):.3f} dbi={g.get('dbi', np.nan):.3f}")

        hd = HDBSCANU(feats, None)
        t = time.time()
        hd.fit(fit_df, seed)
        hd_time = time.time() - t
        lab = hd.labels_
        hdrow = {"method": "HDBSCAN (own sample)", "k_fit": int(len(set(lab)) - (1 if -1 in lab else 0)),
                 "fit_seconds": hd_time, "noise_share": float(np.mean(lab == -1))}
        Xh = common.transform(fit_df.iloc[hd.idx])
        g = intrinsic(Xh, lab, seed_for("E2", "intr", "hdbscan"), noise_label=-1)
        hdrow.update({f"common_{k}": v for k, v in g.items()})
        rows.append(hdrow)
        log(f"  HDBSCAN: k={hdrow['k_fit']} noise={hdrow['noise_share']:.3f} ({hd_time:.1f}s)")

    tab = pd.DataFrame(rows)
    tab.to_csv(res / "methods_intrinsic_timing.csv", index=False)

    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    ind, curves, skipped = evaluate_targets(test_labels, test, independent_targets(test), B,
                                            seed_for("E2", "boot"), reference="ZSH", log=log)
    ind.to_csv(res / "methods_concentration_independent.csv", index=False)
    st, _, _ = evaluate_targets(test_labels, test, structural_targets(test), B,
                                seed_for("E2", "boot_struct"), reference="ZSH", log=log)
    st.to_csv(res / "methods_concentration_structural.csv", index=False)
    write_json({"K": K, "fit_rows": len(fit_df), "eval_rows": len(ev_df), "test_rows": len(test),
                "skipped_targets": skipped, "threads": threads, "curves": curves,
                "zsh_fit_sample_summary": {k: v for k, v in zsh.summary().items() if k != "preprocessor"}},
               res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
