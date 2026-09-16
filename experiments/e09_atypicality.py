"""E9: atypicality scores vs illicit status (Elliptic test timesteps, H6) and vs
emergent/independent annotations on the Bitcoin corpus (exploratory)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.data import TAG_BITS  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, elliptic_train_mask, load_elliptic, load_future  # noqa: E402
from zsh.metrics import ci  # noqa: E402


def scored(y, s, groups, B, seed):
    """ROC-AUC and PR-AUC with a bootstrap over groups (timesteps or blocks)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
           "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
    rng = np.random.default_rng(seed)
    ug, gi = np.unique(groups, return_inverse=True)
    order = np.argsort(gi, kind="stable")
    starts = np.searchsorted(gi[order], np.arange(len(ug)))
    ends = np.append(starts[1:], len(order))
    roc, pr = [], []
    for _ in range(B):
        pick = rng.integers(len(ug), size=len(ug))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        if y[idx].all() or not y[idx].any():
            continue
        roc.append(roc_auc_score(y[idx], s[idx]))
        pr.append(average_precision_score(y[idx], s[idx]))
    out["roc_auc_ci"], out["pr_auc_ci"] = ci(roc), ci(pr)
    lo, hi = out["roc_auc_ci"]
    out["h6_decision"] = ("supported (not positively associated)" if hi <= 0.5 else
                          "contradicted (positively associated)" if lo > 0.5 else "inconclusive")
    return out


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e09_atypicality{sfx}.log")
    res = results_dir("E9" + sfx)
    mdir = out_dir("models" + sfx)
    ldir = out_dir("labels" + sfx)
    ac = CFG["atypicality"]
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    out = {"elliptic": {}, "bitcoin": {}}

    with threadpool_limits(CFG["evaluation"]["threads"]):
        # ---------------- Elliptic ----------------
        df = load_elliptic()
        feats = [c for c in df.columns if c.startswith("f")]
        tr = elliptic_train_mask(df)
        train, test = df[tr].reset_index(drop=True), df[~tr].reset_index(drop=True)
        lab = test.label.to_numpy() >= 0
        y = test.label.to_numpy()[lab] == 1
        groups = test.timestep.to_numpy()[lab]
        zsh = joblib.load(mdir / "elliptic_AF-165_ZSH.joblib")
        Xw_tr, Xw_te = zsh.transform(train), zsh.transform(test)[lab]
        prep_u = Preprocessor(feats).fit(train)
        Xu_tr, Xu_te = prep_u.transform(train), prep_u.transform(test)[lab]
        scores = {}
        for name, (Xtr, Xte) in {"IF (rank-power space)": (Xw_tr, Xw_te),
                                 "IF (unweighted space)": (Xu_tr, Xu_te)}.items():
            iso = IsolationForest(n_estimators=ac["n_estimators"], max_samples=ac["max_samples"],
                                  random_state=seed_for("E9", name), n_jobs=CFG["evaluation"]["threads"])
            iso.fit(Xtr)
            scores[name] = -iso.score_samples(Xte)
        rng = np.random.default_rng(seed_for("E9", "lof_rows"))
        sub = rng.choice(len(Xw_tr), size=min(20_000, len(Xw_tr)), replace=False)
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xw_tr[sub])
        scores["LOF (rank-power space)"] = -lof.score_samples(Xw_te)
        _, d2 = zsh.predict(test[lab], return_dist=True)
        scores["distance to ZSH centroid"] = d2
        for name, s in scores.items():
            out["elliptic"][name] = scored(y, s, groups, B, seed_for("E9", "boot", name))
            log(f"Elliptic {name}: ROC-AUC {out['elliptic'][name]['roc_auc']:.3f} "
                f"{out['elliptic'][name]['roc_auc_ci']} -> {out['elliptic'][name]['h6_decision']}")
        pd.DataFrame({"timestep": groups, "illicit": y, **{k: v for k, v in scores.items()}}).to_parquet(
            out_dir("labels" + sfx) / "elliptic_test_atypicality.parquet", index=False)

        # ---------------- Bitcoin (exploratory) ----------------
        _, btest = dev_test()
        atyp = np.load(ldir / "test_atyp.npy")
        blocks = btest.block_height.to_numpy()
        Bx = 50 if SMOKE else 200
        targets = {"runes": btest.L3.to_numpy() == 1,
                   "exchange_tag": (btest.tags.to_numpy() & TAG_BITS["exchange"]) > 0,
                   "P2WSH_inputs": btest.L2.to_numpy() == 4}
        out["bitcoin"]["TEST"] = {k: scored(v, atyp, blocks, Bx, seed_for("E9", "btc", k))
                                  for k, v in targets.items() if 0 < v.sum() < len(v)}
        if SMOKE or FUTURE_PATH.exists():
            fut = load_future()
            model = joblib.load(mdir / "zsh_primary.joblib")
            iso = joblib.load(mdir / "iforest_primary.joblib")
            fa = -iso.score_samples(model.transform(fut))
            np.save(ldir / "future_atyp.npy", fa)
            ft = {"eocj": fut.eocj.to_numpy() == 1, "runes": fut.L3.to_numpy() == 1}
            out["bitcoin"]["FUTURE"] = {k: scored(v, fa, fut.block_height.to_numpy(), Bx, seed_for("E9", "fut", k))
                                        for k, v in ft.items() if 0 < v.sum() < len(v)}
    write_json(out, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
