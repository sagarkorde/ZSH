# -*- coding: utf-8 -*-
"""Does the zeta curve earn its place, or is it the ranking that does the work?

Table 3's existing "RF-importance" and "geometry-only" rows vary the ranking
criterion while applying the same zeta curve, so they never test the curve
itself. This crosses the two factors:

  ranking criterion : mutual information | RF importance | Laplacian/geometry
  weight curve      : uniform | direct scores | zeta rank-power at s

The ZSH pipeline (semantic seeds + Ward blend + balance refinement) is held
fixed; only the weight vector changes. Scored on the task-aligned metrics ZSH
claims, plus mean best-cluster enrichment across the eight rule families.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import (adjusted_mutual_info_score, balanced_accuracy_score,
                             f1_score, normalized_mutual_info_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BAL = os.environ.get("ZSH_OUTPUT_DIR", "outputs_balanced")
K, SEED, ALPHA = 30, 42, 0.60
WARD_MICRO, SEED_MIN_POINTS = 160, 500
MAX_SHARE, MAX_DEPTH = 0.10, 3
SAMPLE_N, TREE_DEPTH, TEST_SIZE = 200_000, 5, 0.30
REPEAT_SEEDS = [42, 123, 2026]

PRIORITY_RULES = [
    ("has_coinbase", "Coinbase"), ("is_coinjoin_like", "Coinjoin_Mixer"),
    ("is_batch_payment", "Batch_Payment"), ("is_consolidation", "Consolidation"),
    ("is_distribution", "Distribution"), ("is_peer_to_peer", "Standard_P2P"),
    ("has_op_return", "OP_Return"), ("rbf_enabled", "RBF_Enabled"),
]
FLAGS = [c for c, _ in PRIORITY_RULES]
T0 = time.time()


def ts(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def rule_labels(fr):
    lab = np.full(len(fr), "Unknown", dtype=object)
    free = np.ones(len(fr), dtype=bool)
    for col, name in PRIORITY_RULES:
        m = (fr[col].to_numpy(dtype=float) > 0) & free
        lab[m] = name
        free[m] = False
    return lab


def allocate(labels):
    cats, counts = np.unique(labels, return_counts=True)
    sc = {c: int(n) for c, n in zip(cats, counts)
          if c != "Unknown" and int(n) >= SEED_MIN_POINTS}
    tot = sum(sc.values())
    a = {c: max(1, round(K * n / tot)) for c, n in sc.items()}
    while sum(a.values()) > K:
        v = max(a, key=lambda c: (a[c], -sc[c]))
        if a[v] > 1:
            a[v] -= 1
        else:
            break
    while sum(a.values()) < K:
        a[max(a, key=lambda c: sc[c])] += 1
    return a


def seed_centers(X, labels):
    out = []
    for cat, k_sub in allocate(labels).items():
        Xc = X[labels == cat]
        if k_sub == 1:
            out.append(Xc.mean(axis=0))
            continue
        km = MiniBatchKMeans(k_sub, init="k-means++", n_init=5, batch_size=20_000,
                             max_iter=150, random_state=SEED).fit(Xc)
        out.extend(km.cluster_centers_)
    return np.asarray(out, dtype=np.float32)[:K]


def ward_centers(X):
    micro = MiniBatchKMeans(WARD_MICRO, init="k-means++", n_init=5, batch_size=50_000,
                            max_iter=200, random_state=SEED).fit(X)
    wl = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(
        micro.cluster_centers_)
    return np.vstack([micro.cluster_centers_[wl == c].mean(axis=0)
                      for c in range(K)]).astype(np.float32)


def blend(sc, wc):
    cost = ((sc[:, None, :] - wc[None, :, :]) ** 2).sum(axis=2)
    si, wi = linear_sum_assignment(cost)
    al = np.empty_like(wc)
    al[si] = wc[wi]
    return (ALPHA * sc + (1 - ALPHA) * al).astype(np.float32)


def refine(labels, X):
    labels = labels.copy()
    thr = int(MAX_SHARE * len(labels))
    nxt = int(labels.max()) + 1
    depth = {int(l): 0 for l in np.unique(labels)}
    q = list(depth.keys())
    while q:
        cl = q.pop(0)
        m = labels == cl
        if int(m.sum()) <= thr or depth.get(cl, 0) >= MAX_DEPTH:
            continue
        sub = KMeans(2, n_init=5, random_state=SEED).fit(X[m]).labels_
        idx = np.where(m)[0]
        labels[idx[sub == 1]] = nxt
        depth[nxt] = depth[cl] = depth[cl] + 1
        q += [cl, nxt]
        nxt += 1
    return labels


def score(labels, lab):
    n = len(lab)
    wp = mp = we = 0.0
    uniq = np.unique(labels)
    for c in uniq:
        sel = lab[labels == c]
        _, cnt = np.unique(sel, return_counts=True)
        p = cnt / cnt.sum()
        wp += (cnt.max() / len(sel)) * len(sel) / n
        mp += cnt.max() / len(sel)
        we += float(-(p * np.log2(p + 1e-12)).sum()) * len(sel) / n
    Xtr, Xte, ytr, yte = train_test_split(labels.reshape(-1, 1), lab,
                                          test_size=TEST_SIZE, random_state=SEED)
    pred = DecisionTreeClassifier(max_depth=TREE_DEPTH,
                                  random_state=SEED).fit(Xtr, ytr).predict(Xte)
    enr = []
    for fam in set(lab):
        if fam == "Unknown":
            continue
        t = (lab == fam)
        base = t.mean()
        if t.sum() < SEED_MIN_POINTS or base == 0:
            continue
        best = max((((labels == c) & t).sum() / max(1, (labels == c).sum()))
                   for c in uniq)
        enr.append(best / base)
    return {"weighted_purity": wp, "macro_purity": mp / len(uniq),
            "weighted_entropy": we,
            "nmi": normalized_mutual_info_score(lab, labels),
            "ami": adjusted_mutual_info_score(lab, labels),
            "tree_balanced_accuracy": balanced_accuracy_score(yte, pred),
            "tree_macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "mean_enrichment": float(np.mean(enr)), "clusters": int(len(uniq))}


# ------------------------------------------------------------ weight vectors --
cols = list(joblib.load(f"{BAL}/feature_cols.pkl"))
fr = pd.read_csv(f"{BAL}/feature_ranks.csv").set_index("feature")
rf = pd.read_parquet(f"{BAL}/feature_ranks_rf.parquet").set_index("feature")
ge = pd.read_parquet(f"{BAL}/feature_ranks_geometry.parquet").set_index("feature")


def vec(series, norm=True):
    v = np.array([float(series[c]) for c in cols], dtype=np.float32)
    return v / v.sum() if norm else v


SCHEMES = {
    "uniform (no weighting)":        np.full(len(cols), 1.0 / len(cols), np.float32),
    "MI direct":                     vec(fr["mi_score"]),
    "MI + zeta s=1.0":               vec(fr["w_s10"]),
    "MI + zeta s=1.5  (ZSH)":        vec(fr["w_s15"]),
    "MI + zeta s=2.0":               vec(fr["w_s20"]),
    "MI + zeta s=3.0":               vec(fr["w_s30"]),
    "RF direct":                     vec(rf["rf_importance"]),
    "RF + zeta s=1.5":               vec(rf["rf_zeta_weight"]),
    "Laplacian + zeta s=1.5":        vec(ge["geom_zeta_weight"]),
}
ts("weight schemes: " + ", ".join(SCHEMES))

X_all = np.load(f"{BAL}/X_scaled.npy", mmap_mode="r")
flags = pq.read_table(f"{BAL}/results_labeled.parquet", columns=FLAGS).to_pandas()

rows = []
for rs in REPEAT_SEEDS:
    rng = np.random.default_rng(rs)
    idx = np.sort(rng.choice(X_all.shape[0], size=SAMPLE_N, replace=False))
    Xs = np.asarray(X_all[idx], dtype=np.float32)
    lab = rule_labels(flags.iloc[idx])
    ts(f"repeat seed {rs}")
    for name, w in SCHEMES.items():
        Xw = Xs * w
        lb = refine(MiniBatchKMeans(K, init=blend(seed_centers(Xw, lab), ward_centers(Xw)),
                                    n_init=1, batch_size=20_000, max_iter=150,
                                    random_state=SEED).fit_predict(Xw), Xw)
        r = {"repeat": rs, "scheme": name, **score(lb, lab)}
        rows.append(r)
        ts(f"    {name:<24} purity={r['weighted_purity']:.4f} "
           f"NMI={r['nmi']:.4f} bal_acc={r['tree_balanced_accuracy']:.4f} "
           f"enrich={r['mean_enrichment']:.2f}x")

df = pd.DataFrame(rows)
df.to_csv(f"{BAL}/step_weighting_comparison_raw.csv", index=False)
agg = df.groupby("scheme").mean(numeric_only=True).drop(columns=["repeat"]).round(4)
agg = agg.reindex(list(SCHEMES))
agg.to_csv(f"{BAL}/step_weighting_comparison_summary.csv")
print("\n=== WEIGHTING SCHEME COMPARISON (mean of 3 repeats) ===")
print(agg.to_string())
ts("done")
