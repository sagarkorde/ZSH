# -*- coding: utf-8 -*-
"""Bootstrap and permutation validation restricted to unique transactions.

Roughly 48% of the balanced corpus is duplicated minority-stratum rows, so the
published intervals are computed over a sample in which many observations are
exact copies. Identical rows sit at zero distance, which tightens clusters and
overstates the effective sample size. This repeats Section 5.1 drawing only from
rows whose feature vector occurs once, under the same protocol: 200 bootstrap
draws of 5,000 rows stratified by profile, and 300 label permutations.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

BAL = os.environ.get("ZSH_OUTPUT_DIR", "outputs_balanced")
N_BOOT, N_SAMPLE, N_PERM, SEED = 200, 5_000, 300, 42
T0 = time.time()


def ts(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


X = np.load(f"{BAL}/X_scaled.npy", mmap_mode="r")
labels = np.load(f"{BAL}/final_labels.npy")
w = np.asarray(joblib.load(f"{BAL}/zeta_weight_vector.pkl"), dtype=np.float32).ravel()
n = X.shape[0]
ts(f"corpus {n:,} rows")

# ---- locate rows whose feature vector occurs exactly once -------------------
ts("hashing feature rows to find duplicates ...")
h = np.empty(n, dtype=np.uint64)
CH = 1_000_000
for s in range(0, n, CH):
    blk = np.ascontiguousarray(np.asarray(X[s:s + CH], dtype=np.float64))
    acc = np.zeros(len(blk), dtype=np.uint64)
    for j in range(blk.shape[1]):
        acc = (acc * np.uint64(1099511628211)) ^ blk[:, j].view(np.uint64)
        acc ^= (acc >> np.uint64(29))
    h[s:s + CH] = acc
uniq_h, counts = np.unique(h, return_counts=True)
singleton = set(uniq_h[counts == 1].tolist())
mask = np.fromiter((v in singleton for v in h), dtype=bool, count=n)
uniq_idx = np.flatnonzero(mask)
ts(f"distinct feature vectors {len(uniq_h):,} | rows occurring exactly once "
   f"{len(uniq_idx):,} ({100*len(uniq_idx)/n:.1f}% of corpus)")

lab_u = labels[uniq_idx]
profiles, pcounts = np.unique(lab_u, return_counts=True)
ts(f"profiles represented among unique rows: {len(profiles)}")


def draw(rng):
    """Stratified draw of N_SAMPLE rows from the unique pool."""
    per = max(1, N_SAMPLE // len(profiles))
    picks = []
    for p in profiles:
        pool = uniq_idx[lab_u == p]
        if len(pool) == 0:
            continue
        picks.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    out = np.concatenate(picks)
    if len(out) > N_SAMPLE:
        out = rng.choice(out, size=N_SAMPLE, replace=False)
    return np.sort(out)


def metrics(idx):
    Xi = np.asarray(X[idx], dtype=np.float32) * w
    li = labels[idx]
    if len(np.unique(li)) < 2:
        return None
    return (silhouette_score(Xi, li), davies_bouldin_score(Xi, li),
            calinski_harabasz_score(Xi, li))


ts(f"bootstrap: {N_BOOT} draws of {N_SAMPLE:,} unique rows ...")
sil, dbi, chi = [], [], []
rng = np.random.default_rng(SEED)
for b in range(N_BOOT):
    m = metrics(draw(rng))
    if m:
        sil.append(m[0]); dbi.append(m[1]); chi.append(m[2])
    if (b + 1) % 50 == 0:
        ts(f"   {b+1}/{N_BOOT}  running mean Silhouette {np.mean(sil):.4f}")


def ci(v):
    a = np.asarray(v)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


s_m, s_lo, s_hi = ci(sil)
d_m, d_lo, d_hi = ci(dbi)
c_m, c_lo, c_hi = ci(chi)

ts(f"permutation test: {N_PERM} label shuffles ...")
idx = draw(np.random.default_rng(SEED))
Xi = np.asarray(X[idx], dtype=np.float32) * w
li = labels[idx]
observed = silhouette_score(Xi, li)
null = []
prng = np.random.default_rng(SEED)
for p in range(N_PERM):
    null.append(silhouette_score(Xi, prng.permutation(li)))
null = np.asarray(null)
pval = float((null >= observed).sum() + 1) / (N_PERM + 1)

pub = {"silhouette": (0.2159, 0.2069, 0.2256), "dbi": (0.9286, 0.8780, 0.9791),
       "chi": (7171.5, 5805.0, 11474.3), "observed": 0.2205,
       "null_mean": -0.7621, "null_sd": 0.0266}

print("\n=== Section 5.1 recomputed on unique rows only ===")
print("%-26s %-34s %s" % ("metric", "published (full corpus)", "unique rows only"))
print("%-26s %-34s %s" % ("Silhouette",
      "%.4f [%.4f, %.4f]" % pub["silhouette"], "%.4f [%.4f, %.4f]" % (s_m, s_lo, s_hi)))
print("%-26s %-34s %s" % ("Davies-Bouldin",
      "%.4f [%.4f, %.4f]" % pub["dbi"], "%.4f [%.4f, %.4f]" % (d_m, d_lo, d_hi)))
print("%-26s %-34s %s" % ("Calinski-Harabasz",
      "%.1f [%.1f, %.1f]" % pub["chi"], "%.1f [%.1f, %.1f]" % (c_m, c_lo, c_hi)))
print("%-26s %-34s %s" % ("Permutation observed",
      "%.4f" % pub["observed"], "%.4f" % observed))
print("%-26s %-34s %s" % ("Permutation null",
      "%.4f +/- %.4f" % (pub["null_mean"], pub["null_sd"]),
      "%.4f +/- %.4f" % (null.mean(), null.std())))
print("%-26s %-34s %s" % ("p-value", "< 0.001", "%.4f" % pval))

res = {"corpus_rows": int(n), "distinct_feature_vectors": int(len(uniq_h)),
       "rows_occurring_once": int(len(uniq_idx)),
       "unique_fraction": round(len(uniq_idx) / n, 4),
       "bootstrap_iterations": len(sil), "sample_size": N_SAMPLE,
       "silhouette": {"mean": round(s_m, 4), "ci": [round(s_lo, 4), round(s_hi, 4)]},
       "dbi": {"mean": round(d_m, 4), "ci": [round(d_lo, 4), round(d_hi, 4)]},
       "chi": {"mean": round(c_m, 1), "ci": [round(c_lo, 1), round(c_hi, 1)]},
       "permutation": {"observed": round(float(observed), 4),
                       "null_mean": round(float(null.mean()), 4),
                       "null_sd": round(float(null.std()), 4), "p_value": round(pval, 4)},
       "published_full_corpus": {k: v for k, v in pub.items()}}
json.dump(res, open(f"{BAL}/step7_dedup_bootstrap.json", "w"), indent=2)
pd.DataFrame({"silhouette": sil, "dbi": dbi, "chi": chi}).to_csv(
    f"{BAL}/step7_dedup_bootstrap_raw.csv", index=False)
ts("done")
