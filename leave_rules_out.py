# -*- coding: utf-8 -*-
"""Leave-one-family-out validation: is ZSH's advantage circular?

ZSH is seeded from eight rule-derived behavioural families and then evaluated on
alignment with those same families. This holds one family out of the seeding and
asks whether ZSH still recovers it better than an unseeded KMeans++ Elkan.

Three arms per fold, identical feature space and refinement:
  ZSH-LOO   seeded on the 7 retained families, held-out family unseen
  Elkan     no seeds at all
  ZSH-full  seeded on all 8 (the circular reference, for contrast)

Seeding, Ward blending and balance refinement mirror Step_5_ZSH_Clustering.py.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans

import os
BAL = os.environ.get("ZSH_OUTPUT_DIR", "outputs_balanced")
K, SEED, ALPHA = 30, 42, 0.60
WARD_MICRO, SEED_MIN_POINTS = 160, 500
MAX_SHARE, MAX_DEPTH = 0.10, 3
SAMPLE_N, REPEATS = 200_000, 3
REPEAT_SEEDS = [42, 123, 2026]

PRIORITY_RULES = [
    ("has_coinbase", "Coinbase"), ("is_coinjoin_like", "Coinjoin_Mixer"),
    ("is_batch_payment", "Batch_Payment"), ("is_consolidation", "Consolidation"),
    ("is_distribution", "Distribution"), ("is_peer_to_peer", "Standard_P2P"),
    ("has_op_return", "OP_Return"), ("rbf_enabled", "RBF_Enabled"),
]
FLAGS = [c for c, _ in PRIORITY_RULES]
FAMILIES = [n for _, n in PRIORITY_RULES]
T0 = time.time()


def ts(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def rule_labels(frame):
    lab = np.full(len(frame), "Unknown", dtype=object)
    free = np.ones(len(frame), dtype=bool)
    for col, name in PRIORITY_RULES:
        m = (frame[col].to_numpy(dtype=float) > 0) & free
        lab[m] = name
        free[m] = False
    return lab


def allocate(labels):
    cats, counts = np.unique(labels, return_counts=True)
    seed_cats = {c: int(n) for c, n in zip(cats, counts)
                 if c != "Unknown" and int(n) >= SEED_MIN_POINTS}
    tot = sum(seed_cats.values())
    alloc = {c: max(1, round(K * n / tot)) for c, n in seed_cats.items()}
    while sum(alloc.values()) > K:
        v = max(alloc, key=lambda c: (alloc[c], -seed_cats[c]))
        if alloc[v] > 1:
            alloc[v] -= 1
        else:
            break
    while sum(alloc.values()) < K:
        alloc[max(alloc, key=lambda c: seed_cats[c])] += 1
    return alloc


def seed_centers(X, labels, rng):
    out = []
    for cat, k_sub in allocate(labels).items():
        idx = np.where(labels == cat)[0]
        Xc = X[idx]
        if k_sub == 1:
            out.append(Xc.mean(axis=0))
            continue
        km = MiniBatchKMeans(n_clusters=k_sub, init="k-means++", n_init=5,
                             batch_size=20_000, max_iter=150, random_state=SEED)
        km.fit(Xc)
        out.extend(km.cluster_centers_)
    return np.asarray(out, dtype=np.float32)[:K]


def ward_centers(X):
    micro = MiniBatchKMeans(n_clusters=WARD_MICRO, init="k-means++", n_init=5,
                            batch_size=50_000, max_iter=200, random_state=SEED)
    micro.fit(X)
    wl = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(
        micro.cluster_centers_)
    return np.vstack([micro.cluster_centers_[wl == c].mean(axis=0)
                      for c in range(K)]).astype(np.float32)


def blend(sc, wc, alpha=ALPHA):
    cost = ((sc[:, None, :] - wc[None, :, :]) ** 2).sum(axis=2)
    si, wi = linear_sum_assignment(cost)
    aligned = np.empty_like(wc)
    aligned[si] = wc[wi]
    return (alpha * sc + (1 - alpha) * aligned).astype(np.float32)


def refine(labels, X):
    labels = labels.copy()
    thr = int(MAX_SHARE * len(labels))
    nxt = int(labels.max()) + 1
    depth = {int(l): 0 for l in np.unique(labels)}
    queue = list(depth.keys())
    while queue:
        cl = queue.pop(0)
        m = labels == cl
        if int(m.sum()) <= thr or depth.get(cl, 0) >= MAX_DEPTH:
            continue
        km = KMeans(2, n_init=5, random_state=SEED).fit(X[m])
        sub = km.labels_
        idx = np.where(m)[0]
        labels[idx[sub == 1]] = nxt
        depth[nxt] = depth[cl] + 1
        depth[cl] = depth[cl] + 1
        queue += [cl, nxt]
        nxt += 1
    return labels


def recover(labels, is_target):
    """Best single cluster for the held-out family: F1, precision, recall, enrichment."""
    base = is_target.mean()
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "enrichment": 0.0}
    for c in np.unique(labels):
        m = labels == c
        tp = int((m & is_target).sum())
        if tp == 0:
            continue
        prec = tp / int(m.sum())
        rec = tp / int(is_target.sum())
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": prec, "recall": rec,
                    "enrichment": prec / base if base > 0 else 0.0}
    return best


ts("loading features and flags ...")
w = np.asarray(joblib.load(f"{BAL}/zeta_weight_vector.pkl"), dtype=np.float32).ravel()
X_all = np.load(f"{BAL}/X_scaled.npy", mmap_mode="r")
flags = pq.read_table(f"{BAL}/results_labeled.parquet", columns=FLAGS).to_pandas()
ts(f"corpus {X_all.shape}, weights {w.shape}")

rows = []
for rep, rs in enumerate(REPEAT_SEEDS[:REPEATS]):
    rng = np.random.default_rng(rs)
    idx = np.sort(rng.choice(X_all.shape[0], size=SAMPLE_N, replace=False))
    Xw = np.asarray(X_all[idx], dtype=np.float32) * w
    lab = rule_labels(flags.iloc[idx])
    ts(f"repeat {rep + 1} (seed {rs}): sample {SAMPLE_N:,}")

    wc = ward_centers(Xw)

    # arm 2: unseeded Elkan
    elkan = refine(KMeans(K, init="k-means++", n_init=10, max_iter=250,
                          algorithm="elkan", random_state=SEED).fit_predict(Xw), Xw)
    # arm 3: ZSH seeded on all eight (circular reference)
    full = refine(MiniBatchKMeans(K, init=blend(seed_centers(Xw, lab, rng), wc),
                                  n_init=1, batch_size=20_000, max_iter=150,
                                  random_state=SEED).fit_predict(Xw), Xw)

    for fam in FAMILIES:
        is_t = (lab == fam)
        if is_t.sum() < SEED_MIN_POINTS:
            ts(f"    {fam}: only {int(is_t.sum())} rows, skipped")
            continue
        masked = np.where(is_t, "Unknown", lab)          # hide the family from seeding
        loo = refine(MiniBatchKMeans(K, init=blend(seed_centers(Xw, masked, rng), wc),
                                     n_init=1, batch_size=20_000, max_iter=150,
                                     random_state=SEED).fit_predict(Xw), Xw)
        r = {"repeat": rs, "family": fam, "support": int(is_t.sum()),
             "base_rate": float(is_t.mean())}
        for arm, L in (("zsh_loo", loo), ("elkan", elkan), ("zsh_full", full)):
            for k, v in recover(L, is_t).items():
                r[f"{arm}_{k}"] = round(v, 4)
        rows.append(r)
        ts(f"    {fam:<16} LOO F1={r['zsh_loo_f1']:.3f}  "
           f"Elkan F1={r['elkan_f1']:.3f}  full F1={r['zsh_full_f1']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(f"{BAL}/step_loo_rule_holdout_raw.csv", index=False)

agg = df.groupby("family").agg(
    support=("support", "mean"),
    zsh_loo_f1=("zsh_loo_f1", "mean"), elkan_f1=("elkan_f1", "mean"),
    zsh_full_f1=("zsh_full_f1", "mean"),
    zsh_loo_enrich=("zsh_loo_enrichment", "mean"),
    elkan_enrich=("elkan_enrichment", "mean")).reset_index()
agg["loo_minus_elkan"] = (agg.zsh_loo_f1 - agg.elkan_f1).round(4)
agg.to_csv(f"{BAL}/step_loo_rule_holdout_summary.csv", index=False)

print("\n=== LEAVE-ONE-FAMILY-OUT (mean over %d repeats) ===" % REPEATS)
print(agg.to_string(index=False))
wins = int((agg.loo_minus_elkan > 0).sum())
print(f"\nZSH-LOO beats unseeded Elkan on {wins} of {len(agg)} held-out families")
print("mean best-cluster F1  ZSH-LOO %.4f | Elkan %.4f | ZSH-full %.4f"
      % (agg.zsh_loo_f1.mean(), agg.elkan_f1.mean(), agg.zsh_full_f1.mean()))
json.dump({"families": len(agg), "wins": wins,
           "mean_f1_zsh_loo": float(agg.zsh_loo_f1.mean()),
           "mean_f1_elkan": float(agg.elkan_f1.mean()),
           "mean_f1_zsh_full": float(agg.zsh_full_f1.mean())},
          open(f"{BAL}/step_loo_rule_holdout.json", "w"), indent=2)
ts("done")
