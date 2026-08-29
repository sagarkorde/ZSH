# -*- coding: utf-8 -*-
"""Does the Isolation Forest anomaly layer identify financially meaningful events?

The manuscript reports a 4.99% anomaly rate, but that follows from thresholding at
the 95th percentile: it demonstrates the threshold, not the detector. The Elliptic
corpus carries real illicit/licit ground truth, so the layer can be tested directly.

Isolation Forest is fitted on the zeta-weighted Elliptic space with the pipeline's
own settings (contamination 0.05, seed 42), and its continuous score is scored
against the illicit label with ranking metrics an analyst would actually care about.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

E = os.environ.get("ZSH_ELLIPTIC_DIR", "outputs_elliptic")
SEED, CONTAM = 42, 0.05
T0 = time.time()


def ts(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


X = np.asarray(np.load(f"{E}/X_elliptic_weighted.npy", mmap_mode="r"), dtype=np.float32)
y = pq.read_table(f"{E}/elliptic_filtered.parquet",
                  columns=["illicit"]).to_pandas()["illicit"].to_numpy().astype(int)
clusters = np.load(f"{E}/elliptic_cluster_labels.npy")
base = y.mean()
ts(f"{X.shape[0]:,} transactions x {X.shape[1]}d | illicit {y.sum():,} ({base*100:.2f}%)")

ts("fitting Isolation Forest (contamination=0.05, seed 42) ...")
iso = IsolationForest(contamination=CONTAM, random_state=SEED, n_jobs=-1).fit(X)
# score_samples: higher = more normal. Anomaly score is its negation.
anom = -iso.score_samples(X)
flag = iso.predict(X) == -1
ts(f"fitted; flagged {flag.sum():,} ({flag.mean()*100:.2f}%)")

roc = roc_auc_score(y, anom)
pr = average_precision_score(y, anom)
ts(f"ranking quality: ROC-AUC {roc:.4f} | PR-AUC {pr:.4f} (random PR-AUC = {base:.4f})")

rows = []
for K in (100, 500, 1000, 2329, 5000, 10000):
    top = np.argsort(-anom)[:K]
    hits = int(y[top].sum())
    prec = hits / K
    rows.append({"K": K, "illicit_found": hits, "precision_at_K": round(prec, 4),
                 "recall_at_K": round(hits / y.sum(), 4),
                 "enrichment_at_K": round(prec / base, 3),
                 "expected_random": round(K * base, 1)})
topk = pd.DataFrame(rows)
print()
print("=== ranking by anomaly score ===")
print(topk.to_string(index=False))

tp = int((flag & (y == 1)).sum())
prec_f = tp / max(1, flag.sum())
flagged = {"threshold": "95th percentile (contamination 0.05)",
           "n_flagged": int(flag.sum()),
           "precision": round(prec_f, 4),
           "recall": round(tp / y.sum(), 4),
           "enrichment": round(prec_f / base, 3)}
print()
print("=== the deployed binary flag ===")
for k, v in flagged.items():
    print(f"  {k:<16} {v}")

# For contrast: how well does cluster membership alone rank illicit transactions?
best = None
for c in np.unique(clusters):
    m = clusters == c
    if m.sum() < 50:
        continue
    p = y[m].mean()
    if best is None or p > best[1]:
        best = (int(c), p, int(m.sum()), int(y[m].sum()))
print()
print("=== contrast: best single cluster (Section 5.9) ===")
print(f"  cluster {best[0]}: {best[1]*100:.1f}% illicit, {best[1]/base:.2f}x base rate, "
      f"{best[3]:,} of {y.sum():,} illicit captured ({best[3]/y.sum()*100:.1f}%)")

# Does the anomaly score add anything inside that cluster?
m = clusters == best[0]
inner = roc_auc_score(y[m], anom[m]) if len(np.unique(y[m])) > 1 else float("nan")
print(f"  ROC-AUC of anomaly score within that cluster: {inner:.4f}")

res = {"n": int(X.shape[0]), "illicit": int(y.sum()), "base_rate": round(float(base), 4),
       "roc_auc": round(float(roc), 4), "pr_auc": round(float(pr), 4),
       "pr_auc_random": round(float(base), 4),
       "flag": flagged, "top_k": rows,
       "best_cluster": {"id": best[0], "illicit_rate": round(best[1], 4),
                        "enrichment": round(best[1] / base, 3),
                        "coverage": round(best[3] / y.sum(), 4)},
       "anomaly_auc_within_best_cluster": None if np.isnan(inner) else round(float(inner), 4)}
json.dump(res, open(f"{E}/step9b_anomaly_validation.json", "w"), indent=2)
topk.to_csv(f"{E}/step9b_anomaly_topk.csv", index=False)

with open(f"{E}/step9b_anomaly_validation.txt", "w", encoding="utf-8") as fh:
    fh.write("ANOMALY LAYER VALIDATION AGAINST ELLIPTIC ILLICIT LABELS\n")
    fh.write("=" * 62 + "\n")
    fh.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    fh.write(f"Isolation Forest, contamination {CONTAM}, seed {SEED}, fitted on the\n")
    fh.write("zeta-weighted Elliptic feature space (46,564 labelled transactions).\n\n")
    fh.write(f"  ROC-AUC {roc:.4f}   PR-AUC {pr:.4f}   (random PR-AUC {base:.4f})\n\n")
    fh.write(topk.to_string(index=False))
    fh.write("\n\nDeployed binary flag:\n")
    for k, v in flagged.items():
        fh.write(f"  {k:<16} {v}\n")
ts("done")
