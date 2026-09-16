# ============================================================
# STEP 9 — Elliptic Replication Study (reviewer #6, external validation)
#
# WHY THIS SCRIPT EXISTS
#   Step 8 validates ZSH against the pipeline's OWN rule-derived heuristic
#   labels — the code's own docstring already flags this as
#   "profiling-alignment evidence, not independent external ground-truth"
#   (see Step_8_Contextual_Profiling_Comparison.py). The reviewer wants
#   real external ground truth (Elliptic / Chainalysis / public labeled
#   data), specifically testing illicit / coinjoin-like transactions.
#
#   Elliptic's public release exposes 165 anonymized PCA-derived features
#   per transaction — NOT the raw fee/value/address-count fields this
#   pipeline's Steps 2-3 use — so there is no column-level crosswalk onto
#   the custom dataset's feature space, and the two datasets' transaction
#   IDs almost certainly do not overlap (different collection eras).
#   STAGE 0 below checks the ID-overlap possibility directly rather than
#   assuming it; if empty (expected), this script proceeds as an
#   INDEPENDENT REPLICATION STUDY: the ZSH methodology (proxy-clustering +
#   mutual-information Zeta weighting -> clustering) is refit natively on
#   Elliptic's own schema and evaluated against Elliptic's own illicit/
#   licit ground-truth labels. This tests whether the METHOD generalizes,
#   not whether the two corpora cross-reference.
#
#   The rule-based semantic seeding used in Step 5 (PRIORITY_RULES,
#   is_coinjoin_like / has_op_return / etc.) CANNOT transfer here —
#   Elliptic's anonymized features have no equivalent flags — so this
#   script only uses the seeding-agnostic parts of the pipeline:
#   proxy-clustering + MI ranking (STAGE 3, feature-name-agnostic) and
#   Ward-guided / plain KMeans++ candidates (STAGE 5, no semantic seeds).
#
# LIMITATION (stated explicitly, also in the final report):
#   Elliptic's public labels are binary illicit(1) / licit(2) / unknown
#   only — there is no coinjoin/mixing sub-category. This script can
#   test illicit-vs-licit separability but NOT the reviewer's specific
#   ask about coinjoin/mixing transactions without non-public Elliptic
#   metadata.
# ============================================================

import io
import logging
import os
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import (
    ELLIPTIC_FEATURES_CSV, ELLIPTIC_CLASSES_CSV, ELLIPTIC_OUTPUT_DIR,
    PARQUET_PATH,
)

OUTPUT_DIR = ELLIPTIC_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

log_path = os.path.join(OUTPUT_DIR, "step9_log.txt")
_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=_utf8),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
_T0 = time.time()


def ts(msg):
    log.info(f"[{time.time() - _T0:7.1f}s]  {msg}")


def p(name):
    return os.path.join(OUTPUT_DIR, name)


def ckpt(stage):
    return p(f".ckpt9_{stage}.done")


def is_done(stage):
    return os.path.exists(ckpt(stage))


def mark_done(stage):
    Path(ckpt(stage)).touch()
    ts(f"  [CHECKPOINT] {stage} ✓")


RANDOM_STATE = 42
S_DECAY = 1.5              # same primary decay exponent as the main pipeline
PROXY_N_CLUSTERS = 10       # same as Step_3_Zeta_Weighting.py STAGE 1
MI_NEIGHBORS = 5
K_SWEEP = [10, 20, 30]      # K is NOT assumed to transfer from the custom corpus
FIT_SAMPLE = 40_000         # Elliptic labeled subset is ~46.5K rows, use ~all
VAL_FRACTION = 0.15
WARD_MICRO_CLUSTERS = 160
BALANCE_MAX_SHARE = 0.10
BALANCE_MAX_DEPTH = 3
BALANCE_SPLIT_FIT_SAMPLE = 30_000


def apply_balance_refinement(labels, X_ref, max_share=BALANCE_MAX_SHARE,
                              max_depth=BALANCE_MAX_DEPTH,
                              fit_sample=BALANCE_SPLIT_FIT_SAMPLE,
                              random_state=RANDOM_STATE):
    """Same balance-constrained refinement as Step_5_ZSH_Clustering.py:
    recursively bisect any cluster exceeding max_share of the corpus."""
    labels = labels.copy()
    n_total = len(labels)
    threshold_n = int(max_share * n_total)
    next_label = int(labels.max()) + 1
    depth_map = {int(lbl): 0 for lbl in np.unique(labels)}
    to_process = list(depth_map.keys())
    splits_log = []

    while to_process:
        cl = to_process.pop(0)
        cl_mask = labels == cl
        cl_n = int(cl_mask.sum())
        if cl_n <= threshold_n or depth_map.get(cl, 0) >= max_depth:
            continue
        k_split = max(2, int(np.ceil(cl_n / threshold_n)))
        cl_indices = np.where(cl_mask)[0]
        X_cl = X_ref[cl_indices]
        rng_local = np.random.default_rng(random_state + cl * 97 + depth_map[cl])
        fit_n = min(len(X_cl), fit_sample)
        fit_idx_local = rng_local.choice(len(X_cl), size=fit_n, replace=False)
        ts(f"  [Balance refine] cluster {cl}: n={cl_n:,} ({cl_n/n_total*100:.1f}% > "
           f"{max_share*100:.0f}% cap) -> splitting into {k_split} sub-clusters ...")
        km_split = MiniBatchKMeans(n_clusters=k_split, init="k-means++", n_init=5,
                                    batch_size=5_000, max_iter=300, random_state=random_state)
        km_split.fit(X_cl[fit_idx_local])
        sub_labels = km_split.predict(X_cl)
        depth = depth_map[cl] + 1
        new_ids = []
        for sub in np.unique(sub_labels):
            sub_mask_local = sub_labels == sub
            new_id = cl if sub == 0 else next_label
            if sub != 0:
                next_label += 1
            labels[cl_indices[sub_mask_local]] = new_id
            depth_map[new_id] = depth
            new_ids.append(int(new_id))
        splits_log.append({"parent": int(cl), "parent_n": cl_n, "k_split": k_split,
                            "children": new_ids, "depth": depth})
        to_process.extend(new_ids)
    return labels, splits_log


ts("=" * 70)
ts("STEP 9 — Elliptic Replication Study (reviewer #6, external validation)")
ts("=" * 70)

# ============================================================
# STAGE 0 — txid overlap sanity check
# ============================================================
if is_done("overlap_check"):
    ts("STAGE 0 [Overlap Check]: Already done — see step9_txid_overlap.txt")
else:
    ts("STAGE 0 [Overlap Check]: Comparing Elliptic txIds against the custom "
       "dataset's txid column ...")
    t0 = time.time()
    con = duckdb.connect()
    try:
        overlap_row = con.execute(f"""
            SELECT COUNT(*) AS n_overlap
            FROM read_csv_auto('{ELLIPTIC_CLASSES_CSV.replace(chr(92), '/')}') e
            JOIN read_parquet('{PARQUET_PATH.replace(chr(92), '/')}') c
              ON CAST(e.txId AS VARCHAR) = CAST(c.txid AS VARCHAR)
        """).fetchone()
        n_overlap = int(overlap_row[0])
        overlap_note = f"txid overlap between Elliptic and custom dataset: {n_overlap:,} rows"
    except Exception as exc:
        n_overlap = None
        overlap_note = (
            f"Overlap check could not run (likely: custom dataset has no 'txid' "
            f"column at the parquet level, or dtype mismatch). Error: {exc}\n"
            f"Proceeding as an independent replication study regardless."
        )
    con.close()
    ts(f"  {overlap_note}  ({time.time()-t0:.1f}s)")
    with open(p("step9_txid_overlap.txt"), "w", encoding="utf-8") as f:
        f.write(overlap_note + "\n")
    mark_done("overlap_check")

# ============================================================
# STAGE 1 — Load & filter Elliptic
# ============================================================
if is_done("load"):
    ts("STAGE 1 [Load]: Loading cached filtered Elliptic frame ...")
    df = pd.read_parquet(p("elliptic_filtered.parquet"))
else:
    ts("STAGE 1 [Load]: Reading Elliptic features + classes ...")
    t0 = time.time()

    # No header row: col0=txId, col1=timestep, col2..166 = 165 anonymized features
    feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, 166)]
    df_feat = pd.read_csv(ELLIPTIC_FEATURES_CSV, header=None, names=feat_cols)
    df_cls = pd.read_csv(ELLIPTIC_CLASSES_CSV)  # header: txId,class

    df = df_feat.merge(df_cls, on="txId", how="inner")
    ts(f"  Merged: {df.shape}  (class distribution: "
       f"{df['class'].value_counts().to_dict()})")

    df = df[df["class"] != "unknown"].reset_index(drop=True)
    df["illicit"] = (df["class"] == "1").astype(np.int8)  # 1=illicit, 2=licit
    ts(f"  After dropping 'unknown': {df.shape}  "
       f"illicit={df['illicit'].sum():,} ({df['illicit'].mean()*100:.2f}%)  "
       f"licit={(1-df['illicit']).sum():,}")

    df.to_parquet(p("elliptic_filtered.parquet"), index=False)
    ts(f"  STAGE 1 done in {time.time()-t0:.1f}s")
    mark_done("load")

FEATURE_COLS = [c for c in df.columns if c.startswith("feat_")]
n_samples = len(df)
n_features = len(FEATURE_COLS)
ts(f"  n_samples={n_samples:,}  n_features={n_features}")

# ============================================================
# STAGE 2 — RobustScaler
# ============================================================
scaled_path = p("X_elliptic_scaled.npy")
if is_done("scale") and os.path.exists(scaled_path):
    ts("STAGE 2 [Scale]: Loading cached scaled matrix ...")
    X_scaled = np.load(scaled_path)
else:
    ts("STAGE 2 [Scale]: Fitting RobustScaler on the 165 anonymized features ...")
    t0 = time.time()
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].values).astype(np.float32)
    np.save(scaled_path, X_scaled)
    ts(f"  X_scaled shape={X_scaled.shape}  ({time.time()-t0:.1f}s)")
    mark_done("scale")

illicit_labels = df["illicit"].to_numpy(dtype=np.int8)

# ============================================================
# STAGE 3 — Native Zeta weighting (proxy-clustering + MI, no rule seeding)
# ============================================================
weighted_path = p("X_elliptic_weighted.npy")
ranks_path = p("feature_ranks_elliptic.parquet")

if is_done("zeta") and os.path.exists(weighted_path):
    ts("STAGE 3 [Zeta Weighting]: Loading cached weighted matrix ...")
    X_weighted = np.load(weighted_path)
    feature_rank_df = pd.read_parquet(ranks_path)
else:
    ts("STAGE 3 [Zeta Weighting]: Proxy MiniBatchKMeans -> MI -> Zeta ranks ...")
    t0 = time.time()

    proxy = MiniBatchKMeans(
        n_clusters=PROXY_N_CLUSTERS, n_init=5, batch_size=5_000,
        max_iter=300, random_state=RANDOM_STATE,
    )
    proxy_labels = proxy.fit_predict(X_scaled)
    ts(f"  Proxy clustering done ({time.time()-t0:.1f}s) "
       f"clusters_found={np.unique(proxy_labels).size}")

    # Dataset is small (~46K rows) — MI runs on the full set, no subsampling.
    mi_scores = mutual_info_regression(
        X_scaled, proxy_labels.astype(np.float32),
        n_neighbors=MI_NEIGHBORS, random_state=RANDOM_STATE,
    )
    ts(f"  MI estimation done ({time.time()-t0:.1f}s)")

    ranks_array = np.arange(1, n_features + 1, dtype=np.float64)

    def finite_normaliser(s):
        return float(np.sum(1.0 / ranks_array ** s))

    norm = finite_normaliser(S_DECAY)
    seed_df = pd.DataFrame({"feature": FEATURE_COLS, "mi_score": mi_scores.astype(float)})
    con = duckdb.connect()
    con.register("mi_table", seed_df)
    # ROW_NUMBER() (not RANK()) with a deterministic tiebreak: RANK() gives
    # tied MI scores the same rank and skips the next value, breaking the
    # finite normaliser's assumption of a clean 1..d rank sequence (see the
    # same fix in Step_3_Zeta_Weighting.py, discovered via the raw-corpus
    # weight-sum assertion failing on two exactly-tied zero-MI features).
    feature_rank_df = con.execute(f"""
        SELECT
            feature, mi_score,
            ROW_NUMBER() OVER (ORDER BY mi_score DESC, feature ASC) AS rank,
            (1.0 / POWER(ROW_NUMBER() OVER (ORDER BY mi_score DESC, feature ASC), {S_DECAY})) / {norm}
                AS zeta_weight
        FROM mi_table
        ORDER BY rank ASC
    """).df()
    con.close()
    weight_sum = feature_rank_df["zeta_weight"].sum()
    assert abs(weight_sum - 1.0) < 1e-4, f"Weight sum {weight_sum} != 1.0"
    feature_rank_df.to_parquet(ranks_path, index=False)
    ts(f"  Zeta ranks computed, weight_sum={weight_sum:.6f}  ({time.time()-t0:.1f}s)")

    weight_map = dict(zip(feature_rank_df["feature"], feature_rank_df["zeta_weight"]))
    weight_vector = np.array([weight_map[f] for f in FEATURE_COLS], dtype=np.float32)
    X_weighted = (X_scaled * weight_vector).astype(np.float32)
    np.save(weighted_path, X_weighted)
    ts(f"  X_elliptic_weighted.npy written  ({time.time()-t0:.1f}s)")
    mark_done("zeta")

ts("\nTop 15 Elliptic feature ranks (by MI against proxy clustering):")
ts("\n" + feature_rank_df.sort_values("rank").head(15).to_string(index=False))

# ============================================================
# STAGE 4 — (weights already applied in STAGE 3 for this smaller dataset;
#   kept as its own stage number for naming parity with the main pipeline)
# ============================================================
ts("STAGE 4 [Apply Weights]: X_elliptic_weighted.npy ready "
   f"(shape={X_weighted.shape}).")

# ============================================================
# STAGE 5 — Clustering without rule-based seeding, sweep K
# ============================================================
rng = np.random.default_rng(RANDOM_STATE)
val_size = max(500, int(n_samples * VAL_FRACTION))
perm = rng.permutation(n_samples)
val_idx = perm[:val_size]
train_idx = perm[val_size:]
X_train, X_val = X_weighted[train_idx], X_weighted[val_idx]
y_train_illicit, y_val_illicit = illicit_labels[train_idx], illicit_labels[val_idx]

# No StandardScaler here: X_weighted is already RobustScaler'd features times a
# per-column zeta-weight constant, and re-standardizing per column afterward is
# a mathematical no-op that would erase the weighting entirely (same issue
# fixed in Step_5_ZSH_Clustering.py / Step_7_Statistical_Rigor.py).
X_train_norm = X_train.astype(np.float32)
X_val_norm = X_val.astype(np.float32)


def build_ward_guided_centers(X_fit, k, n_micro=WARD_MICRO_CLUSTERS):
    n_micro = max(k, min(n_micro, len(X_fit)))
    micro = MiniBatchKMeans(n_clusters=n_micro, init="k-means++", n_init=5,
                             batch_size=5_000, max_iter=200, random_state=RANDOM_STATE)
    micro.fit(X_fit)
    ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
        micro.cluster_centers_
    )
    return np.vstack(
        [micro.cluster_centers_[ward_labels == cl].mean(axis=0) for cl in range(k)]
    ).astype(np.float32)


if is_done("cluster"):
    ts("STAGE 5 [Cluster]: Loading cached K-sweep results ...")
    k_sweep_df = pd.read_csv(p("step9_k_sweep.csv"))
    best_k = int(k_sweep_df.sort_values("val_silhouette", ascending=False).iloc[0]["k"])
    elliptic_labels = np.load(p("elliptic_cluster_labels.npy"))
else:
    ts("STAGE 5 [Cluster]: Sweeping K ∈ " + str(K_SWEEP) +
       " with KMeans++ Elkan and Ward-guided init (no semantic seeding) ...")
    t0 = time.time()
    k_sweep_rows = []
    fitted_models = {}
    for k in K_SWEEP:
        km_elkan = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=250,
                           algorithm="elkan", random_state=RANDOM_STATE)
        km_elkan.fit(X_train_norm)
        elkan_val_labels = km_elkan.predict(X_val_norm)
        elkan_sil = silhouette_score(X_val_norm, elkan_val_labels, random_state=RANDOM_STATE) \
            if len(np.unique(elkan_val_labels)) > 1 else float("nan")

        ward_centers = build_ward_guided_centers(X_train_norm, k)
        km_ward = KMeans(n_clusters=k, init=ward_centers, n_init=1, max_iter=200,
                          algorithm="elkan", random_state=RANDOM_STATE)
        km_ward.fit(X_train_norm)
        ward_val_labels = km_ward.predict(X_val_norm)
        ward_sil = silhouette_score(X_val_norm, ward_val_labels, random_state=RANDOM_STATE) \
            if len(np.unique(ward_val_labels)) > 1 else float("nan")

        for variant, model, sil in [("kmeans_elkan", km_elkan, elkan_sil),
                                     ("ward_guided", km_ward, ward_sil)]:
            k_sweep_rows.append({"k": k, "variant": variant, "val_silhouette": sil})
            fitted_models[(k, variant)] = model
        ts(f"  k={k:2d}  KMeans++Elkan Sil={elkan_sil:.4f}  "
           f"Ward-guided Sil={ward_sil:.4f}  ({time.time()-t0:.1f}s elapsed)")

    k_sweep_df = pd.DataFrame(k_sweep_rows)
    k_sweep_df.to_csv(p("step9_k_sweep.csv"), index=False)

    best_row = k_sweep_df.sort_values("val_silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    best_variant = best_row["variant"]
    ts(f"  Best config: k={best_k}  variant={best_variant}  "
       f"Sil={best_row['val_silhouette']:.4f}")

    best_model = fitted_models[(best_k, best_variant)]
    X_full_norm = X_weighted.astype(np.float32)
    elliptic_labels = best_model.predict(X_full_norm).astype(np.int32)

    # Balance refinement (same fix as Step_5_ZSH_Clustering.py): the initial
    # K-sweep favored k=10 with 97.6% of rows in one cluster. Recursively
    # split any cluster over BALANCE_MAX_SHARE of the corpus.
    elliptic_labels, elliptic_splits_log = apply_balance_refinement(
        elliptic_labels, X_full_norm
    )
    if elliptic_splits_log:
        pd.DataFrame(elliptic_splits_log).to_csv(p("elliptic_balance_refinement_log.csv"), index=False)
        ts(f"  Balance refinement: {len(elliptic_splits_log)} split(s) -> "
           f"n_clusters={len(np.unique(elliptic_labels))}")

    np.save(p("elliptic_cluster_labels.npy"), elliptic_labels)
    ts(f"  Full-corpus cluster labels saved. n_clusters={len(np.unique(elliptic_labels))}  "
       f"({time.time()-t0:.1f}s)")
    mark_done("cluster")

# ============================================================
# STAGE 6 — External validation against Elliptic's real illicit/licit labels
# ============================================================
ts("\n" + "=" * 70)
ts("STAGE 6 — External validation (Elliptic ground truth)")
ts("=" * 70)


def semantic_metrics(cluster_labels, semantic_labels):
    """Verbatim logic from Step_8_Contextual_Profiling_Comparison.py's
    semantic_metrics(), generalized to any categorical semantic_labels
    (here: binary illicit/licit instead of internal rule labels)."""
    n_total = len(cluster_labels)
    weighted_purity = 0.0
    macro_purity = 0.0
    weighted_entropy = 0.0
    high_purity_clusters = 0
    unique_clusters = np.unique(cluster_labels)
    for cl in unique_clusters:
        mask = cluster_labels == cl
        labels_cl = semantic_labels[mask]
        counts = Counter(labels_cl.tolist())
        top_label, top_count = counts.most_common(1)[0]
        purity = top_count / len(labels_cl)
        probs = np.array(list(counts.values()), dtype=np.float64) / len(labels_cl)
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        weighted_purity += purity * len(labels_cl) / n_total
        macro_purity += purity
        weighted_entropy += entropy * len(labels_cl) / n_total
        if purity >= 0.80:
            high_purity_clusters += 1
    macro_purity /= len(unique_clusters)
    return {
        "weighted_purity": float(weighted_purity),
        "macro_purity": float(macro_purity),
        "weighted_entropy": float(weighted_entropy),
        "nmi_illicit": float(normalized_mutual_info_score(semantic_labels, cluster_labels)),
        "ami_illicit": float(adjusted_mutual_info_score(semantic_labels, cluster_labels)),
        "high_purity_clusters": float(high_purity_clusters),
        "n_clusters": float(len(unique_clusters)),
    }


metrics = semantic_metrics(elliptic_labels, illicit_labels)
ts("Cluster-vs-illicit-label metrics:")
for k_m, v_m in metrics.items():
    ts(f"  {k_m:<20} {v_m:.4f}")

# Per-cluster majority-vote precision/recall for the illicit class:
# treat "cluster majority = illicit" as a per-cluster prediction.
cluster_rows = []
for cl in np.unique(elliptic_labels):
    mask = elliptic_labels == cl
    n_cl = int(mask.sum())
    n_illicit_cl = int(illicit_labels[mask].sum())
    majority_illicit = n_illicit_cl / n_cl >= 0.5
    cluster_rows.append({
        "cluster": int(cl),
        "n": n_cl,
        "n_illicit": n_illicit_cl,
        "pct_illicit": n_illicit_cl / n_cl * 100,
        "majority_vote": "illicit" if majority_illicit else "licit",
    })
cluster_df = pd.DataFrame(cluster_rows).sort_values("pct_illicit", ascending=False)
cluster_df.to_csv(p("step9_cluster_illicit_breakdown.csv"), index=False)

majority_pred = np.array([
    1 if cluster_df.set_index("cluster").loc[cl, "majority_vote"] == "illicit" else 0
    for cl in elliptic_labels
])
tp = int(((majority_pred == 1) & (illicit_labels == 1)).sum())
fp = int(((majority_pred == 1) & (illicit_labels == 0)).sum())
fn = int(((majority_pred == 0) & (illicit_labels == 1)).sum())
precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
ts(f"\nPer-cluster majority-vote illicit detection: "
   f"precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}")
if tp + fp == 0:
    ts("  NOTE: no cluster reached a >=50% illicit majority — expected under "
     f"this corpus's {illicit_labels.mean()*100:.1f}% base illicit rate. "
     "See best-cluster enrichment/lift below for a more informative signal.")

# Best-cluster illicit enrichment/lift — more informative than majority-vote
# under heavy class imbalance (illicit is a small minority overall, so no
# cluster is expected to reach a literal majority).
base_rate = float(illicit_labels.mean())
best_cluster_row = cluster_df.iloc[0]  # already sorted by pct_illicit desc
best_cluster_pct = float(best_cluster_row["pct_illicit"]) / 100.0
best_cluster_lift = (best_cluster_pct / base_rate) if base_rate > 0 else float("nan")
best_cluster_illicit_coverage = float(best_cluster_row["n_illicit"]) / int(illicit_labels.sum())
ts(f"  Best-cluster enrichment: cluster {int(best_cluster_row['cluster'])} holds "
   f"{best_cluster_illicit_coverage*100:.1f}% of all illicit transactions at "
   f"{best_cluster_pct*100:.2f}% concentration ({best_cluster_lift:.2f}x base rate "
   f"of {base_rate*100:.2f}%)")

summary_path = p("step9_elliptic_validation_summary.csv")
pd.DataFrame([{
    "best_k": best_k,
    **metrics,
    "majority_vote_precision": precision,
    "majority_vote_recall": recall,
    "majority_vote_f1": f1,
    "base_illicit_rate": base_rate,
    "best_cluster_illicit_pct": best_cluster_pct,
    "best_cluster_lift_vs_base": best_cluster_lift,
    "best_cluster_illicit_coverage": best_cluster_illicit_coverage,
    "n_rows": n_samples,
    "n_illicit": int(illicit_labels.sum()),
    "n_licit": int((1 - illicit_labels).sum()),
}]).to_csv(summary_path, index=False)
ts(f"Summary -> {summary_path}")

# ============================================================
# STAGE 7 — Report
# ============================================================
report_path = p("step9_elliptic_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("STEP 9 — ELLIPTIC REPLICATION STUDY (reviewer #6, external validation)\n")
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. WHAT THIS VALIDATES\n")
    f.write(
        "This is an INDEPENDENT REPLICATION study, not a crosswalk of the custom\n"
        "dataset's existing clusters. Elliptic's schema (165 anonymized PCA\n"
        "features, no raw fee/value/address-count fields) is incompatible with the\n"
        "custom dataset's ~27 raw-feature space, so cluster labels cannot be mapped\n"
        "directly between the two corpora. Instead, the ZSH methodology (proxy-\n"
        "clustering + MI-based Zeta feature weighting, followed by seeding-agnostic\n"
        "KMeans++/Ward-guided clustering) was refit natively on Elliptic and\n"
        "evaluated against Elliptic's own real illicit/licit ground-truth labels.\n"
        "This tests whether the METHOD generalizes to externally-labeled data, not\n"
        "whether the two datasets cross-reference.\n\n"
    )

    f.write("2. SETUP\n")
    f.write(f"   Elliptic labeled rows used : {n_samples:,} "
            f"(dropped {203769 - n_samples:,} 'unknown'-class rows)\n")
    f.write(f"   Illicit / Licit            : {int(illicit_labels.sum()):,} / "
            f"{int((1-illicit_labels).sum()):,}\n")
    f.write(f"   Feature space               : 165 anonymized Elliptic features, "
            f"RobustScaled, Zeta-weighted (s={S_DECAY})\n")
    f.write(f"   Semantic seeding            : NONE — rule-based PRIORITY_RULES "
            f"seeding (Step 5) cannot transfer to Elliptic's anonymized schema\n")
    f.write(f"   K sweep                     : {K_SWEEP}  (K is NOT assumed to "
            f"transfer from the custom-dataset choice of K=30)\n\n")

    f.write("3. CLUSTER-VS-ILLICIT-LABEL METRICS\n")
    for k_m, v_m in metrics.items():
        f.write(f"   {k_m:<24} {v_m:.4f}\n")
    f.write(f"   majority_vote_precision  {precision:.4f}\n")
    f.write(f"   majority_vote_recall     {recall:.4f}\n")
    f.write(f"   majority_vote_f1         {f1:.4f}\n")
    if tp + fp == 0:
        f.write(
            f"   NOTE: no cluster reached a >=50% illicit majority — expected\n"
            f"   given illicit is only {base_rate*100:.1f}% of labeled rows overall.\n"
            f"   Best-cluster enrichment (below) is the more informative signal\n"
            f"   under this class imbalance.\n"
        )
    f.write(
        f"   best_cluster_illicit_pct     {best_cluster_pct*100:.2f}%  "
        f"({best_cluster_lift:.2f}x base rate {base_rate*100:.2f}%)\n"
        f"   best_cluster_illicit_coverage {best_cluster_illicit_coverage*100:.1f}% "
        f"of all illicit transactions\n\n"
    )

    f.write("4. PER-CLUSTER ILLICIT BREAKDOWN (top 10 by % illicit)\n")
    f.write(cluster_df.head(10).to_string(index=False) + "\n\n")

    f.write("5. WHAT THIS DOES AND DOES NOT ANSWER\n")
    f.write(
        "   ANSWERS: whether ZSH-style clustering, refit on an independent\n"
        "   externally-labeled Bitcoin dataset, recovers a coherent illicit-\n"
        "   enriched cluster structure (illicit-vs-licit separability).\n"
        "   DOES NOT ANSWER: the reviewer's specific request to validate against\n"
        "   'known coinjoin/mixing transactions' — Elliptic's public release only\n"
        "   provides binary illicit/licit labels, with no coinjoin/mixing sub-\n"
        "   category. That specific claim remains untestable without additional,\n"
        "   non-public Elliptic metadata or a different labeled corpus.\n\n"
    )

    f.write("6. SAFE MANUSCRIPT CLAIM\n")
    if tp + fp > 0:
        f.write(
            "   \"On an independent, externally-labeled Bitcoin transaction dataset\n"
            f"   (Elliptic, n={n_samples:,}), the ZSH feature-weighting and clustering\n"
            f"   methodology (re-fit natively, without rule-based semantic seeding)\n"
            f"   achieved a per-cluster majority-vote illicit-detection precision of\n"
            f"   {precision:.3f} and recall of {recall:.3f} "
            f"(NMI={metrics['nmi_illicit']:.3f}\n"
            f"   against ground-truth illicit/licit labels). This provides evidence the\n"
            f"   method generalizes beyond the custom corpus, though Elliptic's binary\n"
            f"   labels cannot confirm the coinjoin/mixing-specific claims made about\n"
            f"   the custom dataset.\"\n\n"
        )
    else:
        f.write(
            "   \"On an independent, externally-labeled Bitcoin transaction dataset\n"
            f"   (Elliptic, n={n_samples:,}), the ZSH feature-weighting and clustering\n"
            f"   methodology (re-fit natively, without rule-based semantic seeding) did\n"
            f"   not produce any cluster with a literal majority of illicit transactions\n"
            f"   (expected: illicit is only {base_rate*100:.1f}% of labeled rows). "
            f"Instead,\n"
            f"   {best_cluster_illicit_coverage*100:.0f}% of all illicit transactions "
            f"concentrated into a single\n"
            f"   cluster at {best_cluster_pct*100:.2f}% illicit density "
            f"({best_cluster_lift:.2f}x the base rate), with\n"
            f"   NMI={metrics['nmi_illicit']:.3f} / AMI={metrics['ami_illicit']:.3f} "
            f"against ground-truth labels. This is\n"
            f"   weak-to-modest evidence the method generalizes beyond the custom\n"
            f"   corpus; it does not support a strong illicit-detection claim, and\n"
            f"   Elliptic's binary labels cannot confirm the coinjoin/mixing-specific\n"
            f"   claims made about the custom dataset.\"\n\n"
        )

    f.write("7. UNSAFE / OVERREACHING CLAIMS TO AVOID\n")
    f.write(
        "   - Do NOT claim this validates the custom dataset's actual clusters\n"
        "     (no ID crosswalk exists — see STAGE 0 / step9_txid_overlap.txt).\n"
        "   - Do NOT claim coinjoin/mixing validation — only illicit/licit is\n"
        "     testable with Elliptic's public labels.\n\n"
    )

    f.write("REVIEWER NOTE\n")
    f.write(
        "   Because Elliptic's schema differs entirely from the custom dataset's\n"
        "   feature space, this script re-derives Zeta weighting and clustering\n"
        "   from scratch on Elliptic — it is evidence of METHOD generalization,\n"
        "   presented as an independent replication study rather than a direct\n"
        "   external validation of the custom dataset's specific cluster outputs.\n"
    )

ts(f"\nReport written -> {report_path}")
ts("\n" + "=" * 70)
ts(f"STEP 9 COMPLETE  (total {time.time()-_T0:.1f}s)")
ts("=" * 70)
