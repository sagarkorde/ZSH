import io
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# STEP 5 — ZSH Clustering: tuned for the corrected protocol
#
# What changed:
#   1. Uses the same weighted-space geometry that Step 7 now evaluates:
#      X_w_norm = StandardScaler(X_weighted)
#   2. Adds a geometry-first branch from Step 3:
#      next-generation ZSH-G metric learning on top of the corrected Zeta space
#   3. Builds semantic seed centers from rule-derived transaction families
#   4. Adds Agglomerative-inspired Ward micro-cluster initialisation to target
#      the compactness / separation gap seen against pure Agglomerative
#   5. Blends semantic seeds with Ward structure to create a silhouette-seeking
#      hybrid candidate without losing interpretability
#   6. Chooses between multiple candidates on a held-out validation subset,
#      using KMeans++ Elkan as the honest validation reference
#   7. Keeps semantic interpretability separate from seed construction:
#      seed labels optimize initialization; semantic labels optimize naming
#
# Corrected baseline:
#   The final comparison still reports intrinsic metrics in X_w_norm,
#   but geometry-first candidates are allowed to train in a preconditioned
#   linear transform of that space and are then judged back in X_w_norm.
# ============================================================

# Fix Unicode for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
OUTPUT_DIR = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_CLUSTERS = 30
FIT_SAMPLE = 500_000
VALIDATION_SAMPLE = 50_000
EVAL_SAMPLE = 60_000
SEED_SUBSAMPLE = 100_000
SEED_MIN_POINTS = 500
REFINE_MAX_ITER = 120
WARD_MICRO_CLUSTERS = 160
SEED_WARD_BLEND_ALPHA = 0.60
GEOM_WARD_BLEND_ALPHA = 0.50
ELKAN_N_INIT = 20
ELKAN_MAX_ITER = 250
ANOMALY_CONT = 0.05
RANDOM_STATE = 42

SEED_LABEL_MODE = "overwrite_last"
SEMANTIC_LABEL_MODE = "first_match"

PRIORITY_RULES = [
    ("has_coinbase", "Coinbase"),
    ("is_coinjoin_like", "Coinjoin_Mixer"),
    ("is_batch_payment", "Batch_Payment"),
    ("is_consolidation", "Consolidation"),
    ("is_distribution", "Distribution"),
    ("is_peer_to_peer", "Standard_P2P"),
    ("has_op_return", "OP_Return"),
    ("rbf_enabled", "RBF_Enabled"),
]

CANDIDATE_DESCRIPTIONS = {
    "seeded_minibatch": "Rule-derived semantic seeds only",
    "seeded_refined": "Rule-derived seeds + full centroid refinement",
    "ward_guided_refined": "Agglomerative-inspired Ward init + refinement",
    "seed_ward_blend": "Semantic seeds blended with Ward structure",
    "geom_elkan": "ZSH-G Elkan in learned metric space",
    "geom_ward_refined": "ZSH-G Ward-guided Elkan refinement",
    "geom_elkan_ward_blend": "ZSH-G Elkan/Ward blended refinement",
}

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
log_path = os.path.join(OUTPUT_DIR, "zsh_improved_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
_T0 = time.time()
rng = np.random.default_rng(RANDOM_STATE)


def ts(msg: str) -> None:
    log.info(msg)


def predict_chunked(model, X, chunk_size=500_000):
    labels = np.zeros(len(X), dtype=np.int32)
    for i in range(0, len(X), chunk_size):
        end = min(i + chunk_size, len(X))
        labels[i:end] = model.predict(X[i:end])
        if end % 2_000_000 == 0 or end == len(X):
            ts(f"    predict: {end:,}/{len(X):,}")
    return labels


def build_rule_labels(frame: pd.DataFrame, mode: str) -> np.ndarray:
    labels = np.full(len(frame), "Unknown", dtype=object)

    if mode == "overwrite_last":
        for col, label in PRIORITY_RULES:
            if col in frame.columns:
                labels[frame[col].to_numpy(dtype=float) > 0] = label
        return labels

    if mode == "first_match":
        unassigned = np.ones(len(frame), dtype=bool)
        for col, label in PRIORITY_RULES:
            if col not in frame.columns:
                continue
            mask = (frame[col].to_numpy(dtype=float) > 0) & unassigned
            labels[mask] = label
            unassigned[mask] = False
        return labels

    raise ValueError(f"Unsupported rule-label mode: {mode}")


def log_rule_distribution(name: str, labels: np.ndarray, total: int) -> None:
    ts(f"\n{name}:")
    cats, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    for cat, cnt in zip(cats[order], counts[order]):
        ts(f"  {cat:<25}  {int(cnt):>10,}  ({cnt / total * 100:.2f}%)")


def allocate_seed_counts(rule_labels: np.ndarray) -> dict[str, int]:
    cats, counts = np.unique(rule_labels, return_counts=True)
    seed_cats = {
        cat: int(cnt)
        for cat, cnt in zip(cats, counts)
        if cat != "Unknown" and int(cnt) >= SEED_MIN_POINTS
    }
    if not seed_cats:
        raise RuntimeError("No rule-derived seed categories met the minimum count.")

    alloc = {
        cat: max(1, round(K_CLUSTERS * cnt / sum(seed_cats.values())))
        for cat, cnt in seed_cats.items()
    }

    while sum(alloc.values()) > K_CLUSTERS:
        victim = max(alloc, key=lambda c: (alloc[c], -seed_cats[c]))
        if alloc[victim] > 1:
            alloc[victim] -= 1
        else:
            break

    while sum(alloc.values()) < K_CLUSTERS:
        donor = max(alloc, key=lambda c: seed_cats[c])
        alloc[donor] += 1

    return alloc


def build_seed_centers(X_ref, rule_labels_ref: np.ndarray, tag: str) -> np.ndarray:
    alloc = allocate_seed_counts(rule_labels_ref)

    ts(f"\n{tag}:")
    for cat, k_sub in sorted(alloc.items(), key=lambda kv: (-kv[1], kv[0])):
        ts(f"  {cat:<25} -> {k_sub:2d} seed(s)")

    centers = []
    for cat, k_sub in alloc.items():
        cat_idx = np.where(rule_labels_ref == cat)[0]
        if len(cat_idx) > SEED_SUBSAMPLE:
            cat_idx = rng.choice(cat_idx, size=SEED_SUBSAMPLE, replace=False)
        X_cat = X_ref[cat_idx]

        if k_sub == 1:
            centers.append(X_cat.mean(axis=0))
            continue

        km_sub = MiniBatchKMeans(
            n_clusters=k_sub,
            init="k-means++",
            n_init=5,
            batch_size=20_000,
            max_iter=150,
            random_state=RANDOM_STATE,
        )
        km_sub.fit(X_cat)
        centers.extend(km_sub.cluster_centers_)

    centers = np.asarray(centers, dtype=np.float32)
    if len(centers) != K_CLUSTERS:
        raise RuntimeError(
            f"Seed count mismatch: expected {K_CLUSTERS}, got {len(centers)}"
        )
    return centers


def score_labels(X_eval, labels: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": float(silhouette_score(X_eval, labels, random_state=RANDOM_STATE)),
        "dbi": float(davies_bouldin_score(X_eval, labels)),
        "chi": float(calinski_harabasz_score(X_eval, labels)),
    }


def select_candidate(candidates: list[dict], baseline_metrics: dict[str, float]) -> dict:
    sil_order = sorted(candidates, key=lambda r: r["silhouette"], reverse=True)
    dbi_order = sorted(candidates, key=lambda r: r["dbi"])
    chi_order = sorted(candidates, key=lambda r: r["chi"], reverse=True)

    sil_rank = {row["name"]: rank for rank, row in enumerate(sil_order, start=1)}
    dbi_rank = {row["name"]: rank for rank, row in enumerate(dbi_order, start=1)}
    chi_rank = {row["name"]: rank for rank, row in enumerate(chi_order, start=1)}

    for row in candidates:
        row["rank_sum"] = (
            sil_rank[row["name"]] + dbi_rank[row["name"]] + chi_rank[row["name"]]
        )
        wins = (
            int(row["silhouette"] > baseline_metrics["silhouette"])
            + int(row["dbi"] < baseline_metrics["dbi"])
            + int(row["chi"] > baseline_metrics["chi"])
        )
        row["wins_vs_val_baseline"] = wins
        row["beats_val_baseline_all3"] = wins == 3

    candidates = sorted(
        candidates,
        key=lambda r: (
            -int(r["beats_val_baseline_all3"]),
            -r["wins_vs_val_baseline"],
            r["rank_sum"],
            -r["silhouette"],
            r["dbi"],
            -r["chi"],
        ),
    )
    return candidates[0]


def best_by_metric(candidates: list[dict], metric: str) -> dict:
    if metric == "silhouette":
        key = lambda r: (r["silhouette"], -r["dbi"], r["chi"])
        return max(candidates, key=key)
    if metric == "dbi":
        key = lambda r: (-r["dbi"], r["silhouette"], r["chi"])
        return min(candidates, key=lambda r: (r["dbi"], -r["silhouette"], -r["chi"]))
    if metric == "chi":
        key = lambda r: (r["chi"], r["silhouette"], -r["dbi"])
        return max(candidates, key=key)
    raise ValueError(f"Unsupported metric: {metric}")


def fit_seeded_minibatch(X_fit, seed_centers: np.ndarray):
    model = MiniBatchKMeans(
        n_clusters=K_CLUSTERS,
        init=seed_centers,
        n_init=1,
        batch_size=50_000,
        max_iter=400,
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit)
    return model


def build_ward_guided_centers(
    X_fit: np.ndarray, tag: str, n_micro_clusters: int = WARD_MICRO_CLUSTERS
) -> np.ndarray:
    n_micro = max(K_CLUSTERS, min(int(n_micro_clusters), len(X_fit)))

    ts(f"\n{tag}:")
    ts(f"  Ward micro-clusters: {n_micro}")

    micro = MiniBatchKMeans(
        n_clusters=n_micro,
        init="k-means++",
        n_init=5,
        batch_size=50_000,
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    micro.fit(X_fit)

    ward_labels = AgglomerativeClustering(
        n_clusters=K_CLUSTERS, linkage="ward"
    ).fit_predict(micro.cluster_centers_)

    centers = np.vstack(
        [
            micro.cluster_centers_[ward_labels == cl].mean(axis=0)
            for cl in range(K_CLUSTERS)
        ]
    ).astype(np.float32)
    return centers


def blend_center_sets(
    seed_centers: np.ndarray,
    ward_centers: np.ndarray,
    alpha: float = SEED_WARD_BLEND_ALPHA,
) -> np.ndarray:
    if seed_centers.shape != ward_centers.shape:
        raise ValueError(
            f"Center shape mismatch: {seed_centers.shape} vs {ward_centers.shape}"
        )

    cost = np.sum(
        (seed_centers[:, None, :] - ward_centers[None, :, :]) ** 2, axis=2
    )
    seed_idx, ward_idx = linear_sum_assignment(cost)
    aligned_ward = np.empty_like(ward_centers)
    aligned_ward[seed_idx] = ward_centers[ward_idx]
    blended = alpha * seed_centers + (1.0 - alpha) * aligned_ward
    return blended.astype(np.float32)


def fit_refined_kmeans(X_fit, init_centers: np.ndarray):
    model = KMeans(
        n_clusters=K_CLUSTERS,
        init=init_centers,
        n_init=1,
        max_iter=REFINE_MAX_ITER,
        algorithm="elkan",
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit)
    return model


def fit_kmeans_elkan(X_fit):
    model = KMeans(
        n_clusters=K_CLUSTERS,
        init="k-means++",
        n_init=ELKAN_N_INIT,
        max_iter=ELKAN_MAX_ITER,
        algorithm="elkan",
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit)
    return model


def predict_and_score(model, X_pred, X_eval):
    labels = model.predict(X_pred)
    return labels, score_labels(X_eval, labels)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
ts("=" * 70)
ts("ZSH CLUSTERING — tuned semantic seeding + centroid refinement")
ts("=" * 70)

X_w = np.load(os.path.join(OUTPUT_DIR, "X_weighted.npy")).astype(np.float32)
df_feat = pd.read_parquet(os.path.join(OUTPUT_DIR, "df_balanced_features.parquet"))
X_geom_pre = None
geom_feature_space = None
for candidate_file, space_name in [
    ("X_zshg_metric.npy", "X_zshg_metric"),
    ("X_weighted_geometry_whitened.npy", "X_weighted_geometry_whitened"),
]:
    geom_path = os.path.join(OUTPUT_DIR, candidate_file)
    if os.path.exists(geom_path):
        X_geom_pre = np.load(geom_path).astype(np.float32)
        geom_feature_space = space_name
        break

n_samples, n_features = X_w.shape
n_samples = min(n_samples, len(df_feat))
if X_geom_pre is not None:
    n_samples = min(n_samples, len(X_geom_pre))
X_w = X_w[:n_samples]
df_feat = df_feat.iloc[:n_samples].reset_index(drop=True)
if X_geom_pre is not None:
    X_geom_pre = X_geom_pre[:n_samples]

ts(f"X_weighted  : {n_samples:,} x {n_features}")
ts(f"df_feat     : {df_feat.shape}")
if X_geom_pre is not None:
    ts(f"X_geom_pre  : {X_geom_pre.shape}  ({geom_feature_space})")
else:
    ts("X_geom_pre  : not found — geometry-first candidates disabled")

ts("\nNormalizing X_weighted into X_w_norm ...")
scaler_w = StandardScaler()
X_w_norm = scaler_w.fit_transform(X_w).astype(np.float32)
del X_w
ts(f"  X_w_norm  : {X_w_norm.shape}  dtype={X_w_norm.dtype}")

seed_rule_labels_full = build_rule_labels(df_feat, SEED_LABEL_MODE)
semantic_rule_labels_full = build_rule_labels(df_feat, SEMANTIC_LABEL_MODE)
log_rule_distribution("Seed-label distribution (for initialization)", seed_rule_labels_full, n_samples)
log_rule_distribution("Semantic-label distribution (for naming)", semantic_rule_labels_full, n_samples)

# ------------------------------------------------------------
# Shared fit / validation split
# ------------------------------------------------------------
fit_size = min(FIT_SAMPLE, n_samples)
fit_idx = rng.choice(n_samples, size=fit_size, replace=False)
rng.shuffle(fit_idx)
val_size = min(VALIDATION_SAMPLE, max(K_CLUSTERS * 100, fit_size // 10))
val_idx = fit_idx[:val_size]
train_idx = fit_idx[val_size:]

X_train = X_w_norm[train_idx]
X_val = X_w_norm[val_idx]
X_geom_train = X_geom_pre[train_idx] if X_geom_pre is not None else None
X_geom_val = X_geom_pre[val_idx] if X_geom_pre is not None else None
seed_rule_labels_train = seed_rule_labels_full[train_idx]

ts("\nShared fit / validation split:")
ts(f"  fit subset       : {fit_size:,}")
ts(f"  train subset     : {len(train_idx):,}")
ts(f"  validation subset: {len(val_idx):,}")

# ------------------------------------------------------------
# Candidate search in corrected weighted space
# ------------------------------------------------------------
ts("\n" + "=" * 70)
ts("STEP 1: Candidate search in X_w_norm")
ts("=" * 70)

t0 = time.time()
baseline_val_model = fit_kmeans_elkan(X_train)
baseline_val_labels = baseline_val_model.predict(X_val)
baseline_val_metrics = score_labels(X_val, baseline_val_labels)
ts(
    "  Validation reference (KMeans++ Elkan, scored in X_w_norm): "
    f"Sil={baseline_val_metrics['silhouette']:.4f}  "
    f"DBI={baseline_val_metrics['dbi']:.4f}  "
    f"CHI={baseline_val_metrics['chi']:.1f}"
)

seed_centers_train = build_seed_centers(
    X_train, seed_rule_labels_train, "Seed allocation on training subset"
)
ward_centers_train = build_ward_guided_centers(
    X_train, "Ward-guided initialization on training subset"
)
blend_centers_train = blend_center_sets(seed_centers_train, ward_centers_train)

stage1_model = fit_seeded_minibatch(X_train, seed_centers_train)
stage1_val_labels, stage1_metrics = predict_and_score(stage1_model, X_val, X_val)

stage2_model = fit_refined_kmeans(X_train, stage1_model.cluster_centers_)
stage2_val_labels, stage2_metrics = predict_and_score(stage2_model, X_val, X_val)

ward_model = fit_refined_kmeans(X_train, ward_centers_train)
ward_val_labels, ward_metrics = predict_and_score(ward_model, X_val, X_val)

blend_model = fit_seeded_minibatch(X_train, blend_centers_train)
blend_val_labels, blend_metrics = predict_and_score(blend_model, X_val, X_val)

candidate_rows = [
    {
        "name": "seeded_minibatch",
        "description": CANDIDATE_DESCRIPTIONS["seeded_minibatch"],
        "feature_space": "X_w_norm",
        **stage1_metrics,
    },
    {
        "name": "seeded_refined",
        "description": CANDIDATE_DESCRIPTIONS["seeded_refined"],
        "feature_space": "X_w_norm",
        **stage2_metrics,
    },
    {
        "name": "ward_guided_refined",
        "description": CANDIDATE_DESCRIPTIONS["ward_guided_refined"],
        "feature_space": "X_w_norm",
        **ward_metrics,
    },
    {
        "name": "seed_ward_blend",
        "description": CANDIDATE_DESCRIPTIONS["seed_ward_blend"],
        "feature_space": "X_w_norm",
        **blend_metrics,
    },
]

if X_geom_train is not None:
    ts(f"\n  Geometry-first candidates in {geom_feature_space} ...")
    geom_elkan_model = fit_kmeans_elkan(X_geom_train)
    _, geom_elkan_metrics = predict_and_score(geom_elkan_model, X_geom_val, X_val)

    geom_ward_centers_train = build_ward_guided_centers(
        X_geom_train, "Ward-guided initialization on geometry-preconditioned subset"
    )
    geom_ward_model = fit_refined_kmeans(X_geom_train, geom_ward_centers_train)
    _, geom_ward_metrics = predict_and_score(geom_ward_model, X_geom_val, X_val)

    geom_blend_centers_train = blend_center_sets(
        geom_elkan_model.cluster_centers_,
        geom_ward_centers_train,
        alpha=GEOM_WARD_BLEND_ALPHA,
    )
    geom_blend_model = fit_refined_kmeans(X_geom_train, geom_blend_centers_train)
    _, geom_blend_metrics = predict_and_score(geom_blend_model, X_geom_val, X_val)

    candidate_rows.extend([
        {
            "name": "geom_elkan",
            "description": CANDIDATE_DESCRIPTIONS["geom_elkan"],
            "feature_space": geom_feature_space,
            **geom_elkan_metrics,
        },
        {
            "name": "geom_ward_refined",
            "description": CANDIDATE_DESCRIPTIONS["geom_ward_refined"],
            "feature_space": geom_feature_space,
            **geom_ward_metrics,
        },
        {
            "name": "geom_elkan_ward_blend",
            "description": CANDIDATE_DESCRIPTIONS["geom_elkan_ward_blend"],
            "feature_space": geom_feature_space,
            **geom_blend_metrics,
        },
    ])

selected = select_candidate(candidate_rows, baseline_val_metrics)

candidate_df = pd.DataFrame(candidate_rows)
sil_winner = best_by_metric(candidate_rows, "silhouette")
dbi_winner = best_by_metric(candidate_rows, "dbi")
chi_winner = best_by_metric(candidate_rows, "chi")
candidate_df["selected_balanced"] = candidate_df["name"].eq(selected["name"])
candidate_df["best_silhouette"] = candidate_df["name"].eq(sil_winner["name"])
candidate_df["best_dbi"] = candidate_df["name"].eq(dbi_winner["name"])
candidate_df["best_chi"] = candidate_df["name"].eq(chi_winner["name"])
candidate_df.to_csv(
    os.path.join(OUTPUT_DIR, "zsh_candidate_search.csv"), index=False
)

pd.DataFrame(
    [
        {
            "context": "balanced_winner",
            "candidate": selected["name"],
            "description": selected["description"],
            "silhouette": selected["silhouette"],
            "dbi": selected["dbi"],
            "chi": selected["chi"],
        },
        {
            "context": "best_silhouette",
            "candidate": sil_winner["name"],
            "description": sil_winner["description"],
            "silhouette": sil_winner["silhouette"],
            "dbi": sil_winner["dbi"],
            "chi": sil_winner["chi"],
        },
        {
            "context": "best_dbi",
            "candidate": dbi_winner["name"],
            "description": dbi_winner["description"],
            "silhouette": dbi_winner["silhouette"],
            "dbi": dbi_winner["dbi"],
            "chi": dbi_winner["chi"],
        },
        {
            "context": "best_chi",
            "candidate": chi_winner["name"],
            "description": chi_winner["description"],
            "silhouette": chi_winner["silhouette"],
            "dbi": chi_winner["dbi"],
            "chi": chi_winner["chi"],
        },
    ]
).to_csv(os.path.join(OUTPUT_DIR, "zsh_context_winners.csv"), index=False)

ts("\nCandidate validation scores:")
for row in candidate_rows:
    ts(
        f"  {row['name']:<18}  Sil={row['silhouette']:.4f}  "
        f"DBI={row['dbi']:.4f}  CHI={row['chi']:.1f}  "
        f"space={row['feature_space']:<28}  "
        f"wins_vs_baseline={row['wins_vs_val_baseline']}  "
        f"rank_sum={row['rank_sum']}"
    )
ts(f"  Selected candidate: {selected['name']}")
ts(
    f"  Context winners -> silhouette: {sil_winner['name']} | "
    f"DBI: {dbi_winner['name']} | CHI: {chi_winner['name']}"
)
ts(f"  Candidate search elapsed: {time.time() - t0:.1f}s")

# ------------------------------------------------------------
# Final baseline and tuned ZSH fit on the full fit subset
# ------------------------------------------------------------
ts("\n" + "=" * 70)
ts("STEP 2: Final baseline and tuned ZSH fit")
ts("=" * 70)

X_fit_full = X_w_norm[fit_idx]
X_geom_fit_full = X_geom_pre[fit_idx] if X_geom_pre is not None else None
seed_rule_labels_fit = seed_rule_labels_full[fit_idx]

ts("  Fitting streaming MiniBatch baseline in X_w_norm ...")
t0 = time.time()
baseline_km = MiniBatchKMeans(
    n_clusters=K_CLUSTERS,
    init="k-means++",
    n_init=10,
    batch_size=50_000,
    max_iter=300,
    random_state=RANDOM_STATE,
)
baseline_km.fit(X_fit_full)
ts(f"  Streaming baseline fit complete in {time.time() - t0:.1f}s")

ts("  Fitting geometry-only reference (KMeans++ Elkan) in X_w_norm ...")
t0 = time.time()
reference_km = fit_kmeans_elkan(X_fit_full)
ts(f"  Geometry-only reference fit complete in {time.time() - t0:.1f}s")

zsh_predict_matrix = X_w_norm
selected_feature_space = selected.get("feature_space", "X_w_norm")

if selected["name"] == "seeded_minibatch":
    ts("  Building semantic seed centers on the full fit subset ...")
    seed_centers_full = build_seed_centers(
        X_fit_full, seed_rule_labels_fit, "Seed allocation on full fit subset"
    )
    ts("  Final model: seeded MiniBatchKMeans ...")
    t0 = time.time()
    zsh_model = fit_seeded_minibatch(X_fit_full, seed_centers_full)
    ts(f"  Final seeded MiniBatch complete in {time.time() - t0:.1f}s")
elif selected["name"] == "seeded_refined":
    ts("  Building semantic seed centers on the full fit subset ...")
    seed_centers_full = build_seed_centers(
        X_fit_full, seed_rule_labels_fit, "Seed allocation on full fit subset"
    )
    ts("  Final model: seeded MiniBatchKMeans + full refinement ...")
    t0 = time.time()
    zsh_stage1 = fit_seeded_minibatch(X_fit_full, seed_centers_full)
    ts(f"  Stage 2a complete in {time.time() - t0:.1f}s")
    t0 = time.time()
    zsh_model = fit_refined_kmeans(X_fit_full, zsh_stage1.cluster_centers_)
    ts(f"  Stage 2b complete in {time.time() - t0:.1f}s")
elif selected["name"] == "ward_guided_refined":
    ts("  Final model: Ward-guided refinement ...")
    ward_centers_full = build_ward_guided_centers(
        X_fit_full, "Ward-guided initialization on full fit subset"
    )
    t0 = time.time()
    zsh_model = fit_refined_kmeans(X_fit_full, ward_centers_full)
    ts(f"  Ward-guided refinement complete in {time.time() - t0:.1f}s")
elif selected["name"] == "seed_ward_blend":
    ts("  Building semantic seed centers on the full fit subset ...")
    seed_centers_full = build_seed_centers(
        X_fit_full, seed_rule_labels_fit, "Seed allocation on full fit subset"
    )
    ts("  Final model: semantic + Ward blended MiniBatchKMeans ...")
    ward_centers_full = build_ward_guided_centers(
        X_fit_full, "Ward-guided initialization on full fit subset"
    )
    blend_centers_full = blend_center_sets(seed_centers_full, ward_centers_full)
    t0 = time.time()
    zsh_model = fit_seeded_minibatch(X_fit_full, blend_centers_full)
    ts(f"  Semantic + Ward blend complete in {time.time() - t0:.1f}s")
elif selected["name"] == "geom_elkan":
    if X_geom_fit_full is None:
        raise RuntimeError("Geometry-first artifacts missing for geom_elkan candidate.")
    ts(f"  Final model: geometry-first Elkan in {geom_feature_space} ...")
    t0 = time.time()
    zsh_model = fit_kmeans_elkan(X_geom_fit_full)
    zsh_predict_matrix = X_geom_pre
    ts(f"  Geometry-first Elkan complete in {time.time() - t0:.1f}s")
elif selected["name"] == "geom_ward_refined":
    if X_geom_fit_full is None:
        raise RuntimeError("Geometry-first artifacts missing for geom_ward_refined candidate.")
    ts(f"  Final model: geometry-first Ward-guided refinement in {geom_feature_space} ...")
    geom_ward_centers_full = build_ward_guided_centers(
        X_geom_fit_full, "Ward-guided initialization on geometry-preconditioned full fit subset"
    )
    t0 = time.time()
    zsh_model = fit_refined_kmeans(X_geom_fit_full, geom_ward_centers_full)
    zsh_predict_matrix = X_geom_pre
    ts(f"  Geometry-first Ward refinement complete in {time.time() - t0:.1f}s")
elif selected["name"] == "geom_elkan_ward_blend":
    if X_geom_fit_full is None:
        raise RuntimeError("Geometry-first artifacts missing for geom_elkan_ward_blend candidate.")
    ts(f"  Final model: geometry-first Elkan/Ward blended refinement in {geom_feature_space} ...")
    geom_elkan_full = fit_kmeans_elkan(X_geom_fit_full)
    geom_ward_centers_full = build_ward_guided_centers(
        X_geom_fit_full, "Ward-guided initialization on geometry-preconditioned full fit subset"
    )
    geom_blend_centers_full = blend_center_sets(
        geom_elkan_full.cluster_centers_,
        geom_ward_centers_full,
        alpha=GEOM_WARD_BLEND_ALPHA,
    )
    t0 = time.time()
    zsh_model = fit_refined_kmeans(X_geom_fit_full, geom_blend_centers_full)
    zsh_predict_matrix = X_geom_pre
    ts(f"  Geometry-first Elkan/Ward blend complete in {time.time() - t0:.1f}s")
else:
    raise RuntimeError(f"Unsupported selected candidate: {selected['name']}")

# ------------------------------------------------------------
# Full-dataset label assignment
# ------------------------------------------------------------
ts("\n" + "=" * 70)
ts("STEP 3: Predicting full-dataset labels")
ts("=" * 70)

ts(f"  Predicting streaming MiniBatch baseline labels for all {n_samples:,} points ...")
baseline_labels = predict_chunked(baseline_km, X_w_norm)
u_base, c_base = np.unique(baseline_labels, return_counts=True)
ts(
    f"  Streaming baseline clusters: {len(u_base)} | min={c_base.min():,}  "
    f"max={c_base.max():,}  mean={c_base.mean():.0f}"
)

ts(f"  Predicting tuned ZSH labels for all {n_samples:,} points ...")
zsh_labels = predict_chunked(zsh_model, zsh_predict_matrix)
u_zsh, c_zsh = np.unique(zsh_labels, return_counts=True)
ts(
    f"  ZSH clusters: {len(u_zsh)} | min={c_zsh.min():,}  "
    f"max={c_zsh.max():,}  mean={c_zsh.mean():.0f}"
)

# ------------------------------------------------------------
# Semantic post-labeling
# ------------------------------------------------------------
ts("\n" + "=" * 70)
ts("STEP 4: Semantic post-labeling")
ts("=" * 70)

cluster_semantic = {}
label_count = {}

for cl in range(K_CLUSTERS):
    cl_mask = zsh_labels == cl
    if cl_mask.sum() == 0:
        cluster_semantic[cl] = f"Empty_{cl}"
        continue

    cl_rules = semantic_rule_labels_full[cl_mask]
    cats, cnts = np.unique(cl_rules, return_counts=True)
    named = [(c, n) for c, n in zip(cats, cnts) if c != "Unknown"]
    best_cat = max(named, key=lambda x: x[1])[0] if named else "Unknown"

    prev = label_count.get(best_cat, 0)
    label_count[best_cat] = prev + 1
    cluster_semantic[cl] = best_cat if prev == 0 else f"{best_cat}_{prev}"

zsh_profiles = np.array([cluster_semantic[l] for l in zsh_labels], dtype=object)

ts("  Cluster -> semantic label:")
for cl in range(K_CLUSTERS):
    cl_n = int((zsh_labels == cl).sum())
    ts(f"    {cl:2d} -> {cluster_semantic[cl]:<24} n={cl_n:>9,}")

# ------------------------------------------------------------
# Anomaly detection
# ------------------------------------------------------------
ts("\n" + "=" * 70)
ts("STEP 5: Anomaly detection")
ts("=" * 70)

if_samp = min(200_000, n_samples)
if_idx = rng.choice(n_samples, size=if_samp, replace=False)
X_if = X_w_norm[if_idx]

ts(f"  Fitting Isolation Forest on {if_samp:,} points ...")
t0 = time.time()
iso = IsolationForest(
    n_estimators=200,
    contamination=ANOMALY_CONT,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso.fit(X_if)
ts(f"  Isolation Forest fit complete in {time.time() - t0:.1f}s")

ts(f"  Scoring all {n_samples:,} points ...")
anomaly_scores = np.zeros(n_samples, dtype=np.float32)
for i in range(0, n_samples, 500_000):
    end = min(i + 500_000, n_samples)
    anomaly_scores[i:end] = -iso.score_samples(X_w_norm[i:end])
    if end % 2_000_000 == 0 or end == n_samples:
        ts(f"    anomaly score: {end:,}/{n_samples:,}")

threshold = np.percentile(anomaly_scores, 95)
anomaly_flags = (anomaly_scores > threshold).astype(np.int8)
ts(f"  Anomaly rate: {anomaly_flags.mean() * 100:.2f}%")

# ------------------------------------------------------------
# Corrected evaluation: tuned ZSH vs stronger geometry-only reference
# ------------------------------------------------------------
ts("\n" + "=" * 70)
ts("STEP 6: Corrected evaluation — ZSH vs geometry-only reference")
ts("=" * 70)

per_cl = max(1, EVAL_SAMPLE // K_CLUSTERS)
eval_idx = []
for cl in range(K_CLUSTERS):
    cl_idx = np.where(zsh_labels == cl)[0]
    n_pick = min(per_cl, len(cl_idx))
    if n_pick > 0:
        eval_idx.extend(rng.choice(cl_idx, size=n_pick, replace=False).tolist())
eval_idx = np.array(eval_idx, dtype=np.int64)
rng.shuffle(eval_idx)

ts(f"  Evaluation set: {len(eval_idx):,} points")

X_eval = X_w_norm[eval_idx]
y_zsh_eval = zsh_labels[eval_idx]
y_base_eval = baseline_labels[eval_idx]
y_ref_eval = reference_km.predict(X_eval)

zsh_metrics = score_labels(X_eval, y_zsh_eval)
base_metrics = score_labels(X_eval, y_base_eval)
ref_metrics = score_labels(X_eval, y_ref_eval)

reference_method = "KMeans++ Elkan (geometry-only reference)"

sil_zsh = zsh_metrics["silhouette"]
dbi_zsh = zsh_metrics["dbi"]
chi_zsh = zsh_metrics["chi"]
sil_baseline = ref_metrics["silhouette"]
dbi_baseline = ref_metrics["dbi"]
chi_baseline = ref_metrics["chi"]
sil_mb = base_metrics["silhouette"]
dbi_mb = base_metrics["dbi"]
chi_mb = base_metrics["chi"]

sil_imp = ((sil_zsh - sil_baseline) / abs(sil_baseline)) * 100
dbi_imp = ((dbi_baseline - dbi_zsh) / abs(dbi_baseline)) * 100
chi_imp = ((chi_zsh - chi_baseline) / abs(chi_baseline)) * 100

beats_sil = sil_zsh > sil_baseline
beats_dbi = dbi_zsh < dbi_baseline
beats_chi = chi_zsh > chi_baseline
beats_all = beats_sil and beats_dbi and beats_chi

ts("\n" + "=" * 72)
ts("RESULTS SUMMARY")
ts("=" * 72)
ts(f"{'Metric':<38} {'KMeans++ Elkan':>16} {'Tuned ZSH':>12} {'Delta':>8}")
ts("-" * 72)
ts(
    f"{'Silhouette  (higher = better)':<38} "
    f"{sil_baseline:>16.4f} {sil_zsh:>12.4f} {sil_imp:>+7.1f}%  {'✓' if beats_sil else '✗'}"
)
ts(
    f"{'Davies-Bouldin (lower = better)':<38} "
    f"{dbi_baseline:>16.4f} {dbi_zsh:>12.4f} {dbi_imp:>+7.1f}%  {'✓' if beats_dbi else '✗'}"
)
ts(
    f"{'Calinski-Harabasz (higher=better)':<38} "
    f"{chi_baseline:>16.1f} {chi_zsh:>12.1f} {chi_imp:>+7.1f}%  {'✓' if beats_chi else '✗'}"
)
ts(f"{'Clusters':<38} {K_CLUSTERS:>16d} {K_CLUSTERS:>12d}  {'-':>8}")
ts(f"{'Semantic labels':<38} {'No':>16} {'Yes':>12}  {'✓':>8}")
ts(
    f"{'Anomaly detection':<38} {'No':>16} "
    f"{f'{anomaly_flags.mean() * 100:.1f}%':>12}  {'✓':>8}"
)
ts(f"{'Selected feature space':<38} {'X_w_norm':>16} {selected_feature_space:>12}  {'linear':>8}")
ts(f"{'Evaluation space':<38} {'X_w_norm':>16} {'X_w_norm':>12}  {'same':>8}")
ts("=" * 72)
ts(
    "  Auxiliary MiniBatch baseline on the same evaluation set: "
    f"Sil={sil_mb:.4f}  DBI={dbi_mb:.4f}  CHI={chi_mb:.1f}"
)

if beats_all:
    ts("\nSUCCESS: tuned ZSH beats the geometry-only reference on all three corrected metrics.")
else:
    ts(f"\nZSH does not beat {reference_method} on every intrinsic metric:")
    if not beats_sil:
        ts(f"  - Silhouette : ZSH={sil_zsh:.4f}  <  Ref={sil_baseline:.4f}")
    if not beats_dbi:
        ts(f"  - DBI        : ZSH={dbi_zsh:.4f}  >  Ref={dbi_baseline:.4f}")
    if not beats_chi:
        ts(f"  - CHI        : ZSH={chi_zsh:.1f}  <  Ref={chi_baseline:.1f}")

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
ts(f"\nSaving outputs to {OUTPUT_DIR} ...")

np.save(os.path.join(OUTPUT_DIR, "zsh_improved_labels.npy"), zsh_labels)
np.save(os.path.join(OUTPUT_DIR, "zsh_improved_profiles.npy"), zsh_profiles)
np.save(os.path.join(OUTPUT_DIR, "zsh_improved_anomaly_flags.npy"), anomaly_flags)
np.save(os.path.join(OUTPUT_DIR, "zsh_improved_anomaly_scores.npy"), anomaly_scores)

comparison_df = pd.DataFrame(
    [
        {
            "Metric": "Silhouette",
            "Baseline_Method": reference_method,
            "Baseline_KMeans": sil_baseline,
            "Reference_Score": sil_baseline,
            "Improved_ZSH": sil_zsh,
            "Improvement_Pct": sil_imp,
            "ZSH_Better": beats_sil,
            "Selected_Candidate": selected["name"],
            "Selected_Feature_Space": selected_feature_space,
        },
        {
            "Metric": "Davies_Bouldin",
            "Baseline_Method": reference_method,
            "Baseline_KMeans": dbi_baseline,
            "Reference_Score": dbi_baseline,
            "Improved_ZSH": dbi_zsh,
            "Improvement_Pct": dbi_imp,
            "ZSH_Better": beats_dbi,
            "Selected_Candidate": selected["name"],
            "Selected_Feature_Space": selected_feature_space,
        },
        {
            "Metric": "Calinski_Harabasz",
            "Baseline_Method": reference_method,
            "Baseline_KMeans": chi_baseline,
            "Reference_Score": chi_baseline,
            "Improved_ZSH": chi_zsh,
            "Improvement_Pct": chi_imp,
            "ZSH_Better": beats_chi,
            "Selected_Candidate": selected["name"],
            "Selected_Feature_Space": selected_feature_space,
        },
    ]
)
comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "zsh_improved_comparison.csv"), index=False
)

zsh_method_name = (
    "ZSH-G Metric"
    if selected_feature_space == "X_zshg_metric"
    else (
        "ZSH Geometry-First"
        if selected_feature_space == "X_weighted_geometry_whitened"
        else "ZSH Hybrid"
    )
)

pd.DataFrame(
    [
        {
            "method": "MiniBatch KMeans",
            "silhouette": sil_mb,
            "dbi": dbi_mb,
            "chi": chi_mb,
        },
        {
            "method": "KMeans++ Elkan",
            "silhouette": sil_baseline,
            "dbi": dbi_baseline,
            "chi": chi_baseline,
        },
        {
            "method": zsh_method_name,
            "silhouette": sil_zsh,
            "dbi": dbi_zsh,
            "chi": chi_zsh,
            "candidate": selected["name"],
            "feature_space": selected_feature_space,
        },
    ]
).to_csv(os.path.join(OUTPUT_DIR, "zsh_reference_comparison.csv"), index=False)

profile_dist = pd.Series(zsh_profiles).value_counts().reset_index()
profile_dist.columns = ["Profile", "Count"]
profile_dist["Percentage"] = profile_dist["Count"] / n_samples * 100
profile_dist.to_csv(
    os.path.join(OUTPUT_DIR, "zsh_improved_profiles.csv"), index=False
)

ts("\nTop 20 profiles by size:")
for _, row in profile_dist.head(20).iterrows():
    ts(f"  {row['Profile']:<28}  {int(row['Count']):>10,}  ({row['Percentage']:.2f}%)")

ts(f"\nTotal runtime: {time.time() - _T0:.1f}s")
ts("=" * 70)
ts("STEP 5 COMPLETE")
ts("=" * 70)
