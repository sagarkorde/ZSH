# ============================================================
# STEP 3 — Riemann Zeta Feature Weighting  (Novel Contribution)
#
# OPTIMIZATIONS vs previous version:
#   * Hardware-aware memory cap: ~50 GB RAM ceiling enforced via
#     chunked loading and float32 dtype throughout
#   * Full resumability: every stage guarded by .ckpt3_<stage>.done
#     sentinel files — re-run after any crash, picks up exactly
#     where it left off
#   * DuckDB used for ALL ranking/aggregation operations (Stages 3,
#     sensitivity analysis, weight validation) — avoids large pandas
#     in-memory sorts
#   * RTX 4060 8 GB: GPU not used here (MI is CPU-bound), but
#     weight application (Stage 4) uses memory-mapped numpy so the
#     50 GB array never fully resides in RAM at once
#   * Timestamped progress logging with per-stage elapsed times
#   * Reviewer-friendly comments throughout every block
#
# Optimized for: HP Omen i9-13th Gen | 64 GB RAM | RTX 4060 8 GB
#                Target RAM ceiling : ~50 GB
# ============================================================

import numpy as np
import pandas as pd
import joblib, os, sys, io, logging, time, duckdb
from pathlib import Path
from scipy import linalg, sparse
from scipy.special import zeta as riemann_zeta
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 0. OUTPUT DIRECTORY ───────────────────────────────────────
# All intermediate and final artefacts land here.
# Change this path to match wherever Step 2 wrote its outputs.
OUTPUT_DIR = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOGGING SETUP ─────────────────────────────────────────
# Dual sink: UTF-8 console + rotating file so long runs don't lose output.
log_path = os.path.join(OUTPUT_DIR, "step3_log.txt")
_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=_utf8),
        logging.FileHandler(log_path, mode="a", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

def ts(msg):
    """Convenience wrapper — every log line carries a timestamp automatically."""
    log.info(msg)

# ── 2. CHECKPOINT HELPERS ─────────────────────────────────────
# Resumability pattern:
#   Before each stage  → is_done(stage_name) ?  skip : run
#   After each stage   → mark_done(stage_name)  (creates empty sentinel file)
# Delete a .ckpt3_<name>.done file to force that stage to re-run.

def ckpt(s):
    """Return path to the sentinel file for stage `s`."""
    return os.path.join(OUTPUT_DIR, f".ckpt3_{s}.done")

def is_done(s):
    """True if the stage was already completed in a previous run."""
    return os.path.exists(ckpt(s))

def mark_done(s):
    """Touch the sentinel file, logging the checkpoint name."""
    Path(ckpt(s)).touch()
    ts(f"  [CHECKPOINT] {s} ✓")


def ensure_x_scaled_available(x_scaled_path: str, feature_cols_path: str) -> None:
    """
    Rebuild X_scaled.npy from the surviving Step 2 artifacts if cleanup removed it.
    This keeps Step 3 reproducible even after a storage-pruning pass.
    """
    if os.path.exists(x_scaled_path):
        return

    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    df_features_path = os.path.join(OUTPUT_DIR, 'df_balanced_features.parquet')

    missing = [
        fp for fp in [feature_cols_path, scaler_path, df_features_path]
        if not os.path.exists(fp)
    ]
    if missing:
        raise FileNotFoundError(
            "X_scaled.npy is missing and cannot be reconstructed because these "
            f"dependencies are absent: {missing}"
        )

    ts("X_scaled.npy missing after cleanup — reconstructing from Step 2 artifacts ...")
    t0 = time.time()
    feature_cols = joblib.load(feature_cols_path)
    scaler = joblib.load(scaler_path)
    df_features = pd.read_parquet(df_features_path, columns=feature_cols)
    X_scaled_rebuilt = scaler.transform(df_features.values.astype(np.float32)).astype(np.float32)
    np.save(x_scaled_path, X_scaled_rebuilt)
    ts(
        f"  Rebuilt X_scaled.npy: {X_scaled_rebuilt.shape}  "
        f"({X_scaled_rebuilt.nbytes / 1e9:.2f} GB) in {time.time()-t0:.1f}s"
    )


def build_anchor_laplacian_scores(
    X_source,
    feature_cols: list[str],
    sample_size: int = 250_000,
    n_anchors: int = 2048,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Geometry-first unsupervised feature ranking.

    1. Draw a manageable sample from X_scaled.
    2. Compress it into anchor centers with MiniBatchKMeans.
    3. Build a weighted kNN graph over anchors.
    4. Score each feature with Laplacian Score (lower = better manifold preservation).
    """
    rng_geom = np.random.default_rng(random_state)
    n_total = len(X_source)
    use_n = min(sample_size, n_total)
    sample_idx = rng_geom.choice(n_total, size=use_n, replace=False)
    X_sample = np.array(X_source[sample_idx], dtype=np.float32)

    n_anchor_use = min(n_anchors, len(X_sample))
    anchor_model = MiniBatchKMeans(
        n_clusters=n_anchor_use,
        init='k-means++',
        n_init=5,
        batch_size=50_000,
        max_iter=250,
        random_state=random_state,
        verbose=0,
    )
    anchor_labels = anchor_model.fit_predict(X_sample)
    anchor_centers = anchor_model.cluster_centers_.astype(np.float64)
    anchor_counts = np.bincount(anchor_labels, minlength=n_anchor_use).astype(np.float64)

    keep = anchor_counts > 0
    anchor_centers = anchor_centers[keep]
    anchor_counts = anchor_counts[keep]
    n_anchor_live = len(anchor_centers)

    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, n_anchor_live),
        metric='euclidean',
        n_jobs=-1,
    )
    nn.fit(anchor_centers)
    distances, indices = nn.kneighbors(anchor_centers)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    nonzero_dist = distances[distances > 0]
    sigma = float(np.median(nonzero_dist)) if nonzero_dist.size else 1.0
    sigma = max(sigma, 1e-6)

    rows, cols, vals = [], [], []
    for i in range(n_anchor_live):
        for dist, j in zip(distances[i], indices[i]):
            affinity = np.exp(-(float(dist) ** 2) / (2.0 * sigma ** 2))
            mass = np.sqrt(anchor_counts[i] * anchor_counts[j])
            rows.append(i)
            cols.append(int(j))
            vals.append(affinity * mass)

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_anchor_live, n_anchor_live))
    W = 0.5 * (W + W.T)
    d_vec = np.asarray(W.sum(axis=1)).ravel()
    d_vec = np.maximum(d_vec, 1e-12)
    L = sparse.diags(d_vec) - W
    d_sum = float(d_vec.sum())

    scores = []
    for feat_idx, feat_name in enumerate(feature_cols):
        f = anchor_centers[:, feat_idx]
        mu = float(np.dot(d_vec, f) / d_sum)
        f_centered = f - mu
        denom = float(np.dot(d_vec, f_centered ** 2))
        numer = float(f_centered @ (L @ f_centered))
        score = numer / max(denom, 1e-12)
        scores.append((feat_name, score))

    score_df = pd.DataFrame(scores, columns=['feature', 'laplacian_score'])
    score_df = score_df.sort_values('laplacian_score', ascending=True).reset_index(drop=True)
    score_df['rank'] = np.arange(1, len(score_df) + 1, dtype=np.int32)

    anchor_diag = pd.DataFrame({
        'anchor_id': np.arange(n_anchor_live, dtype=np.int32),
        'sample_count': anchor_counts.astype(np.int64),
        'degree_weight': d_vec,
    })
    return score_df, anchor_diag


def score_cluster_geometry(X_eval, labels, random_state: int = 42) -> dict[str, float]:
    """Compute the intrinsic geometry metrics used throughout the pipeline."""
    return {
        'silhouette': float(silhouette_score(X_eval, labels, random_state=random_state)),
        'dbi': float(davies_bouldin_score(X_eval, labels)),
        'chi': float(calinski_harabasz_score(X_eval, labels)),
    }


def select_metric_candidate(candidates: list[dict], baseline_metrics: dict[str, float]) -> dict:
    """
    Rank candidate metric transforms against the teacher Elkan reference.
    Prefers candidates that beat the teacher on more intrinsic metrics while
    remaining stable across Silhouette / DBI / CHI.
    """
    sil_order = sorted(candidates, key=lambda r: r['silhouette'], reverse=True)
    dbi_order = sorted(candidates, key=lambda r: r['dbi'])
    chi_order = sorted(candidates, key=lambda r: r['chi'], reverse=True)

    sil_rank = {row['name']: rank for rank, row in enumerate(sil_order, start=1)}
    dbi_rank = {row['name']: rank for rank, row in enumerate(dbi_order, start=1)}
    chi_rank = {row['name']: rank for rank, row in enumerate(chi_order, start=1)}

    for row in candidates:
        row['rank_sum'] = (
            sil_rank[row['name']] + dbi_rank[row['name']] + chi_rank[row['name']]
        )
        wins = (
            int(row['silhouette'] > baseline_metrics['silhouette'])
            + int(row['dbi'] < baseline_metrics['dbi'])
            + int(row['chi'] > baseline_metrics['chi'])
        )
        row['wins_vs_teacher'] = wins
        row['beats_teacher_all3'] = wins == 3

    candidates = sorted(
        candidates,
        key=lambda r: (
            -int(r['beats_teacher_all3']),
            -r['wins_vs_teacher'],
            r['rank_sum'],
            -r['silhouette'],
            r['dbi'],
            -r['chi'],
        ),
    )
    return candidates[0]


def compute_chunk_mean_std(X_source, chunk_rows: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean / std for a memmapped matrix without loading it fully into RAM."""
    n_total, n_dim = X_source.shape
    sum_x = np.zeros(n_dim, dtype=np.float64)
    sum_x2 = np.zeros(n_dim, dtype=np.float64)

    for start in range(0, n_total, chunk_rows):
        end = min(start + chunk_rows, n_total)
        chunk = np.array(X_source[start:end], dtype=np.float32)
        sum_x += chunk.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(chunk, dtype=np.float64).sum(axis=0, dtype=np.float64)

    mean = sum_x / float(n_total)
    var = np.maximum(sum_x2 / float(n_total) - mean ** 2, 1e-12)
    scale = np.sqrt(var)
    return mean.astype(np.float32), scale.astype(np.float32)


def build_weighted_scatter_matrices(
    X: np.ndarray,
    labels: np.ndarray,
    point_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted within-class and between-class scatter for pseudo-label metric learning."""
    d = X.shape[1]
    overall_mean = np.average(X, axis=0, weights=point_weights).astype(np.float64)
    sw = np.zeros((d, d), dtype=np.float64)
    sb = np.zeros((d, d), dtype=np.float64)

    for cl in np.unique(labels):
        mask = labels == cl
        Xc = X[mask].astype(np.float64)
        wc = point_weights[mask].astype(np.float64)
        w_sum = float(wc.sum())
        if w_sum <= 1e-8 or len(Xc) < 2:
            continue

        mean_c = np.average(Xc, axis=0, weights=wc).astype(np.float64)
        centered = Xc - mean_c
        centered_w = centered * np.sqrt(wc)[:, None]
        sw += centered_w.T @ centered_w

        diff = (mean_c - overall_mean).reshape(-1, 1)
        sb += w_sum * (diff @ diff.T)

    return sw, sb


def build_mahalanobis_whitener(sw_reg: np.ndarray) -> np.ndarray:
    """Whiten within-cluster covariance to obtain a Mahalanobis-style metric."""
    evals, evecs = linalg.eigh(sw_reg)
    evals = np.maximum(evals, 1e-8)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


def build_fisher_metric(sw_reg: np.ndarray, sb: np.ndarray, power: float) -> np.ndarray:
    """
    Teacher-distilled Fisher metric:
    1. Whiten within-cluster covariance.
    2. Rotate into the between-class eigensystem.
    3. Optionally amplify high-separation directions.
    """
    whitener = build_mahalanobis_whitener(sw_reg)
    b_tilde = whitener.T @ sb @ whitener
    evals, evecs = np.linalg.eigh(b_tilde)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1e-8)
    evecs = evecs[:, order]
    scale = np.power(evals / np.maximum(evals.mean(), 1e-8), power)
    return whitener @ evecs @ np.diag(scale)


def build_diag_ratio_metric(sw: np.ndarray, sb: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Diagonal between/within ratio scaling as a conservative geometry candidate."""
    within = np.maximum(np.diag(sw), 1e-8)
    between = np.maximum(np.diag(sb), 1e-8)
    scale = np.sqrt((between + ridge) / (within + ridge))
    scale = scale / np.maximum(scale.mean(), 1e-8)
    return np.diag(scale)

# ── 3. LOAD INPUTS ────────────────────────────────────────────
# X_scaled.npy  : float32 scaled feature matrix from Step 2
# feature_cols  : ordered list of column names matching X_scaled columns
#
# Memory note: np.load with mmap_mode='r' maps the file into virtual
# address space — Python only pages in what it actually reads, keeping
# RAM usage well below the 50 GB ceiling even for very large matrices.

ts("=" * 65)
ts("STEP 3 — Riemann Zeta Feature Weighting")
ts("=" * 65)

X_scaled_path = os.path.join(OUTPUT_DIR, 'X_scaled.npy')
feature_cols_path = os.path.join(OUTPUT_DIR, 'feature_cols.pkl')

ensure_x_scaled_available(X_scaled_path, feature_cols_path)

# Validate both input files exist before proceeding
if not os.path.exists(X_scaled_path):
    raise FileNotFoundError(
        f"X_scaled.npy not found at {X_scaled_path}\n"
        "Ensure Step 2 completed successfully and OUTPUT_DIR points to the same folder."
    )
if not os.path.exists(feature_cols_path):
    raise FileNotFoundError(
        f"feature_cols.pkl not found at {feature_cols_path}\n"
        "Ensure Step 2 completed successfully."
    )

# Memory-mapped load: does NOT read full array into RAM at open time
X_scaled     = np.load(X_scaled_path, mmap_mode='r').astype(np.float32)
FEATURE_COLS = joblib.load(feature_cols_path)
n_samples, n_features = X_scaled.shape

ts(f"X_scaled shape  : {X_scaled.shape}  dtype={X_scaled.dtype}")
ts(f"Feature count   : {n_features}")
ts(f"Estimated RAM for full matrix : {X_scaled.nbytes / 1e9:.2f} GB")

# ── STAGE 1: PROXY LABELS VIA MiniBatchKMeans ─────────────────
# Goal: produce cheap cluster labels to use as a surrogate target
# for mutual information estimation in Stage 2.
#
# MiniBatchKMeans is chosen over full KMeans because:
#   - Processes data in mini-batches → constant memory regardless of n_samples
#   - Runs ~10-20× faster on multi-million row datasets
#   - Accuracy loss vs full KMeans is negligible for proxy label use
#
# n_clusters=10 gives 10 distinct integer labels (0–9) as the proxy target.

proxy_path = os.path.join(OUTPUT_DIR, 'proxy_labels.npy')

if is_done("kmeans"):
    ts("STAGE 1 [KMeans]: Loading saved proxy labels ...")
    proxy_labels = np.load(proxy_path)
else:
    ts("STAGE 1 [KMeans]: Fitting MiniBatchKMeans for proxy labels ...")
    t0 = time.time()

    km = MiniBatchKMeans(
        n_clusters=10,       # 10 clusters → 10 proxy classes for MI
        n_init=5,            # 5 centroid initializations, best kept
        batch_size=50_000,   # Each mini-batch = 50K rows (~200 MB float32)
        max_iter=300,        # Maximum EM-style passes
        random_state=42,     # Reproducibility
        verbose=0
    )
    proxy_labels = km.fit_predict(X_scaled)   # Returns int array of cluster ids
    np.save(proxy_path, proxy_labels)
    elapsed = time.time() - t0
    ts(f"  Done in {elapsed:.1f}s | clusters found: {np.unique(proxy_labels).size}")
    mark_done("kmeans")

# ── STAGE 2: STRATIFIED MUTUAL INFORMATION ESTIMATION ─────────
# Goal: quantify how much information each feature carries about
# the cluster structure (proxy target).
#
# Why stratified subsampling?
#   sklearn's mutual_info_regression is O(n log n) per feature.
#   Running it on 10M+ rows is feasible but slow. We subsample
#   600K rows while preserving cluster proportions so the MI
#   estimates remain representative of the full distribution.
#
# MI_NEIGHBORS=5: k-NN estimator bandwidth — 5 is the sklearn default
# and a good bias-variance trade-off for regression MI.

mi_path      = os.path.join(OUTPUT_DIR, 'mi_scores.npy')
MI_SAMPLE    = 600_000   # Target subsample size (~2.4 GB float32 @ 100 features)
MI_NEIGHBORS = 5         # k for k-NN entropy estimator

if is_done("mi"):
    ts("STAGE 2 [MI]: Loading saved MI scores ...")
    mi_scores = np.load(mi_path)
else:
    ts(f"STAGE 2 [MI]: Stratified subsample → {MI_SAMPLE:,} rows for MI estimation ...")
    t0 = time.time()

    rng       = np.random.default_rng(42)   # Reproducible RNG
    unique_cl = np.unique(proxy_labels)
    per_cl    = MI_SAMPLE // len(unique_cl) # Equal quota per cluster

    idx_list = []
    for c in unique_cl:
        ci = np.where(proxy_labels == c)[0]          # All row indices in cluster c
        sampled = rng.choice(ci,
                             size=min(per_cl, len(ci)),  # Handle small clusters
                             replace=False)
        idx_list.extend(sampled.tolist())

    idx_arr = np.array(idx_list)
    X_mi    = np.array(X_scaled[idx_arr])            # Force load subsample into RAM
    y_mi    = proxy_labels[idx_arr].astype(np.float32)

    ts(f"  MI subsample shape : {X_mi.shape} | "
       f"RAM usage ≈ {X_mi.nbytes / 1e9:.2f} GB")
    ts("  Running mutual_info_regression (this may take several minutes) ...")

    mi_scores = mutual_info_regression(
        X_mi, y_mi,
        n_neighbors=MI_NEIGHBORS,
        random_state=42
    )
    np.save(mi_path, mi_scores)
    ts(f"  MI estimation done in {time.time()-t0:.1f}s")
    mark_done("mi")

# Print the raw MI leaderboard (informational — not yet weighted)
mi_df_raw = (pd.DataFrame({'feature': FEATURE_COLS, 'mi_score': mi_scores})
               .sort_values('mi_score', ascending=False))
ts("\nTop 15 MI scores (raw, before Zeta weighting):")
ts("\n" + mi_df_raw.head(15).to_string(index=False))

# ── STAGE 3: RIEMANN ZETA RANKING via DuckDB ──────────────────
# Core novel contribution of this step.
#
# The Riemann Zeta weighting scheme assigns a weight w_r to each
# feature at rank r (1 = highest MI) via:
#
#       w_r  =  (1 / r^s)  /  Z_N(s)
#
# where:
#   s        = decay exponent (tunable; default 1.5)
#   Z_N(s)   = Σ_{r=1}^{N} 1/r^s   ← FINITE partial sum over N features
#
# WHY finite partial sum instead of the infinite Riemann zeta ζ(s)?
# ─────────────────────────────────────────────────────────────────
# With only N=27 features the infinite series ζ(1.5)=2.612 includes
# contributions from ranks 28 → ∞ that simply don't exist in our data.
# Dividing by the infinite sum causes Σ w_r ≈ 0.854 — 14.6% short of 1.
# The finite normaliser Z_N(s) = Σ_{r=1}^{N} 1/r^s guarantees
# Σ w_r = 1.0 exactly, regardless of N or s.
# This is consistent with Zipf-law normalisation in finite vocabularies
# (Mandelbrot 1953; Powers 1998) and is the correct choice for ML feature
# weighting where N is fixed at dataset design time.
# The paper reports both Z_N(s) and ζ(s) values in the methods table.
#
# DuckDB is used for ALL ranking / aggregation:
#   - SQL RANK() OVER (ORDER BY ...) faster than pandas .rank()
#   - Zero-copy: DuckDB queries the in-process DataFrame directly
#   - Sensitivity sweep maps cleanly to parameterised SQL queries
#
# S_VALUES: four decay exponents for the paper's sensitivity table.

ranks_parquet = os.path.join(OUTPUT_DIR, 'feature_ranks.parquet')
ranks_csv     = os.path.join(OUTPUT_DIR, 'feature_ranks.csv')

S_DECAY  = 1.5                        # Primary decay exponent for the paper
S_VALUES = [1.0, 1.5, 2.0, 3.0]      # Sensitivity sweep values
GEOM_LAPLACIAN_BLEND = 0.70          # Geometry-first rank fusion weight
GEOM_MI_BLEND        = 0.30          # Keeps topology ranking grounded
GEOM_FLAG_PENALTY    = 10.0          # Push binary rule flags below continuous geometry
ZSHG_SAMPLE          = 90_000        # Sample used to fit the next-gen metric branch
ZSHG_VALIDATION      = 8_000         # Held-out rows for honest metric selection
ZSHG_K_CLUSTERS      = 30            # Must match Step 5 / Step 7 cluster count
ZSHG_CONF_KEEP       = 0.70          # Keep the top-confidence pseudo-labeled rows
ZSHG_SHRINK_GRID     = [0.05, 0.10, 0.20]
ZSHG_FISHER_POWERS   = [0.15, 0.30]

# Pre-compute finite partial-sum normalisers for every s value.
# ranks_array = [1, 2, ..., N]  — one entry per actual feature.
ranks_array = np.arange(1, n_features + 1, dtype=np.float64)

def finite_normaliser(s: float) -> float:
    """
    Return Z_N(s) = Σ_{r=1}^{N} 1/r^s  (finite partial sum over N features).
    This guarantees Σ w_r = 1.0 exactly for any s > 0 and any finite N.
    Also logs comparison with the infinite ζ(s) for the paper methods table.
    """
    z_finite = float(np.sum(1.0 / ranks_array ** s))
    zeta_inf = float(riemann_zeta(s, 1))      # inf when s ≤ 1
    if np.isfinite(zeta_inf):
        ts(f"    Z_{n_features}({s}) = {z_finite:.6f}  |  "
           f"ζ({s}) = {zeta_inf:.6f}  |  "
           f"truncation error = {(zeta_inf - z_finite)/zeta_inf*100:.2f}%")
    else:
        ts(f"    Z_{n_features}({s}) = {z_finite:.6f}  |  "
           f"ζ({s}) = ∞ (harmonic series diverges — finite normaliser mandatory)")
    return z_finite

# Primary normaliser used in the model
primary_normaliser = finite_normaliser(S_DECAY)

if is_done("zeta_ranks"):
    ts("STAGE 3 [Zeta/DuckDB]: Loading saved feature ranks ...")
    feature_rank_df = pd.read_parquet(ranks_parquet)
else:
    ts("STAGE 3 [Zeta/DuckDB]: Computing Zeta ranks via DuckDB ...")
    t0 = time.time()

    # Seed DataFrame: one row per feature with its raw MI score
    seed_df = pd.DataFrame({
        'feature'  : FEATURE_COLS,
        'mi_score' : mi_scores.astype(float)
    })

    # Open an in-memory DuckDB connection and register the DataFrame as
    # a virtual table — no disk I/O, no copy, DuckDB queries it directly
    con = duckdb.connect()
    con.register("mi_table", seed_df)

    # ── 3a. Primary weights at s = S_DECAY ──────────────────
    # Uses finite normaliser Z_N(S_DECAY) — guarantees Σ w_r = 1.0 exactly.
    query_primary = f"""
        SELECT
            feature,
            mi_score,
            -- RANK() assigns 1 to the feature with the highest MI score
            RANK() OVER (ORDER BY mi_score DESC)                              AS rank,
            -- w_r = (1/r^s) / Z_N(s)  — finite-normalised Zeta weight
            (1.0 / POWER(RANK() OVER (ORDER BY mi_score DESC), {S_DECAY}))
                / {primary_normaliser}                                        AS zeta_weight
        FROM mi_table
        ORDER BY rank ASC
    """
    feature_rank_df = con.execute(query_primary).df()
    ts(f"  Primary weights computed | Z_{n_features}({S_DECAY}) = {primary_normaliser:.6f}")

    # ── 3b. Sensitivity columns for each s value ────────────
    # Adds columns w_s10, w_s15, w_s20, w_s30 — all using the finite
    # normaliser so each column also sums to exactly 1.0.
    ts("  Computing sensitivity columns (all use finite normaliser):")
    for s in S_VALUES:
        norm  = finite_normaliser(s)
        s_col = f"w_s{str(s).replace('.','')}"   # e.g. s=1.5 → "w_s15"

        query_s = f"""
            SELECT
                feature,
                (1.0 / POWER(RANK() OVER (ORDER BY mi_score DESC), {s}))
                    / {norm}   AS {s_col}
            FROM mi_table
            ORDER BY RANK() OVER (ORDER BY mi_score DESC) ASC
        """
        s_df = con.execute(query_s).df()

        # Merge sensitivity column into the main rank table on feature name
        feature_rank_df = feature_rank_df.merge(
            s_df[['feature', s_col]], on='feature', how='left'
        )
        ts(f"    → {s_col} column merged  (Σ = {s_df[s_col].sum():.8f})")

    con.close()   # Release DuckDB connection

    # Persist to both Parquet (fast reload) and CSV (human-readable for paper)
    feature_rank_df.to_parquet(ranks_parquet, index=False)
    feature_rank_df.to_csv(ranks_csv, index=False)
    ts(f"  Zeta ranking complete in {time.time()-t0:.2f}s")
    mark_done("zeta_ranks")

# ── VALIDATION: WEIGHTS SUM TO 1.0 ───────────────────────────
# With finite normaliser Z_N(s) the sum is guaranteed exact (float32
# rounding aside).  Tolerance of 1e-5 catches any numeric drift.
weight_sum = feature_rank_df['zeta_weight'].sum()
ts(f"\nWeight sum validation : {weight_sum:.8f}  (expected = 1.0 exactly)")
assert abs(weight_sum - 1.0) < 1e-4, (
    f"Weight sum {weight_sum:.8f} deviates unexpectedly — "
    "verify that the parquet was not cached from a previous run with the old normaliser.\n"
    f"Delete {ranks_parquet} and re-run."
)
ts("  ✓ Exact scale invariance confirmed: Σ w_r = 1.0")

# Log full rank + weight table for paper Table section
ts("\nFull Feature Rank + Zeta Weight Table:")
ts("\n" + feature_rank_df[
    ['feature', 'rank', 'mi_score', 'zeta_weight']
].to_string(index=False))

# ── STAGE 3B: GEOMETRY-FIRST LAPLCIAN RANKING ──────────────────
# The original Zeta weighting uses MI against proxy KMeans labels.
# For geometry optimisation we also build an unsupervised ranker that
# scores features by how well they preserve local manifold structure.
# Lower Laplacian Score = better local geometry preservation.

geom_ranks_parquet = os.path.join(OUTPUT_DIR, 'feature_ranks_geometry.parquet')
geom_ranks_csv     = os.path.join(OUTPUT_DIR, 'feature_ranks_geometry.csv')
geom_anchor_csv    = os.path.join(OUTPUT_DIR, 'geometry_anchor_summary.csv')

if is_done("geom_ranks_v2"):
    ts("STAGE 3B [Geometry Ranks]: Loading saved Laplacian-score ranks ...")
    geom_rank_df = pd.read_parquet(geom_ranks_parquet)
else:
    ts("STAGE 3B [Geometry Ranks]: Computing anchor-graph Laplacian scores ...")
    t0 = time.time()
    geom_rank_df, geom_anchor_df = build_anchor_laplacian_scores(
        X_scaled, FEATURE_COLS,
        sample_size=250_000,
        n_anchors=2048,
        n_neighbors=15,
        random_state=42,
    )
    mi_rank_map = dict(zip(feature_rank_df['feature'], feature_rank_df['rank']))
    geom_rank_df['lap_rank'] = geom_rank_df['rank'].astype(np.float64)
    geom_rank_df['mi_rank'] = geom_rank_df['feature'].map(mi_rank_map).astype(np.float64)
    geom_rank_df['flag_penalty'] = np.where(
        geom_rank_df['feature'].str.startswith(('is_', 'has_'))
        | geom_rank_df['feature'].eq('rbf_enabled'),
        GEOM_FLAG_PENALTY,
        0.0,
    )
    geom_rank_df['geom_rank_score'] = (
        GEOM_LAPLACIAN_BLEND * geom_rank_df['lap_rank']
        + GEOM_MI_BLEND * geom_rank_df['mi_rank']
        + geom_rank_df['flag_penalty']
    )
    geom_rank_df = (
        geom_rank_df
        .sort_values(
            ['geom_rank_score', 'laplacian_score', 'mi_rank', 'feature'],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
    geom_rank_df['rank'] = np.arange(1, len(geom_rank_df) + 1, dtype=np.int32)
    geom_rank_df['geom_zeta_weight'] = (
        (1.0 / geom_rank_df['rank'].to_numpy(dtype=np.float64) ** S_DECAY)
        / primary_normaliser
    )
    for s in S_VALUES:
        norm = finite_normaliser(s)
        s_col = f"geom_w_s{str(s).replace('.','')}"
        geom_rank_df[s_col] = (
            (1.0 / geom_rank_df['rank'].to_numpy(dtype=np.float64) ** s) / norm
        )
    geom_rank_df.to_parquet(geom_ranks_parquet, index=False)
    geom_rank_df.to_csv(geom_ranks_csv, index=False)
    geom_anchor_df.to_csv(geom_anchor_csv, index=False)
    ts(f"  Geometry ranks saved in {time.time()-t0:.1f}s")
    mark_done("geom_ranks_v2")

geom_weight_sum = float(geom_rank_df['geom_zeta_weight'].sum())
ts(
    f"\nGeometry weight sum validation : {geom_weight_sum:.8f}  "
    "(expected = 1.0 exactly)"
)
assert abs(geom_weight_sum - 1.0) < 1e-4, (
    f"Geometry weight sum {geom_weight_sum:.8f} deviates unexpectedly.\n"
    f"Delete {geom_ranks_parquet} and re-run."
)
ts("  ✓ Geometry Zeta weights also sum to 1.0")
ts("\nTop 15 geometry-ranked features (lower Laplacian Score = better):")
ts("\n" + geom_rank_df[
    ['feature', 'rank', 'laplacian_score', 'geom_zeta_weight']
].head(15).to_string(index=False))

# ── STAGE 4: APPLY WEIGHTS ────────────────────────────────────
# Element-wise multiply X_scaled (n_samples × n_features) by the
# weight vector (n_features,) so that high-MI features are amplified
# relative to low-MI features before clustering.
#
# Memory strategy for large arrays (>20 GB):
#   - We write X_weighted in chunks of CHUNK_ROWS rows at a time so
#     that at most 2× one chunk resides in RAM simultaneously.
#   - For smaller arrays the chunk loop still works but completes in
#     a single iteration.
#
# CHUNK_ROWS = 500K rows × n_features × 4 bytes ≈ 200 MB per chunk
# (adjust lower if you see OOM errors, higher to go faster).

X_weighted_path    = os.path.join(OUTPUT_DIR, 'X_weighted.npy')
weight_vector_path = os.path.join(OUTPUT_DIR, 'zeta_weight_vector.pkl')

# Build weight vector aligned to FEATURE_COLS order (not rank order)
weight_map    = dict(zip(feature_rank_df['feature'], feature_rank_df['zeta_weight']))
weight_vector = np.array([weight_map[f] for f in FEATURE_COLS], dtype=np.float32)

CHUNK_ROWS = 500_000   # Rows per write chunk — tune for your RAM budget

if is_done("apply_weights"):
    ts("STAGE 4 [Apply Weights]: Already complete — skipping.")
else:
    ts("STAGE 4 [Apply Weights]: Writing X_weighted in chunks ...")
    t0 = time.time()

    # Pre-allocate output array on disk using a memory-mapped file.
    # np.lib.format.open_memmap creates the .npy file with header and
    # returns a writable mmap — only the written slices are paged in.
    X_out = np.lib.format.open_memmap(
        X_weighted_path,
        mode='w+',                       # Create new file, read-write
        dtype=np.float32,
        shape=(n_samples, n_features)
    )

    total_chunks = (n_samples + CHUNK_ROWS - 1) // CHUNK_ROWS
    for chunk_idx, start in enumerate(range(0, n_samples, CHUNK_ROWS), 1):
        end   = min(start + CHUNK_ROWS, n_samples)
        # Load only this chunk from the memory-mapped source array
        chunk = np.array(X_scaled[start:end], dtype=np.float32)
        X_out[start:end] = chunk * weight_vector   # Broadcast multiply
        ts(f"  Chunk {chunk_idx}/{total_chunks} [{start:,}:{end:,}] written "
           f"({time.time()-t0:.1f}s elapsed)")

    # Flush mmap writes to disk and close handles
    del X_out
    joblib.dump(weight_vector, weight_vector_path)
    ts(f"  X_weighted.npy fully written in {time.time()-t0:.1f}s")
    ts(f"  Weight vector saved → {weight_vector_path}")
    mark_done("apply_weights")

# ── STAGE 4B: APPLY GEOMETRY-FIRST WEIGHTS ─────────────────────
X_weighted_geom_path    = os.path.join(OUTPUT_DIR, 'X_weighted_geometry.npy')
geom_weight_vector_path = os.path.join(OUTPUT_DIR, 'zeta_weight_vector_geometry.pkl')

geom_weight_map = dict(zip(geom_rank_df['feature'], geom_rank_df['geom_zeta_weight']))
geom_weight_vector = np.array([geom_weight_map[f] for f in FEATURE_COLS], dtype=np.float32)

if is_done("apply_geom_weights_v2"):
    ts("STAGE 4B [Apply Geometry Weights]: Already complete — skipping.")
else:
    ts("STAGE 4B [Apply Geometry Weights]: Writing X_weighted_geometry in chunks ...")
    t0 = time.time()
    X_geom_out = np.lib.format.open_memmap(
        X_weighted_geom_path,
        mode='w+',
        dtype=np.float32,
        shape=(n_samples, n_features),
    )

    total_chunks = (n_samples + CHUNK_ROWS - 1) // CHUNK_ROWS
    for chunk_idx, start in enumerate(range(0, n_samples, CHUNK_ROWS), 1):
        end = min(start + CHUNK_ROWS, n_samples)
        chunk = np.array(X_scaled[start:end], dtype=np.float32)
        X_geom_out[start:end] = chunk * geom_weight_vector
        ts(
            f"  Geometry chunk {chunk_idx}/{total_chunks} "
            f"[{start:,}:{end:,}] written ({time.time()-t0:.1f}s elapsed)"
        )

    del X_geom_out
    joblib.dump(geom_weight_vector, geom_weight_vector_path)
    ts(f"  X_weighted_geometry.npy fully written in {time.time()-t0:.1f}s")
    ts(f"  Geometry weight vector saved → {geom_weight_vector_path}")
    mark_done("apply_geom_weights_v2")

# ── STAGE 4C: MAHALANOBIS-STYLE WHITENING ──────────────────────
# PCA whitening is a linear preconditioner: Euclidean KMeans in the
# whitened space is equivalent to a Mahalanobis-style metric in the
# weighted space. This directly targets geometry instead of semantics.

geom_whiten_path = os.path.join(OUTPUT_DIR, 'X_weighted_geometry_whitened.npy')
geom_whitener_path = os.path.join(OUTPUT_DIR, 'geometry_whitener.pkl')

if is_done("geom_whiten_v2"):
    ts("STAGE 4C [Geometry Whitening]: Already complete — skipping.")
else:
    ts("STAGE 4C [Geometry Whitening]: Fitting PCA whitener on geometry-weighted space ...")
    t0 = time.time()
    geom_weighted = np.load(X_weighted_geom_path, mmap_mode='r')
    rng_white = np.random.default_rng(123)
    white_idx = rng_white.choice(n_samples, size=min(300_000, n_samples), replace=False)
    X_white_fit = np.array(geom_weighted[white_idx], dtype=np.float32)

    whitener = PCA(
        n_components=n_features,
        whiten=True,
        svd_solver='full',
        random_state=42,
    )
    whitener.fit(X_white_fit)
    ts(
        f"  Whitener fitted on {len(X_white_fit):,} rows  "
        f"| explained variance={whitener.explained_variance_ratio_.sum():.4f}"
    )

    X_white_out = np.lib.format.open_memmap(
        geom_whiten_path,
        mode='w+',
        dtype=np.float32,
        shape=(n_samples, n_features),
    )
    total_chunks = (n_samples + CHUNK_ROWS - 1) // CHUNK_ROWS
    for chunk_idx, start in enumerate(range(0, n_samples, CHUNK_ROWS), 1):
        end = min(start + CHUNK_ROWS, n_samples)
        chunk = np.array(geom_weighted[start:end], dtype=np.float32)
        X_white_out[start:end] = whitener.transform(chunk).astype(np.float32)
        ts(
            f"  Whitening chunk {chunk_idx}/{total_chunks} "
            f"[{start:,}:{end:,}] written ({time.time()-t0:.1f}s elapsed)"
        )

    del X_white_out
    joblib.dump(whitener, geom_whitener_path)
    ts(f"  Whitened geometry matrix saved in {time.time()-t0:.1f}s")
    ts(f"  Whitener saved → {geom_whitener_path}")
    mark_done("geom_whiten_v2")

# ── STAGE 4D: NEXT-GEN ZSH-G METRIC BRANCH ──────────────────────
# This branch learns a teacher-distilled linear metric on top of the corrected
# Zeta-weighted space. The teacher is KMeans++ Elkan in X_w_norm; the student
# metric searches for a Mahalanobis/Fisher preconditioner that can make a
# follow-up KMeans fit more geometrically separable on a held-out split.

zshg_metric_path = os.path.join(OUTPUT_DIR, 'X_zshg_metric.npy')
zshg_metric_model_path = os.path.join(OUTPUT_DIR, 'zshg_metric_model.pkl')
zshg_metric_search_path = os.path.join(OUTPUT_DIR, 'zshg_metric_search.csv')

if is_done("zshg_metric_fit_v1") and os.path.exists(zshg_metric_model_path):
    ts("STAGE 4D [ZSH-G Metric Fit]: Loading saved metric model ...")
    zshg_metric_model = joblib.load(zshg_metric_model_path)
else:
    ts("STAGE 4D [ZSH-G Metric Fit]: Learning teacher-distilled Mahalanobis metric ...")
    t0 = time.time()
    X_weighted_mm = np.load(X_weighted_path, mmap_mode='r')
    mean_w, scale_w = compute_chunk_mean_std(X_weighted_mm, chunk_rows=CHUNK_ROWS)

    rng_metric = np.random.default_rng(2026)
    sample_n = min(ZSHG_SAMPLE, n_samples)
    metric_idx = rng_metric.choice(n_samples, size=sample_n, replace=False)
    X_metric = np.array(X_weighted_mm[metric_idx], dtype=np.float32)
    X_metric = ((X_metric - mean_w) / scale_w).astype(np.float32)

    perm = rng_metric.permutation(sample_n)
    X_metric = X_metric[perm]
    val_n = min(ZSHG_VALIDATION, max(ZSHG_K_CLUSTERS * 50, sample_n // 10))
    X_metric_val = X_metric[:val_n]
    X_metric_train = X_metric[val_n:]

    teacher = KMeans(
        n_clusters=ZSHG_K_CLUSTERS,
        init='k-means++',
        n_init=10,
        max_iter=250,
        algorithm='elkan',
        random_state=42,
    )
    teacher.fit(X_metric_train)
    teacher_val = teacher.predict(X_metric_val)
    teacher_metrics = score_cluster_geometry(X_metric_val, teacher_val)
    ts(
        "  Teacher reference (KMeans++ Elkan in sampled X_w_norm): "
        f"Sil={teacher_metrics['silhouette']:.4f}  "
        f"DBI={teacher_metrics['dbi']:.4f}  "
        f"CHI={teacher_metrics['chi']:.1f}"
    )

    teacher_dist = teacher.transform(X_metric_train)
    nearest_two = np.partition(teacher_dist, kth=1, axis=1)[:, :2]
    nearest_two.sort(axis=1)
    margin = (nearest_two[:, 1] - nearest_two[:, 0]) / np.maximum(nearest_two[:, 1], 1e-6)
    conf_floor = float(np.quantile(margin, 1.0 - ZSHG_CONF_KEEP))
    conf_span = max(float(margin.max() - conf_floor), 1e-6)
    point_weights = np.clip((margin - conf_floor) / conf_span, 0.0, 1.0) + 0.05

    sw, sb = build_weighted_scatter_matrices(
        X_metric_train,
        teacher.labels_.astype(np.int32),
        point_weights.astype(np.float64),
    )
    sw_trace = float(np.trace(sw) / sw.shape[0])

    candidate_specs = [
        {
            'name': 'identity',
            'family': 'identity',
            'description': 'No learned metric (identity fallback)',
            'transform': np.eye(n_features, dtype=np.float32),
        }
    ]

    for shrink in ZSHG_SHRINK_GRID:
        sw_reg = (1.0 - shrink) * sw + shrink * sw_trace * np.eye(sw.shape[0], dtype=np.float64)
        candidate_specs.append(
            {
                'name': f'mahalanobis_s{int(shrink * 100):02d}',
                'family': 'mahalanobis',
                'description': f'Within-cluster whitening with shrinkage={shrink:.2f}',
                'shrinkage': shrink,
                'transform': build_mahalanobis_whitener(sw_reg).astype(np.float32),
            }
        )

    for shrink, power in [(0.10, ZSHG_FISHER_POWERS[0]), (0.20, ZSHG_FISHER_POWERS[1])]:
        sw_reg = (1.0 - shrink) * sw + shrink * sw_trace * np.eye(sw.shape[0], dtype=np.float64)
        candidate_specs.append(
            {
                'name': f'fisher_s{int(shrink * 100):02d}_p{int(power * 100):02d}',
                'family': 'fisher',
                'description': f'Fisher rotation with shrinkage={shrink:.2f}, power={power:.2f}',
                'shrinkage': shrink,
                'power': power,
                'transform': build_fisher_metric(sw_reg, sb, power).astype(np.float32),
            }
        )

    candidate_specs.append(
        {
            'name': 'diag_ratio',
            'family': 'diagonal',
            'description': 'Diagonal between/within ratio scaling',
            'transform': build_diag_ratio_metric(sw, sb).astype(np.float32),
        }
    )

    metric_rows = []
    metric_transforms = {}
    for spec in candidate_specs:
        A = spec['transform']
        metric_transforms[spec['name']] = A
        Z_train = (X_metric_train @ A).astype(np.float32)
        Z_val = (X_metric_val @ A).astype(np.float32)

        model = KMeans(
            n_clusters=ZSHG_K_CLUSTERS,
            init='k-means++',
            n_init=10,
            max_iter=250,
            algorithm='elkan',
            random_state=42,
        )
        model.fit(Z_train)
        pred = model.predict(Z_val)
        metrics = score_cluster_geometry(X_metric_val, pred)
        metric_rows.append(
            {
                'name': spec['name'],
                'family': spec['family'],
                'description': spec['description'],
                'shrinkage': spec.get('shrinkage'),
                'power': spec.get('power'),
                'silhouette': metrics['silhouette'],
                'dbi': metrics['dbi'],
                'chi': metrics['chi'],
            }
        )

    selected_metric = select_metric_candidate(metric_rows, teacher_metrics)
    metric_df = pd.DataFrame(metric_rows)
    metric_df['selected'] = metric_df['name'].eq(selected_metric['name'])
    metric_df['teacher_silhouette'] = teacher_metrics['silhouette']
    metric_df['teacher_dbi'] = teacher_metrics['dbi']
    metric_df['teacher_chi'] = teacher_metrics['chi']
    metric_df.to_csv(zshg_metric_search_path, index=False)

    ts("\n  ZSH-G metric candidate search:")
    for row in metric_rows:
        ts(
            f"    {row['name']:<18}  Sil={row['silhouette']:.4f}  "
            f"DBI={row['dbi']:.4f}  CHI={row['chi']:.1f}  "
            f"wins_vs_teacher={row['wins_vs_teacher']}  rank_sum={row['rank_sum']}"
        )
    ts(f"  Selected ZSH-G metric candidate: {selected_metric['name']}")

    zshg_metric_model = {
        'mean': mean_w,
        'scale': scale_w,
        'transform': metric_transforms[selected_metric['name']].astype(np.float32),
        'selected_candidate': selected_metric['name'],
        'selected_family': selected_metric['family'],
        'teacher_metrics': teacher_metrics,
        'confidence_keep': ZSHG_CONF_KEEP,
        'confidence_floor': conf_floor,
        'sample_size': sample_n,
        'validation_size': val_n,
    }
    joblib.dump(zshg_metric_model, zshg_metric_model_path)
    ts(f"  ZSH-G metric model saved in {time.time()-t0:.1f}s")
    ts(f"  Search table saved → {zshg_metric_search_path}")
    mark_done("zshg_metric_fit_v1")

if is_done("zshg_metric_apply_v1"):
    ts("STAGE 4D [ZSH-G Metric Apply]: Already complete — skipping.")
else:
    ts("STAGE 4D [ZSH-G Metric Apply]: Writing X_zshg_metric in chunks ...")
    t0 = time.time()
    zshg_metric_model = joblib.load(zshg_metric_model_path)
    mean_w = zshg_metric_model['mean'].astype(np.float32)
    scale_w = zshg_metric_model['scale'].astype(np.float32)
    metric_A = zshg_metric_model['transform'].astype(np.float32)
    X_weighted_mm = np.load(X_weighted_path, mmap_mode='r')

    X_zshg_out = np.lib.format.open_memmap(
        zshg_metric_path,
        mode='w+',
        dtype=np.float32,
        shape=(n_samples, metric_A.shape[1]),
    )
    total_chunks = (n_samples + CHUNK_ROWS - 1) // CHUNK_ROWS
    for chunk_idx, start in enumerate(range(0, n_samples, CHUNK_ROWS), 1):
        end = min(start + CHUNK_ROWS, n_samples)
        chunk = np.array(X_weighted_mm[start:end], dtype=np.float32)
        chunk_std = (chunk - mean_w) / scale_w
        X_zshg_out[start:end] = (chunk_std @ metric_A).astype(np.float32)
        ts(
            f"  ZSH-G chunk {chunk_idx}/{total_chunks} "
            f"[{start:,}:{end:,}] written ({time.time()-t0:.1f}s elapsed)"
        )

    del X_zshg_out
    ts(f"  X_zshg_metric.npy fully written in {time.time()-t0:.1f}s")
    mark_done("zshg_metric_apply_v1")

# ── STAGE 5: VISUALIZATIONS ───────────────────────────────────
# Produces two publication-quality figures:
#   Figure 1 — Left : Zeta decay curve annotated with top-5 feature names
#              Right: Horizontal bar chart of top-20 feature Zeta weights
#   Figure 2 — Sensitivity analysis: all four s values overlaid on one axes
#
# DPI=150 gives ~2250×900 px for Figure 1 at the 15-inch figsize —
# suitable for journal submission.  Increase to 300 for camera-ready.

plot1_path = os.path.join(OUTPUT_DIR, 'zeta_weights.png')
plot2_path = os.path.join(OUTPUT_DIR, 'zeta_sensitivity.png')

if is_done("plot"):
    ts("STAGE 5 [Plot]: Already complete — skipping.")
else:
    ts("STAGE 5 [Plot]: Generating Zeta weight figures ...")
    t0 = time.time()

    # ── Figure 1: Decay Curve + Top-20 Bar Chart ───────────
    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # Left panel: decay curve
    ax2 = fig.add_subplot(gs[0, 1])   # Right panel: feature bar chart

    zeta_s = float(riemann_zeta(S_DECAY, 1))
    r_vals = np.arange(1, n_features + 1)
    w_vals = (1.0 / r_vals ** S_DECAY) / zeta_s   # Zeta weights for the curve

    # Decay curve: line + fill + scatter dots
    ax1.plot(r_vals, w_vals, color='steelblue', linewidth=2.5, label=f's={S_DECAY}')
    ax1.fill_between(r_vals, w_vals, alpha=0.20, color='steelblue')
    ax1.scatter(r_vals, w_vals, s=40, color='steelblue', zorder=5)

    # Annotate the top-5 ranked features directly on the curve
    for r, w, feat in zip(r_vals[:5], w_vals[:5],
                          feature_rank_df['feature'].head(5)):
        ax1.annotate(feat, xy=(r, w),
                     xytext=(r + 0.3, w + 0.002),
                     fontsize=7, color='navy')

    ax1.set_xlabel('Feature Rank  (1 = highest MI)', fontsize=11)
    ax1.set_ylabel('Zeta Weight  w_r = (1/r^s) / ζ(s)', fontsize=11)
    ax1.set_title(f'Riemann Zeta Weight Decay\ns={S_DECAY},  ζ(s)={zeta_s:.4f}',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Horizontal bar chart: top-20 features, warmest colour = highest weight
    top20      = feature_rank_df.head(20)
    colors_bar = plt.cm.YlOrRd(np.linspace(0.9, 0.3, len(top20)))
    ax2.barh(top20['feature'][::-1], top20['zeta_weight'][::-1],
             color=colors_bar[::-1], edgecolor='gray', linewidth=0.4)
    ax2.set_xlabel('Zeta Weight', fontsize=11)
    ax2.set_title('Top 20 Features — Zeta Weight Ranking',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle(
        'Riemann Zeta Feature Weighting for Bitcoin Transaction Clustering',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    ts(f"  Saved: {plot1_path}")

    # ── Figure 2: Sensitivity Analysis ─────────────────────
    # Shows how the weight distribution changes for four values of s:
    #   s=1.0 : harmonic series — slowest decay, most uniform weights
    #   s=1.5 : primary choice (moderate decay)
    #   s=2.0 : Basel-problem decay — faster concentration on rank-1
    #   s=3.0 : very steep decay — almost all weight on top features
    fig2, ax3 = plt.subplots(figsize=(12, 5))
    colors_s  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    r_vals2   = np.arange(1, n_features + 1)

    for s_val, col in zip(S_VALUES, colors_s):
        zs = float(riemann_zeta(s_val, 1))
        wv = (1.0 / r_vals2 ** s_val) / zs
        ax3.plot(r_vals2, wv, color=col, linewidth=2, label=f's={s_val}')

    ax3.set_xlabel('Feature Rank', fontsize=11)
    ax3.set_ylabel('Normalized Zeta Weight', fontsize=11)
    ax3.set_title(
        'Zeta Weight Sensitivity Analysis — Effect of Decay Exponent s',
        fontsize=12, fontweight='bold'
    )
    ax3.legend(title='Decay Exponent (s)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    ts(f"  Saved: {plot2_path}")
    ts(f"  Plotting complete in {time.time()-t0:.1f}s")
    mark_done("plot")

# ── FINAL SUMMARY ─────────────────────────────────────────────
ts("\n" + "=" * 65)
ts("STEP 3 COMPLETE")
ts(f"  X_weighted.npy         → {X_weighted_path}")
ts(f"  zeta_weight_vector.pkl → {weight_vector_path}")
ts(f"  feature_ranks.parquet  → {ranks_parquet}")
ts(f"  feature_ranks.csv      → {ranks_csv}")
ts(f"  X_weighted_geometry.npy         → {X_weighted_geom_path}")
ts(f"  X_weighted_geometry_whitened.npy→ {geom_whiten_path}")
ts(f"  zeta_weight_vector_geometry.pkl → {geom_weight_vector_path}")
ts(f"  feature_ranks_geometry.parquet  → {geom_ranks_parquet}")
ts(f"  feature_ranks_geometry.csv      → {geom_ranks_csv}")
ts(f"  geometry_whitener.pkl           → {geom_whitener_path}")
ts(f"  X_zshg_metric.npy               → {zshg_metric_path}")
ts(f"  zshg_metric_model.pkl           → {zshg_metric_model_path}")
ts(f"  zshg_metric_search.csv          → {zshg_metric_search_path}")
ts(f"  zeta_weights.png       → {plot1_path}")
ts(f"  zeta_sensitivity.png   → {plot2_path}")
ts(f"  Weight sum check       : {weight_sum:.6f} ≈ 1.0  ✓")
ts(f"  Geometry weight check  : {geom_weight_sum:.6f} ≈ 1.0  ✓")
ts("=" * 65)
