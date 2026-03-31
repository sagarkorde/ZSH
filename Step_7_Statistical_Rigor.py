# ============================================================
# STEP 7 — Statistical Rigour, SOTA Comparison & Ablation Study
#           CORRECTED VERSION
# ============================================================
#
# HARDWARE TARGET: HP Omen i9-13th Gen | 64 GB RAM | RTX 4060
# RAM budget: ≤ 50 GB
#
# ROOT CAUSE OF PREVIOUS FAILURES (now fixed):
# ─────────────────────────────────────────────────────────────
# PROBLEM 1 — Wrong evaluation space throughout.
#   Step 5 scored ZSH in X_w (Zeta-weighted 27D feature space)
#   → Sil=0.3562, DBI=1.0317, CHI=32,299. Step 7 used UMAP-15D
#   instead, producing Sil=−0.1418. UMAP is a topological
#   manifold — Euclidean silhouette is meaningless in it.
#   FIX: X_w is the ONLY evaluation space in this script.
#        UMAP-15D is never used for metric computation.
#
# PROBLEM 2 — Ablation Condition D collapsed to Sil=−0.466.
#   Isolation Forest was incorrectly modifying cluster labels.
#   FIX: Condition D uses IDENTICAL cluster labels as Condition C.
#        Isolation Forest adds a binary anomaly flag per row
#        (an independent output layer) — it never changes cluster IDs.
#        Condition D reports: same clustering as C + anomaly rate.
#
# PROBLEM 3 — SOTA methods fitted in UMAP-15D.
#   Rival methods (KMeans, HDBSCAN, GMM, Agglomerative) were
#   trained in UMAP-15D, which gives them different cluster
#   geometry than ZSH trained in X_w. Unfair comparison.
#   FIX: ALL methods are fitted AND evaluated in X_w space.
#
# EVALUATION SPACE RULE (enforced everywhere in this script):
#   - Training space: X_w (Zeta-weighted 27D) for all methods
#   - Scoring space:  X_w (same) for silhouette / DBI / CHI
#   - UMAP-15D:       used ONLY in Step 4 for visualisation
#
# DESIGN:
#   • Checkpointed — re-run safely, completed stages are skipped
#   • DuckDB stores all numerical results for reproducible queries
#   • Timestamped progress on every log line
#   • All figures saved at 200 DPI, publication-ready
#
# OUTPUTS:
#   fig9_bootstrap_ci.png         Bootstrap CI error-bar figure
#   fig10_sota_comparison.png     SOTA grouped bar chart
#   fig11_ablation_study.png      Ablation contribution chart
#   step7_stats_report.txt        Paper-paste-ready numbers
#   step7_sota_comparison.csv     SOTA table (CSV)
#   step7_metric_winners.csv      Best method per intrinsic metric
#   step7_ablation_study.csv      Ablation table (CSV)
#   paper_positioning_notes.txt   Discussion / abstract framing guidance
#   step7_metrics.db              DuckDB store of all results
# ============================================================

import numpy as np
import pandas as pd
import joblib, os, sys, io, logging, time, pickle, warnings, duckdb
from pathlib import Path
from datetime import datetime
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("WARNING: hdbscan not installed. HDBSCAN row will be skipped.")

# ══════════════════════════════════════════════════════════════
# 0.  CONFIGURATION — edit ONLY this block
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Statistical test parameters
N_BOOTSTRAP   = 200      # bootstrap iterations for CI estimation
N_BOOT_SAMPLE = 5_000    # rows per bootstrap draw (silhouette is O(n²) — keep ≤5K)
N_PERMUTE     = 300      # label-shuffle iterations for permutation test

# SOTA and ablation subsample sizes
N_SOTA_SAMPLE   = 10_000   # rows for SOTA scoring
ABLATION_SAMPLE = 20_000   # rows for ablation scoring

# Clustering parameters
K_FINAL  = 30     # number of clusters (must match Step 5 output)
RNG_SEED = 42     # global random seed
RUN_VERSION = "xw_norm_v7"
WARD_MICRO_CLUSTERS = 160
ELKAN_N_INIT = 20

# ══════════════════════════════════════════════════════════════
# 1.  LOGGER (UTF-8 dual-sink: console + file)
# ══════════════════════════════════════════════════════════════

log_path = os.path.join(OUTPUT_DIR, "step7_log.txt")
_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=_utf8),
        logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)
_T0 = time.time()
def ts(msg): log.info(f"[{time.time()-_T0:7.1f}s]  {msg}")

def ckpt(s):    return os.path.join(OUTPUT_DIR, f".ckpt7_{RUN_VERSION}_{s}.done")
def is_done(s): return os.path.exists(ckpt(s))
def mark_done(s):
    Path(ckpt(s)).touch()
    ts(f"  CHECKPOINT -> [{s}]")

p   = lambda name: os.path.join(OUTPUT_DIR, name)
rng = np.random.default_rng(RNG_SEED)
ZSHG_PRECOND_PATH = p('X_zshg_metric.npy')
LEGACY_GEOM_PRECOND_PATH = p('X_weighted_geometry_whitened.npy')
ZSHG_MODEL_PATH = p('zshg_metric_model.pkl')
ZSHG_MODEL = (
    joblib.load(ZSHG_MODEL_PATH)
    if os.path.exists(ZSHG_MODEL_PATH)
    else None
)
ZSHG_SELECTED_CANDIDATE = (
    str(ZSHG_MODEL.get('selected_candidate', 'unknown'))
    if isinstance(ZSHG_MODEL, dict)
    else None
)
ZSHG_IS_IDENTITY = ZSHG_SELECTED_CANDIDATE == 'identity'
if os.path.exists(ZSHG_PRECOND_PATH):
    GEOM_PRECOND_PATH = ZSHG_PRECOND_PATH
    if ZSHG_IS_IDENTITY:
        GEOM_PRECOND_LABEL = 'ZSH-G identity fallback (same geometry as X_w_norm)'
        GEOM_PRECOND_SHORT = '+Elkan'
        GEOM_PRECOND_CONDITION = 'C: + Elkan Geometry Refinement'
    else:
        GEOM_PRECOND_LABEL = 'teacher-distilled ZSH-G metric space'
        GEOM_PRECOND_SHORT = '+Metric'
        GEOM_PRECOND_CONDITION = 'C: + ZSH-G Metric Preconditioning'
elif os.path.exists(LEGACY_GEOM_PRECOND_PATH):
    GEOM_PRECOND_PATH = LEGACY_GEOM_PRECOND_PATH
    GEOM_PRECOND_LABEL = 'Laplacian-Zeta + whitening'
    GEOM_PRECOND_SHORT = '+Geom'
    GEOM_PRECOND_CONDITION = 'C: + Geometry Preconditioning'
else:
    GEOM_PRECOND_PATH = None
    GEOM_PRECOND_LABEL = None
    GEOM_PRECOND_SHORT = '+Seeds'
    GEOM_PRECOND_CONDITION = 'C: + Zeta + Rule Seeds'
HAS_GEOM_PRECOND = GEOM_PRECOND_PATH is not None

# ══════════════════════════════════════════════════════════════
# 2.  DUCKDB METRICS STORE
# ══════════════════════════════════════════════════════════════
# All numeric results are written to DuckDB immediately after
# computation — crash-safe, queryable, reproducible.

METRICS_DB = p('step7_metrics.db')
def get_con(): return duckdb.connect(METRICS_DB)

def init_db():
    con = get_con()
    con.execute("""
        CREATE TABLE IF NOT EXISTS bootstrap_results (
            metric   VARCHAR, mean DOUBLE, ci_lo DOUBLE, ci_hi DOUBLE,
            n_iter   INTEGER, n_sample INTEGER,
            eval_space VARCHAR, run_ts TIMESTAMP DEFAULT current_timestamp
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS permutation_test (
            metric VARCHAR, observed DOUBLE, null_mean DOUBLE, null_std DOUBLE,
            p_value DOUBLE, n_permute INTEGER,
            eval_space VARCHAR, run_ts TIMESTAMP DEFAULT current_timestamp
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS sota_results (
            method VARCHAR, k INTEGER,
            silhouette DOUBLE, dbi DOUBLE, chi DOUBLE,
            eval_space VARCHAR, run_ts TIMESTAMP DEFAULT current_timestamp
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS ablation_results (
            condition VARCHAR, silhouette DOUBLE, dbi DOUBLE, chi DOUBLE,
            anomaly_rate DOUBLE, eval_space VARCHAR,
            run_ts TIMESTAMP DEFAULT current_timestamp
        )""")
    for table in [
        "bootstrap_results",
        "permutation_test",
        "sota_results",
        "ablation_results",
    ]:
        con.execute(f"DELETE FROM {table}")
    con.close()
    ts("DuckDB metrics store initialised.")

ts("=" * 70)
ts("STEP 7 — Statistical Rigour (CORRECTED — X_w_norm evaluation space)")
ts("=" * 70)
if HAS_GEOM_PRECOND:
    ts(
        "Geometry-preconditioned Step 3 artifact detected: yes "
        f"({GEOM_PRECOND_LABEL}; {os.path.basename(GEOM_PRECOND_PATH)})"
    )
    if ZSHG_SELECTED_CANDIDATE is not None:
        ts(f"ZSH-G selected metric candidate: {ZSHG_SELECTED_CANDIDATE}")
else:
    ts("Geometry-preconditioned Step 3 artifact detected: no")
init_db()

# ══════════════════════════════════════════════════════════════
# 3.  LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════

ts("\nLoading artifacts …")
t0 = time.time()

# ── Resolve label file (try both naming conventions) ──────────
def find_file(*candidates):
    for c in candidates:
        if os.path.exists(c): return c
    raise FileNotFoundError("None found:\n  " + "\n  ".join(candidates))

LABELS_PATH = find_file(
    p('zsh_improved_labels.npy'),   # ZSH semantic labels (Step 5) — PREFERRED
    p('final_labels.npy'),
    p('final_labels_optimized.npy')
)
ANOM_PATH = find_file(
    p('zsh_improved_anomaly_flags.npy'),
    p('anomaly_flags.npy'),
    p('anomaly_flags_optimized.npy')
)

# Memory-mapped load — only pages needed rows into RAM
labels_mm  = np.load(LABELS_PATH, mmap_mode='r')
anom_mm    = np.load(ANOM_PATH,   mmap_mode='r')
X_w_mm     = np.load(p('X_weighted.npy'),  mmap_mode='r')
feat_cols  = joblib.load(p('feature_cols.pkl'))
df_feat    = pd.read_parquet(p('df_balanced_features.parquet'))

n_rows = min(len(labels_mm), len(anom_mm), len(X_w_mm), len(df_feat))
ts(f"  Aligned n_rows: {n_rows:,}")

labels_arr = np.array(labels_mm[:n_rows], dtype=np.int32)
anom_arr   = np.array(anom_mm[:n_rows],   dtype=np.int8)

ts("  Loading X_w (Zeta-weighted 27D) into RAM …")
X_w = np.array(X_w_mm[:n_rows], dtype=np.float32)
ts(f"  X_w shape: {X_w.shape}  | RAM≈{X_w.nbytes/1e9:.2f} GB")

ts("  Standardizing X_w once to match Step 5 evaluation geometry …")
t_norm = time.time()
X_w = StandardScaler().fit_transform(X_w).astype(np.float32)
ts(f"  X_w_norm shape: {X_w.shape}  | {time.time()-t_norm:.1f}s")

# X_raw: unweighted features from Step 2 — used only in ablation Condition A
X_raw = df_feat[feat_cols].values[:n_rows].astype(np.float32)

ts(f"  X_raw shape: {X_raw.shape}")
ts(f"  labels unique: {np.unique(labels_arr).size}  anomaly_rate: {anom_arr.mean()*100:.2f}%")

legacy_final = p('final_labels.npy')
if os.path.exists(legacy_final) and os.path.normcase(LABELS_PATH) != os.path.normcase(legacy_final):
    legacy_unique = np.unique(np.load(legacy_final, mmap_mode='r')).size
    active_unique = np.unique(labels_arr).size
    if legacy_unique != active_unique:
        ts(f"  WARNING: legacy final_labels.npy has {legacy_unique} clusters; "
           f"Step 7 will use {os.path.basename(LABELS_PATH)} with {active_unique} clusters.")

ts(f"  Total load: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════
# EVALUATION SPACE DECLARATION (printed once, referenced below)
# ══════════════════════════════════════════════════════════════
EVAL_SPACE = "X_w_norm (globally StandardScaled Zeta-weighted 27D — matches Step 5 geometry)"
ts(f"\nEVALUATION SPACE (all metrics): {EVAL_SPACE}")

# ══════════════════════════════════════════════════════════════
# HELPER: Stratified subsample index
# ══════════════════════════════════════════════════════════════
def stratified_idx(n_target, labels, seed=42):
    """
    Return ~n_target indices sampled proportionally from each cluster.
    Ensures every cluster is represented (min 2 points).
    """
    rng_s = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    fracs  = counts / counts.sum()
    target = np.maximum((fracs * n_target).astype(int), 2)
    idx    = []
    for cl, tgt in zip(unique, target):
        cl_idx = np.where(labels == cl)[0]
        chosen = rng_s.choice(cl_idx, size=min(tgt, len(cl_idx)), replace=False)
        idx.extend(chosen)
    idx = np.array(idx)
    rng_s.shuffle(idx)
    return idx[:n_target]

# Pre-compute a stratified pool for bootstrap (stays in RAM)
POOL_N   = min(200_000, n_rows)
pool_idx = stratified_idx(POOL_N, labels_arr, seed=42)
X_pool   = X_w[pool_idx]       # X_w space — consistent with Step 5 metric computation
L_pool   = labels_arr[pool_idx]
ts(f"  Bootstrap pool: {len(pool_idx):,} rows  (stratified from X_w)")

# ══════════════════════════════════════════════════════════════
# STAGE 1 — BOOTSTRAP CONFIDENCE INTERVALS (95%)
# ══════════════════════════════════════════════════════════════
# Computes the distribution of Silhouette, DBI, and CHI across
# 200 bootstrap resamples of 50,000 rows from the stratified pool.
# All metrics are computed in X_w_norm space — the SAME geometry where
# Step 5 reported Sil=0.3562, ensuring direct comparability.
# ══════════════════════════════════════════════════════════════

boot_cache = p(f'step7_bootstrap_cache_{RUN_VERSION}.pkl')

if is_done("bootstrap") and os.path.exists(boot_cache):
    ts("\nBOOTSTRAP: Loading from cache …")
    with open(boot_cache, 'rb') as f:
        boot_res = pickle.load(f)
else:
    ts(f"\nBOOTSTRAP: {N_BOOTSTRAP} iterations × {N_BOOT_SAMPLE:,} rows in {EVAL_SPACE} …")
    sil_b, dbi_b, chi_b = [], [], []
    freq = max(1, N_BOOTSTRAP // 10)

    for i in range(N_BOOTSTRAP):
        # Bootstrap resample (with replacement) from the stratified pool
        idx_b = rng.choice(len(pool_idx), size=N_BOOT_SAMPLE, replace=True)
        Xb    = X_pool[idx_b]
        Lb    = L_pool[idx_b]
        if len(np.unique(Lb)) < 2:
            continue

        # All three metrics in X_w_norm space
        sil_b.append(silhouette_score(Xb, Lb, random_state=RNG_SEED))
        dbi_b.append(davies_bouldin_score(Xb, Lb))
        chi_b.append(calinski_harabasz_score(Xb, Lb))

        if (i + 1) % freq == 0:
            ts(f"  Bootstrap {i+1}/{N_BOOTSTRAP}  "
               f"sil_mean={np.mean(sil_b):.4f}")

    boot_res = {
        'silhouette': np.array(sil_b),
        'dbi':        np.array(dbi_b),
        'chi':        np.array(chi_b),
    }
    with open(boot_cache, 'wb') as f:
        pickle.dump(boot_res, f)
    ts(f"  Bootstrap done. sil_mean={np.mean(sil_b):.4f}")
    mark_done("bootstrap")

def ci95(arr):
    return np.percentile(arr, 2.5), np.mean(arr), np.percentile(arr, 97.5)

sil_lo, sil_mu, sil_hi = ci95(boot_res['silhouette'])
dbi_lo, dbi_mu, dbi_hi = ci95(boot_res['dbi'])
chi_lo, chi_mu, chi_hi = ci95(boot_res['chi'])

ts(f"\n  Silhouette  CI95: [{sil_lo:.4f}, {sil_hi:.4f}]  mean={sil_mu:.4f}")
ts(f"  DBI         CI95: [{dbi_lo:.4f}, {dbi_hi:.4f}]  mean={dbi_mu:.4f}")
ts(f"  CHI         CI95: [{chi_lo:.0f}, {chi_hi:.0f}]  mean={chi_mu:.0f}")

# Write to DuckDB
con = get_con()
for metric, mu, lo, hi in [
    ('silhouette', sil_mu, sil_lo, sil_hi),
    ('dbi', dbi_mu, dbi_lo, dbi_hi),
    ('chi', chi_mu, chi_lo, chi_hi),
]:
    con.execute(
        "INSERT INTO bootstrap_results VALUES (?,?,?,?,?,?,?,current_timestamp)",
        [metric, mu, lo, hi, N_BOOTSTRAP, N_BOOT_SAMPLE, EVAL_SPACE]
    )
con.close()

# ══════════════════════════════════════════════════════════════
# STAGE 2 — PERMUTATION TEST (cluster structure significance)
# ══════════════════════════════════════════════════════════════
# H₀: the cluster assignment is no better than random labels.
# Method: compute silhouette on actual labels, then on 500
# random permutations of the same labels.
# p-value = fraction of permutations that score HIGHER than observed.
# Critical: BOTH observed and null are computed in X_w_norm space.
# ══════════════════════════════════════════════════════════════

perm_cache = p(f'step7_permutation_cache_{RUN_VERSION}.pkl')

if is_done("permutation") and os.path.exists(perm_cache):
    ts("\nPERMUTATION: Loading from cache …")
    with open(perm_cache, 'rb') as f:
        perm_data = pickle.load(f)
    obs_sil   = perm_data['obs_sil']
    perm_null = perm_data['null_sil']
else:
    ts(f"\nPERMUTATION: {N_PERMUTE} shuffles in {EVAL_SPACE} …")

    # Use a fixed subsample for speed — same indices every permutation
    perm_idx = stratified_idx(N_BOOT_SAMPLE, labels_arr, seed=99)
    X_perm   = X_w[perm_idx]    # X_w_norm space — consistent with Step 5
    L_perm   = labels_arr[perm_idx]

    obs_sil  = silhouette_score(X_perm, L_perm, random_state=RNG_SEED)
    ts(f"  Observed Sil (X_w_norm): {obs_sil:.4f}")

    null_sil = []
    freq_p   = max(1, N_PERMUTE // 5)
    for i in range(N_PERMUTE):
        L_shuffled = rng.permutation(L_perm)
        null_sil.append(silhouette_score(X_perm, L_shuffled,
                                          random_state=RNG_SEED))
        if (i + 1) % freq_p == 0:
            ts(f"  Permutation {i+1}/{N_PERMUTE}  "
               f"null_mean={np.mean(null_sil):.4f}")

    perm_null  = np.array(null_sil)
    perm_data  = {'obs_sil': obs_sil, 'null_sil': perm_null}
    with open(perm_cache, 'wb') as f:
        pickle.dump(perm_data, f)
    mark_done("permutation")

# p-value: fraction of null scores ≥ observed
p_val_sil = (perm_null >= obs_sil).mean()
ts(f"\n  Observed Sil = {obs_sil:.4f}")
ts(f"  Null mean    = {perm_null.mean():.4f}  ±{perm_null.std():.4f}")
ts(f"  p-value      = {p_val_sil:.4f}  "
   f"({'< 0.001' if p_val_sil < 0.001 else f'{p_val_sil:.4f}'})")

# Write to DuckDB
con = get_con()
con.execute(
    "INSERT INTO permutation_test VALUES (?,?,?,?,?,?,?,current_timestamp)",
    ['silhouette', obs_sil, perm_null.mean(), perm_null.std(),
     p_val_sil, N_PERMUTE, EVAL_SPACE]
)
con.close()

# ══════════════════════════════════════════════════════════════
# STAGE 3 — SOTA COMPARISON
# ══════════════════════════════════════════════════════════════
# All rival methods are fitted AND scored in X_w_norm space.
# ZSH labels come from Step 5 (pre-computed), also scored in X_w_norm.
# This is the only valid apples-to-apples comparison.
#
# Methods compared:
#   ZSH (ours)      — Step 5 labels, scored in X_w_norm
#   Vanilla KMeans  — k=30, fitted on X_w_norm, scored in X_w_norm
#   HDBSCAN         — fitted on X_w_norm subsample, scored in X_w_norm
#   GMM             — k=30, fitted on X_w_norm, scored in X_w_norm
#   Agglomerative   — k=30, fitted on X_w_norm subsample, scored in X_w_norm
# ══════════════════════════════════════════════════════════════

sota_cache = p(f'step7_sota_cache_{RUN_VERSION}.pkl')

if is_done("sota") and os.path.exists(sota_cache):
    ts("\nSOTA: Loading from cache …")
    with open(sota_cache, 'rb') as f:
        sota_results = pickle.load(f)
else:
    ts(f"\nSOTA: Fitting all methods in {EVAL_SPACE} …")
    ts(f"  Subsample: {N_SOTA_SAMPLE:,} rows")

    # Fixed stratified subsample — same for all methods
    sota_idx = stratified_idx(N_SOTA_SAMPLE, labels_arr, seed=77)
    X_sota   = X_w[sota_idx]    # X_w_norm — same space for every method
    L_zsh    = labels_arr[sota_idx]

    rows = []

    def score_row(name, labels, k, X=X_sota):
        """Compute all 3 metrics in X_w_norm space."""
        if len(np.unique(labels)) < 2:
            return {'method': name, 'k': k,
                    'silhouette': np.nan, 'dbi': np.nan, 'chi': np.nan}
        return {
            'method':     name,
            'k':          k,
            'silhouette': silhouette_score(X, labels, random_state=RNG_SEED),
            'dbi':        davies_bouldin_score(X, labels),
            'chi':        calinski_harabasz_score(X, labels),
        }

    # ── ZSH (Step 5 labels) ──────────────────────────────────
    ts("  Scoring ZSH labels …")
    t_m = time.time()
    row = score_row('ZSH (Ours, k=30)', L_zsh, K_FINAL)
    rows.append(row)
    ts(f"  ZSH  Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
       f"CHI={row['chi']:.0f}  ({time.time()-t_m:.1f}s)")

    # ── Vanilla KMeans k=30 on X_w_norm ──────────────────────
    ts("  Fitting Vanilla KMeans k=30 on X_w_norm …")
    t_m = time.time()
    km = MiniBatchKMeans(
        n_clusters=K_FINAL, init='k-means++',
        n_init=10, batch_size=20_000, random_state=RNG_SEED
    )
    L_km = km.fit_predict(X_sota)   # fit AND predict on X_w_norm subsample
    row  = score_row('Vanilla KMeans (k=30)', L_km, K_FINAL)
    rows.append(row)
    ts(f"  KM   Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
       f"CHI={row['chi']:.0f}  ({time.time()-t_m:.1f}s)")

    # ── Full KMeans++ Elkan k=30 on X_w_norm ─────────────────
    ts("  Fitting KMeans++ Elkan k=30 on X_w_norm …")
    t_m = time.time()
    km_elkan = KMeans(
        n_clusters=K_FINAL, init='k-means++', n_init=ELKAN_N_INIT,
        max_iter=250, algorithm='elkan', random_state=RNG_SEED
    )
    L_km_elkan = km_elkan.fit_predict(X_sota)
    row = score_row('KMeans++ Elkan (k=30)', L_km_elkan, K_FINAL)
    rows.append(row)
    ts(f"  KME  Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
       f"CHI={row['chi']:.0f}  ({time.time()-t_m:.1f}s)")

    # ── HDBSCAN on X_w_norm ──────────────────────────────────
    if HAS_HDBSCAN:
        ts("  Fitting HDBSCAN on X_w_norm …")
        t_m = time.time()
        mcs = max(200, N_SOTA_SAMPLE // 200)   # adaptive min_cluster_size
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=20,
            cluster_selection_method='eom', core_dist_n_jobs=-1
        )
        L_hdb = hdb.fit_predict(X_sota)
        # Replace noise (-1) with nearest valid cluster for scoring
        if (L_hdb == -1).any():
            from sklearn.neighbors import NearestNeighbors
            valid_mask = L_hdb != -1
            noise_mask = ~valid_mask
            if valid_mask.sum() > 0:
                nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
                nn.fit(X_sota[valid_mask])
                _, nn_idx = nn.kneighbors(X_sota[noise_mask])
                L_hdb[noise_mask] = L_hdb[valid_mask][nn_idx.flatten()]
        n_hdb = len(np.unique(L_hdb))
        noise_rate = (hdb.labels_ == -1).mean() * 100
        ts(f"  HDBSCAN done  clusters={n_hdb}  noise={noise_rate:.1f}%  "
           f"({time.time()-t_m:.1f}s)")
        row = score_row('HDBSCAN', L_hdb, n_hdb)
        rows.append(row)
        ts(f"  HDBSCAN  Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
           f"CHI={row['chi']:.0f}")
    else:
        ts("  HDBSCAN skipped (not installed).")

    # ── GMM k=30 on X_w_norm ─────────────────────────────────
    ts("  Fitting GMM k=30 on X_w_norm …")
    t_m = time.time()
    X_sota_gmm = X_sota.astype(np.float64, copy=False)
    try:
        gmm = GaussianMixture(
            n_components=K_FINAL, covariance_type='diag',
            max_iter=200, reg_covar=1e-5, random_state=RNG_SEED
        )
        L_gmm = gmm.fit_predict(X_sota_gmm)
    except ValueError:
        ts("  GMM initial fit was numerically unstable; retrying with stronger regularization …")
        gmm = GaussianMixture(
            n_components=K_FINAL, covariance_type='diag',
            max_iter=200, reg_covar=1e-4, random_state=RNG_SEED
        )
        L_gmm = gmm.fit_predict(X_sota_gmm)
    row   = score_row('GMM (k=30)', L_gmm, K_FINAL)
    rows.append(row)
    ts(f"  GMM  Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
       f"CHI={row['chi']:.0f}  ({time.time()-t_m:.1f}s)")

    # ── Ward-guided hybrid k=30 on X_w_norm ──────────────────
    ts("  Fitting Ward-guided hybrid k=30 on X_w_norm …")
    t_m = time.time()
    micro = MiniBatchKMeans(
        n_clusters=min(max(K_FINAL, WARD_MICRO_CLUSTERS), len(X_sota)),
        init='k-means++', n_init=5, batch_size=20_000,
        max_iter=200, random_state=RNG_SEED
    )
    micro.fit(X_sota)
    micro_groups = AgglomerativeClustering(
        n_clusters=K_FINAL, linkage='ward'
    ).fit_predict(micro.cluster_centers_)
    ward_init = np.vstack(
        [micro.cluster_centers_[micro_groups == k].mean(axis=0) for k in range(K_FINAL)]
    )
    ward_hybrid = KMeans(
        n_clusters=K_FINAL, init=ward_init, n_init=1,
        max_iter=200, algorithm='elkan', random_state=RNG_SEED
    )
    L_ward_hybrid = ward_hybrid.fit_predict(X_sota)
    row = score_row('Ward-Guided Hybrid (k=30)', L_ward_hybrid, K_FINAL)
    rows.append(row)
    ts(f"  WGH  Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
       f"CHI={row['chi']:.0f}  ({time.time()-t_m:.1f}s)")

    # ── Agglomerative k=30 on X_w_norm ───────────────────────
    ts("  Fitting Agglomerative k=30 on X_w_norm …")
    t_m = time.time()
    agg = AgglomerativeClustering(n_clusters=K_FINAL, linkage='ward')
    L_agg = agg.fit_predict(X_sota)
    row   = score_row('Agglomerative (k=30)', L_agg, K_FINAL)
    rows.append(row)
    ts(f"  Agg  Sil={row['silhouette']:.4f}  DBI={row['dbi']:.4f}  "
       f"CHI={row['chi']:.0f}  ({time.time()-t_m:.1f}s)")

    sota_results = pd.DataFrame(rows)
    with open(sota_cache, 'wb') as f:
        pickle.dump(sota_results, f)

    # Write to DuckDB
    con = get_con()
    for _, row_s in sota_results.iterrows():
        con.execute(
            "INSERT INTO sota_results VALUES (?,?,?,?,?,?,current_timestamp)",
            [row_s['method'], int(row_s['k']), float(row_s['silhouette']),
             float(row_s['dbi']), float(row_s['chi']), EVAL_SPACE]
        )
    con.close()

    sota_results.to_csv(p('step7_sota_comparison.csv'), index=False)
    ts("\n  SOTA table saved.")
    mark_done("sota")

metric_winners = pd.DataFrame(
    [
        {
            'metric': 'silhouette',
            'winner_method': sota_results.loc[sota_results['silhouette'].idxmax(), 'method'],
            'winner_value': float(sota_results['silhouette'].max()),
            'direction': 'higher',
        },
        {
            'metric': 'dbi',
            'winner_method': sota_results.loc[sota_results['dbi'].idxmin(), 'method'],
            'winner_value': float(sota_results['dbi'].min()),
            'direction': 'lower',
        },
        {
            'metric': 'chi',
            'winner_method': sota_results.loc[sota_results['chi'].idxmax(), 'method'],
            'winner_value': float(sota_results['chi'].max()),
            'direction': 'higher',
        },
    ]
)
metric_winners.to_csv(p('step7_metric_winners.csv'), index=False)

ts("\n  SOTA results:")
ts("\n" + sota_results.to_string(index=False))
ts("\n  Metric-specific winners:")
ts("\n" + metric_winners.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# STAGE 4 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════
# Tests four conditions to isolate each ZSH component.
# If a geometry-first Step 3 artifact exists, Condition C becomes the
# pure-geometry preconditioning branch (either ZSH-G metric or legacy
# Laplacian-Zeta whitening).
# Otherwise we fall back to the legacy rule-seeded condition.
#
#   Condition A: Vanilla KMeans k=30 on X_raw_norm (no zeta, no seeds)
#                ← Baseline: no ZSH components active
#   Condition B: KMeans k=30 on X_w_norm (Zeta weighting only)
#                ← Shows: what zeta weighting contributes alone
#   Condition C: either
#                (a) KMeans++ Elkan on geometry-preconditioned space, or
#                (b) KMeans k=30 on X_w_norm with rule-seeded centroids
#                ← Shows: what the main Step 5 optimisation layer adds on top of zeta
#   Condition D: Condition C labels + Isolation Forest anomaly flag
#                CRITICAL FIX: Condition D uses IDENTICAL cluster
#                labels as Condition C. IF only adds a binary
#                anomaly flag (a separate output) — it NEVER
#                changes cluster IDs. Silhouette of D = sil of C.
#                Condition D additionally reports anomaly_rate=5%.
#
# ALL conditions scored in X_w_norm space.
# ══════════════════════════════════════════════════════════════

ablation_cache = p(f'step7_ablation_cache_{RUN_VERSION}.pkl')

def normalize_ablation_results(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ablation condition labels reviewer-safe across cache versions."""
    df = df.copy()
    if 'condition' in df.columns:
        df['condition'] = df['condition'].replace({
            'D: Full ZSH (Condition C + Isolation Forest flag)':
                'D: Condition C + Isolation Forest flag (separate output)',
        })
    return df

if is_done("ablation") and os.path.exists(ablation_cache):
    ts("\nABLATION: Loading from cache …")
    with open(ablation_cache, 'rb') as f:
        ablation_results = normalize_ablation_results(pickle.load(f))
    with open(ablation_cache, 'wb') as f:
        pickle.dump(ablation_results, f)
else:
    ts(f"\nABLATION: 4 conditions, n={ABLATION_SAMPLE:,} rows, eval in {EVAL_SPACE} …")

    # Fixed stratified subsample for all 4 conditions
    abl_idx = stratified_idx(ABLATION_SAMPLE, labels_arr, seed=55)
    X_abl_w = X_w[abl_idx]       # ← scoring/training space (X_w_norm)
    # Re-standardize the non-zeta baseline on its own feature space so KMeans
    # is not dominated by raw-scale artifacts in Condition A.
    X_abl_r = StandardScaler().fit_transform(X_raw[abl_idx]).astype(np.float32)
    X_abl_geom = None
    if HAS_GEOM_PRECOND:
        X_geom_mm = np.load(GEOM_PRECOND_PATH, mmap_mode='r')
        X_abl_geom = np.array(X_geom_mm[abl_idx], dtype=np.float32)
    L_abl   = labels_arr[abl_idx]
    A_abl   = anom_arr[abl_idx]

    def abl_score(name, labels, anom_rate=None, X_eval=X_abl_w):
        """Score a condition in X_w_norm space."""
        n_unique = len(np.unique(labels))
        sil = silhouette_score(X_eval, labels, random_state=RNG_SEED)
        dbi = davies_bouldin_score(X_eval, labels)
        chi = calinski_harabasz_score(X_eval, labels)
        ts(f"  {name:<45}  Sil={sil:.4f}  DBI={dbi:.4f}  CHI={chi:.0f}")
        return {'condition': name, 'silhouette': sil, 'dbi': dbi,
                'chi': chi, 'anomaly_rate': anom_rate}

    abl_rows = []

    # ── Condition A: Vanilla KMeans on X_raw_norm ────────────
    ts("\n  Condition A — Vanilla KMeans (no zeta, no seeds) on X_raw_norm …")
    t_a = time.time()
    km_a = MiniBatchKMeans(
        n_clusters=K_FINAL, init='k-means++',
        n_init=10, batch_size=20_000, random_state=RNG_SEED
    )
    L_a = km_a.fit_predict(X_abl_r)
    # Score Condition A in X_w_norm (same space as all other conditions)
    abl_rows.append(abl_score(
        'A: Vanilla KMeans (no zeta, no seeds)', L_a, anom_rate=None
    ))
    ts(f"    ({time.time()-t_a:.1f}s)")

    # ── Condition B: KMeans on X_w_norm (zeta-weighted, no seeds) ─
    ts("\n  Condition B — KMeans on Zeta-weighted X_w_norm (no seeds) …")
    t_b = time.time()
    km_b = MiniBatchKMeans(
        n_clusters=K_FINAL, init='k-means++',
        n_init=10, batch_size=20_000, random_state=RNG_SEED
    )
    L_b = km_b.fit_predict(X_abl_w)
    abl_rows.append(abl_score(
        'B: + Zeta Weighting (no seeds)', L_b, anom_rate=None
    ))
    ts(f"    ({time.time()-t_b:.1f}s)")

    if HAS_GEOM_PRECOND:
        ts(f"\n  Condition C — Geometry preconditioning ({GEOM_PRECOND_LABEL}) …")
        t_c = time.time()
        km_c = KMeans(
            n_clusters=K_FINAL,
            init='k-means++',
            n_init=ELKAN_N_INIT,
            max_iter=250,
            algorithm='elkan',
            random_state=RNG_SEED,
        )
        L_c = km_c.fit_predict(X_abl_geom)
        abl_rows.append(abl_score(
            GEOM_PRECOND_CONDITION, L_c, anom_rate=None
        ))
        ts(f"    ({time.time()-t_c:.1f}s)")
    else:
        # ── Condition C: KMeans on X_w_norm with rule-seeded centroids ─
        # Rule-seeded init: compute centroids from boolean flag means,
        # then use as init='array' for KMeans (same as Step 5 approach)
        ts("\n  Condition C — Zeta + Rule Seeds …")
        t_c = time.time()

        RULE_FLAGS = {
            'Coinjoin_Mixer': 'is_coinjoin_like',
            'Batch_Payment':  'is_batch_payment',
            'Consolidation':  'is_consolidation',
            'Distribution':   'is_distribution',
            'Standard_P2P':   'is_peer_to_peer',
        }
        available_flags = {k: v for k, v in RULE_FLAGS.items()
                           if v in df_feat.columns}
        ts(f"    Rule flags used: {list(available_flags.keys())}")

        seed_centroids = []
        for rule_name, flag_col in available_flags.items():
            flag_vals = df_feat[flag_col].values[:n_rows].astype(float)
            mask = flag_vals[abl_idx] > 0.5
            if mask.sum() >= 2:
                seed_centroids.append(X_abl_w[mask].mean(axis=0))
                ts(f"    {rule_name}: {mask.sum():,} seed points")

        # Fill remaining centroids with KMeans++ if fewer than K_FINAL seeds
        n_seeds = len(seed_centroids)
        if 2 <= n_seeds < K_FINAL:
            km_fill = MiniBatchKMeans(
                n_clusters=K_FINAL - n_seeds, init='k-means++',
                n_init=5, batch_size=10_000, random_state=RNG_SEED
            )
            km_fill.fit(X_abl_w)
            seed_centroids.extend(list(km_fill.cluster_centers_))

        init_arr = np.array(seed_centroids[:K_FINAL], dtype=np.float32)
        km_c = MiniBatchKMeans(
            n_clusters=K_FINAL, init=init_arr,
            n_init=1, batch_size=20_000, random_state=RNG_SEED
        )
        L_c = km_c.fit_predict(X_abl_w)
        abl_rows.append(abl_score(
            'C: + Zeta + Rule Seeds', L_c, anom_rate=None
        ))
        ts(f"    ({time.time()-t_c:.1f}s)")

    # ── Condition D: Condition C + Isolation Forest anomaly flag ─
    # CRITICAL: Isolation Forest does NOT change cluster labels.
    # It adds a per-transaction binary anomaly flag as a SEPARATE output.
    # The clustering quality (Sil/DBI/CHI) is IDENTICAL to Condition C.
    # The additional value of Condition D = anomaly detection at 5.00%.
    ts("\n  Condition D — Condition C + IF anomaly flag (separate output) …")
    anomaly_rate_d = float(A_abl.mean() * 100)
    abl_rows.append({
        'condition'    : 'D: Condition C + Isolation Forest flag (separate output)',
        'silhouette'   : abl_rows[-1]['silhouette'],  # identical to C
        'dbi'          : abl_rows[-1]['dbi'],          # identical to C
        'chi'          : abl_rows[-1]['chi'],           # identical to C
        'anomaly_rate' : anomaly_rate_d
    })
    ts(f"    Cluster scores = Condition C  "
       f"|  Anomaly detection rate: {anomaly_rate_d:.2f}%")
    ts(f"    (IF is an independent output layer — cluster IDs are unchanged)")

    ablation_results = normalize_ablation_results(pd.DataFrame(abl_rows))
    with open(ablation_cache, 'wb') as f:
        pickle.dump(ablation_results, f)

    # Write to DuckDB
    con = get_con()
    for _, row_a in ablation_results.iterrows():
        con.execute(
            "INSERT INTO ablation_results VALUES (?,?,?,?,?,?,current_timestamp)",
            [row_a['condition'], float(row_a['silhouette']),
             float(row_a['dbi']), float(row_a['chi']),
             float(row_a['anomaly_rate']) if row_a['anomaly_rate'] else None,
             EVAL_SPACE]
        )
    con.close()

    mark_done("ablation")

ablation_results.to_csv(p('step7_ablation_study.csv'), index=False)
ts("\n  Ablation table saved.")
ts("\n  Ablation results:")
ts("\n" + ablation_results.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# STAGE 5 — FIGURES
# ══════════════════════════════════════════════════════════════

PALETTE = {
    'zsh':    '#2196F3',
    'km':     '#FF5722',
    'hdb':    '#9C27B0',
    'gmm':    '#4CAF50',
    'agg':    '#FF9800',
    'gray':   '#90A4AE',
    'A': '#9E9E9E', 'B': '#64B5F6', 'C': '#1976D2', 'D': '#0D47A1',
}

# ── FIG 9: Bootstrap CI Error Bar ─────────────────────────────
fig9_path = p('fig9_bootstrap_ci.png')
if not is_done("fig9"):
    ts("\nFIG9: Bootstrap CI figure …")
    t0 = time.time()

    metrics_info = [
        ('silhouette', 'Silhouette Score\n(Higher is Better)',
         sil_mu, sil_lo, sil_hi, '#2196F3'),
        ('dbi', 'Davies-Bouldin Index\n(Lower is Better)',
         dbi_mu, dbi_lo, dbi_hi, '#FF5722'),
        ('chi', 'Calinski-Harabasz Index\n(Higher is Better)',
         chi_mu, chi_lo, chi_hi, '#4CAF50'),
    ]

    fig9, axes9 = plt.subplots(1, 3, figsize=(15, 6))

    for ax, (m, title, mu, lo, hi, col) in zip(axes9, metrics_info):
        # Distribution histogram
        arr = boot_res[m]
        ax.hist(arr, bins=30, color=col, alpha=0.6, edgecolor='white',
                linewidth=0.5, zorder=2)
        ax.axvline(mu, color='black', linewidth=2, label=f'Mean={mu:.4f}')
        ax.axvline(lo, color='red', linewidth=1.5, linestyle='--',
                   label=f'95% CI [{lo:.4f}, {hi:.4f}]')
        ax.axvline(hi, color='red', linewidth=1.5, linestyle='--')
        # Shade CI
        ax.axvspan(lo, hi, alpha=0.15, color='red', zorder=1)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Metric value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, zorder=0)

    fig9.suptitle(
        f'Bootstrap Confidence Intervals (95%)\n'
        f'{N_BOOTSTRAP} iterations × {N_BOOT_SAMPLE:,} rows '
        f'| Evaluation space: {EVAL_SPACE}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(fig9_path, dpi=200, bbox_inches='tight')
    plt.close(fig9)
    ts(f"  Saved fig9 ({time.time()-t0:.1f}s)")
    mark_done("fig9")

# ── FIG 10: SOTA Comparison ───────────────────────────────────
fig10_path = p('fig10_sota_comparison.png')
if not is_done("fig10"):
    ts("\nFIG10: SOTA comparison …")
    t0 = time.time()

    methods = sota_results['method'].tolist()
    x       = np.arange(len(methods))
    width   = 0.25

    # Colour: ZSH in blue, rivals in muted grey/orange
    def mcolor(m, good_col, bad_col='#90A4AE'):
        return good_col if 'ZSH' in m else bad_col

    fig10, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Silhouette + DBI
    sil_vals = sota_results['silhouette'].values
    dbi_vals = sota_results['dbi'].values
    sil_cols = [PALETTE['zsh'] if 'ZSH' in m else '#B0BEC5' for m in methods]
    dbi_cols = [PALETTE['km']  if 'ZSH' in m else '#CFD8DC' for m in methods]

    b_sil = ax_l.bar(x - width/2, sil_vals, width, color=sil_cols,
                      label='Silhouette (↑)', zorder=3)
    b_dbi = ax_l.bar(x + width/2, dbi_vals, width, color=dbi_cols,
                      label='DBI (↓)', alpha=0.85, zorder=3)

    for bar in b_sil:
        h = bar.get_height()
        ax_l.text(bar.get_x()+bar.get_width()/2,
                  h + 0.01 if h >= 0 else h - 0.05,
                  f'{h:.3f}', ha='center', va='bottom', fontsize=7,
                  fontweight='bold' if h == max(sil_vals) else 'normal')
    for bar in b_dbi:
        h = bar.get_height()
        ax_l.text(bar.get_x()+bar.get_width()/2, h + 0.05,
                  f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax_l.axhline(0, color='black', linewidth=0.5)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax_l.set_ylabel('Score', fontsize=10)
    ax_l.set_title('Silhouette Score & Davies-Bouldin Index\n'
                   '(All methods trained & evaluated in X_w_norm space)',
                   fontsize=10, fontweight='bold')
    ax_l.legend(fontsize=9)
    ax_l.grid(axis='y', alpha=0.3, zorder=0)

    # Right: Calinski-Harabasz
    chi_vals = sota_results['chi'].values
    chi_cols = [PALETTE['zsh'] if 'ZSH' in m else '#90A4AE' for m in methods]
    b_chi    = ax_r.bar(x, chi_vals, 0.5, color=chi_cols, zorder=3)
    chi_max  = max(chi_vals)
    for bar, val in zip(b_chi, chi_vals):
        label = f'{val/1000:.1f}K' if val >= 1000 else f'{val:.0f}'
        ax_r.text(bar.get_x()+bar.get_width()/2,
                  val + chi_max*0.01, label,
                  ha='center', va='bottom', fontsize=7,
                  fontweight='bold' if val == chi_max else 'normal')

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax_r.set_ylabel('Calinski-Harabasz Index (↑ better)', fontsize=10)
    ax_r.set_title('Calinski-Harabasz Index\n'
                   '(Higher = more compact, better-separated clusters)',
                   fontsize=10, fontweight='bold')
    ax_r.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f'{v/1000:.0f}K' if v>=1000 else f'{v:.0f}')
    )
    ax_r.grid(axis='y', alpha=0.3, zorder=0)

    fig10.suptitle(
        'ZSH vs SOTA Clustering Methods — Intrinsic Quality\n'
        f'Evaluation space: {EVAL_SPACE} | n={N_SOTA_SAMPLE:,}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(fig10_path, dpi=200, bbox_inches='tight')
    plt.close(fig10)
    ts(f"  Saved fig10 ({time.time()-t0:.1f}s)")
    mark_done("fig10")

# ── FIG 11: Ablation Study ────────────────────────────────────
fig11_path = p('fig11_ablation_study.png')
if not is_done("fig11"):
    ts("\nFIG11: Ablation study figure …")
    t0 = time.time()

    df_abl   = ablation_results.copy()
    conds    = df_abl['condition'].tolist()
    short = (
        ['A\n(Baseline)', 'B\n(+Zeta)', f'C\n({GEOM_PRECOND_SHORT})', 'D\n(+IF flag)']
        if HAS_GEOM_PRECOND
        else ['A\n(Baseline)', 'B\n(+Zeta)', 'C\n(+Seeds)', 'D\n(+IF flag)']
    )
    x        = np.arange(len(conds))
    abl_cols = [PALETTE['A'], PALETTE['B'], PALETTE['C'], PALETTE['D']]

    base_sil = df_abl.iloc[0]['silhouette']
    base_dbi = df_abl.iloc[0]['dbi']
    base_chi = df_abl.iloc[0]['chi']

    fig11, axes11 = plt.subplots(1, 3, figsize=(16, 6))

    metric_cfg = [
        ('silhouette', 'Silhouette Score (↑ better)', base_sil, True),
        ('dbi',        'Davies-Bouldin (↓ better)',   base_dbi, False),
        ('chi',        'Calinski-Harabasz (↑ better)',base_chi, True),
    ]

    for ax, (col, title, base_val, higher_better) in zip(axes11, metric_cfg):
        vals = df_abl[col].values
        bars = ax.bar(x, vals, color=abl_cols, width=0.5, zorder=3)

        for bar, val, cond in zip(bars, vals, conds):
            delta = (val - base_val) / abs(base_val) * 100
            if not higher_better:
                delta = -delta
            sign  = '+' if delta >= 0 else ''
            color = 'darkgreen' if delta > 0 else ('red' if delta < -5 else 'gray')
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + abs(max(vals))*0.015,
                    f'{sign}{delta:.1f}%',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        if col == 'chi':
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v,_: f'{v/1000:.0f}K' if v>=1000 else f'{v:.0f}'))

    # Add the anomaly rate annotation for Condition D
    d_rate = df_abl.iloc[3]['anomaly_rate']
    if d_rate is not None:
        axes11[0].annotate(
            f'D also adds\n{d_rate:.1f}% anomaly detection',
            xy=(3, df_abl.iloc[3]['silhouette']),
            xytext=(2.2, df_abl.iloc[3]['silhouette'] * 0.85),
            fontsize=7, color='navy',
            arrowprops=dict(arrowstyle='->', color='navy', lw=0.8)
        )

    # Legend below
    legend_lines = [
        'A: Vanilla KMeans on X_raw_norm (no zeta, no seeds) — baseline',
        'B: KMeans on X_w_norm (Zeta weighting only)',
        (
            f'C: KMeans++ Elkan on {GEOM_PRECOND_LABEL} '
            '(scored in X_w_norm)'
            if HAS_GEOM_PRECOND
            else 'C: KMeans on X_w_norm with rule-seeded centroids'
        ),
        'D: Condition C labels + Isolation Forest anomaly flag (separate output; cluster IDs unchanged)',
    ]
    fig11.text(0.5, -0.04, '\n'.join(legend_lines), ha='center', va='top',
               fontsize=8, family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E1', alpha=0.9))

    fig11.suptitle(
        'Ablation Study — Contribution of Each ZSH Component\n'
        f'n={ABLATION_SAMPLE:,} | All scored in {EVAL_SPACE} | % change vs Condition A',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(fig11_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig11)
    ts(f"  Saved fig11 ({time.time()-t0:.1f}s)")
    mark_done("fig11")

# ══════════════════════════════════════════════════════════════
# STAGE 6 — PAPER-READY STATISTICS REPORT
# ══════════════════════════════════════════════════════════════

report_path = p('step7_stats_report.txt')
positioning_path = p('paper_positioning_notes.txt')
ts("\nWriting paper-ready statistics report …")

def pct_chg(new, old): return (new - old) / abs(old) * 100
def dbi_pct(new, old): return (old - new) / abs(old) * 100   # lower=better

# Ablation deltas
a_row = ablation_results.iloc[0]
b_row = ablation_results.iloc[1]
c_row = ablation_results.iloc[2]
d_row = ablation_results.iloc[3]
anom_count = int(anom_arr.sum())
n_profiles = int(np.unique(labels_arr).size)
if HAS_GEOM_PRECOND:
    if ZSHG_IS_IDENTITY:
        ablation_component_name = "Elkan geometry refinement"
        ablation_component_note = (
            "Condition C uses the ZSH-G identity fallback, so it is equivalent to "
            "KMeans++ Elkan on X_w_norm rather than a nontrivial learned transform."
        )
        ablation_component_snippet = (
            f"Relative to zeta weighting alone, replacing MiniBatch KMeans with full "
            f"KMeans++ Elkan changes the clustering geometry by Sil "
            f"{pct_chg(c_row.silhouette, b_row.silhouette):+.1f}%, "
            f"DBI {dbi_pct(c_row.dbi, b_row.dbi):+.1f}%, and "
            f"CHI {pct_chg(c_row.chi, b_row.chi):+.1f}%."
        )
    else:
        ablation_component_name = (
            "ZSH-G metric preconditioning"
            if os.path.normcase(GEOM_PRECOND_PATH) == os.path.normcase(ZSHG_PRECOND_PATH)
            else "Geometry preconditioning"
        )
        ablation_component_note = (
            f"Condition C trains in the {GEOM_PRECOND_LABEL} but is scored in X_w_norm."
        )
        ablation_component_snippet = (
            f"Relative to zeta weighting alone, {ablation_component_name.lower()} changes the clustering "
            f"geometry by Sil {pct_chg(c_row.silhouette, b_row.silhouette):+.1f}%, "
            f"DBI {dbi_pct(c_row.dbi, b_row.dbi):+.1f}%, and "
            f"CHI {pct_chg(c_row.chi, b_row.chi):+.1f}%."
        )
else:
    ablation_component_name = "Rule seeds"
    ablation_component_note = ""
    ablation_component_snippet = (
        f"Relative to zeta weighting alone, rule-seeded initialisation changes the clustering "
        f"geometry by Sil {pct_chg(c_row.silhouette, b_row.silhouette):+.1f}%, "
        f"DBI {dbi_pct(c_row.dbi, b_row.dbi):+.1f}%, and "
        f"CHI {pct_chg(c_row.chi, b_row.chi):+.1f}%."
    )
zsh_row = sota_results.loc[sota_results['method'] == 'ZSH (Ours, k=30)'].iloc[0]
sil_winner_row = sota_results.loc[sota_results['silhouette'].idxmax()]
dbi_winner_row = sota_results.loc[sota_results['dbi'].idxmin()]
chi_winner_row = sota_results.loc[sota_results['chi'].idxmax()]
sig_str = '< 0.001' if p_val_sil < 0.001 else f'{p_val_sil:.4f}'
winner_methods = set(metric_winners['winner_method'].tolist())
if len(winner_methods) == 1:
    winner_summary = (
        f"{sil_winner_row['method']} attains the best values on all three intrinsic metrics"
    )
else:
    winner_summary = (
        f"{sil_winner_row['method']} achieves the highest Silhouette, "
        f"{dbi_winner_row['method']} the lowest DBI, and "
        f"{chi_winner_row['method']} the highest CHI"
    )

primary_cmp_path = p('zsh_improved_comparison.csv')
primary_claim_line = (
    "Primary corrected baseline comparison file was not found, so baseline deltas are omitted."
)
if os.path.exists(primary_cmp_path):
    try:
        primary_cmp = pd.read_csv(primary_cmp_path).set_index('Metric')
        primary_ref_name = "baseline reference"
        if 'Baseline_Method' in primary_cmp.columns and primary_cmp['Baseline_Method'].notna().any():
            primary_ref_name = str(primary_cmp['Baseline_Method'].dropna().iloc[0])
        sil_imp_primary = float(primary_cmp.loc['Silhouette', 'Improvement_Pct'])
        dbi_imp_primary = float(primary_cmp.loc['Davies_Bouldin', 'Improvement_Pct'])
        chi_imp_primary = float(primary_cmp.loc['Calinski_Harabasz', 'Improvement_Pct'])
        sil_better = bool(primary_cmp.loc['Silhouette', 'ZSH_Better'])
        dbi_better = bool(primary_cmp.loc['Davies_Bouldin', 'ZSH_Better'])
        chi_better = bool(primary_cmp.loc['Calinski_Harabasz', 'ZSH_Better'])
        wins_primary = int(sil_better) + int(dbi_better) + int(chi_better)
        if wins_primary == 3:
            primary_claim_line = (
                f"Against {primary_ref_name} in the same X_w_norm space, "
                f"ZSH improves Silhouette by {sil_imp_primary:+.1f}%, DBI by {dbi_imp_primary:+.1f}%, "
                f"and CHI by {chi_imp_primary:+.1f}%."
            )
        else:
            primary_claim_line = (
                f"Against {primary_ref_name} in the same X_w_norm space, ZSH is mixed on intrinsic "
                f"geometry: Silhouette {sil_imp_primary:+.1f}%, DBI {dbi_imp_primary:+.1f}%, "
                f"and CHI {chi_imp_primary:+.1f}% relative change versus the reference."
            )
    except Exception:
        pass

safe_claim = (
    "ZSH should be positioned as a statistically validated blockchain transaction "
    "profiling framework that combines geometric clustering, semantic interpretability, "
    "and anomaly detection, rather than as the single best geometry-only optimizer."
)
avoid_claim = (
    "Avoid claiming that ZSH is the best overall intrinsic clustering method across all "
    "SOTA baselines or that it dominates KMeans++ Elkan / Agglomerative on Silhouette, DBI, and CHI."
)
discussion_text = (
    f"While {sil_winner_row['method']} achieves the strongest intrinsic geometric scores "
    f"on the current same-space SOTA table, it yields unlabeled clusters optimized only for "
    f"Euclidean compactness. In contrast, ZSH produces {n_profiles} semantically labeled "
    f"transaction profiles and flags {anom_count:,} anomalous transactions ({anom_arr.mean()*100:.2f}%), "
    "making the output directly actionable for blockchain forensic analysis. Accordingly, ZSH "
    "should be interpreted as a profiling and forensic-support framework, not as a pure "
    "geometry-only clustering competitor."
)
fig10_caption = (
    f"Figure 10 compares ZSH with geometry-only baselines in the same standardized Zeta-weighted "
    f"space. Although {winner_summary} "
    f"(Sil winner={sil_winner_row['silhouette']:.4f}; DBI winner={dbi_winner_row['dbi']:.4f}; "
    f"CHI winner={chi_winner_row['chi']:.1f}), "
    "ZSH remains competitive while uniquely providing semantic profile labels and anomaly flags."
)
limitations_text = (
    f"A limitation of the current study is that ZSH does not achieve the top intrinsic "
    f"geometry scores in the broader same-space SOTA comparison: ZSH records "
    f"Sil={zsh_row['silhouette']:.4f}, DBI={zsh_row['dbi']:.4f}, and CHI={zsh_row['chi']:.1f}, "
    f"whereas {winner_summary}. The contribution of ZSH therefore lies in interpretability, "
    "domain alignment, and anomaly-aware profiling rather than absolute dominance on intrinsic geometry alone."
)
abstract_snippet = (
    f"We present ZSH, a blockchain transaction profiling framework that clusters "
    f"{n_rows:,} transactions into {n_profiles} semantically interpretable profiles and flags "
    f"{anom_count:,} anomalous cases. Across {N_BOOTSTRAP} bootstrap iterations, ZSH achieves "
    f"Silhouette={sil_mu:.4f} (95% CI [{sil_lo:.4f}, {sil_hi:.4f}]) with permutation-test "
    f"significance p {sig_str}. {primary_claim_line} Unlike geometry-only baselines, ZSH outputs "
    "forensically meaningful profile labels and anomaly indicators suitable for downstream blockchain analysis."
)
umap_note = (
    "UMAP is used for visualization only in the corrected pipeline. The choice of init='random' "
    "was made to avoid spectral-initialization failures on disconnected transaction graphs and "
    "does not affect the Step 5 / Step 7 clustering metrics, which are computed directly in X_w_norm."
)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("ZSH CLUSTERING — STATISTICAL RIGOR REPORT (CORRECTED)\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Evaluation space: {EVAL_SPACE}\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. BOOTSTRAP CONFIDENCE INTERVALS (95%)\n")
    f.write(f"   Evaluation space : {EVAL_SPACE}\n")
    f.write(f"   Iterations       : {N_BOOTSTRAP}\n")
    f.write(f"   Sample per iter  : {N_BOOT_SAMPLE:,}\n\n")
    f.write(f"   Silhouette        : {sil_mu:.4f}  95% CI [{sil_lo:.4f}, {sil_hi:.4f}]\n")
    f.write(f"   Davies-Bouldin    : {dbi_mu:.4f}  95% CI [{dbi_lo:.4f}, {dbi_hi:.4f}]\n")
    f.write(f"   Calinski-Harabasz : {chi_mu:.1f}   95% CI [{chi_lo:.1f}, {chi_hi:.1f}]\n\n")

    f.write("2. PERMUTATION TEST (Silhouette, H0: labels = random)\n")
    f.write(f"   Evaluation space : {EVAL_SPACE}\n")
    f.write(f"   Permutations     : {N_PERMUTE}\n")
    f.write(f"   Observed Sil     : {obs_sil:.4f}\n")
    f.write(f"   Null mean Sil    : {perm_null.mean():.4f}  ± {perm_null.std():.4f}\n")
    sig_str = '< 0.001' if p_val_sil < 0.001 else f'{p_val_sil:.4f}'
    f.write(f"   p-value          : {sig_str}  "
            f"({'SIGNIFICANT' if p_val_sil < 0.05 else 'NOT significant'} at α=0.05)\n\n")

    f.write("3. SOTA COMPARISON\n")
    f.write(f"   Evaluation space : {EVAL_SPACE}\n")
    f.write(f"   Subsample        : {N_SOTA_SAMPLE:,}\n")
    f.write(f"   All rival methods trained AND evaluated in same space as ZSH.\n\n")
    f.write(sota_results.to_string(index=False))
    f.write("\n\n")

    f.write("3b. METRIC-SPECIFIC WINNERS\n")
    f.write(metric_winners.to_string(index=False))
    f.write("\n\n")

    f.write("4. ABLATION STUDY\n")
    f.write(f"   Evaluation space : {EVAL_SPACE}\n")
    f.write(f"   Subsample        : {ABLATION_SAMPLE:,}\n")
    f.write(f"   NOTE: Condition D cluster labels are IDENTICAL to Condition C.\n")
    if ablation_component_note:
        f.write(f"   NOTE: {ablation_component_note}\n")
    f.write(f"   Isolation Forest adds an independent binary anomaly flag only.\n\n")
    f.write(ablation_results.to_string(index=False))
    f.write("\n\n")

    f.write("4b. COMPONENT CONTRIBUTIONS\n")
    f.write(f"   Zeta weighting (B vs A):   "
            f"Sil {pct_chg(b_row.silhouette, a_row.silhouette):+.1f}%  "
            f"DBI {dbi_pct(b_row.dbi, a_row.dbi):+.1f}%  "
            f"CHI {pct_chg(b_row.chi, a_row.chi):+.1f}%\n")
    f.write(f"   {ablation_component_name} (C vs B):       "
            f"Sil {pct_chg(c_row.silhouette, b_row.silhouette):+.1f}%  "
            f"DBI {dbi_pct(c_row.dbi, b_row.dbi):+.1f}%  "
            f"CHI {pct_chg(c_row.chi, b_row.chi):+.1f}%\n")
    f.write(f"   Condition C + anomaly flag vs baseline (D vs A): "
            f"Sil {pct_chg(d_row.silhouette, a_row.silhouette):+.1f}%  "
            f"DBI {dbi_pct(d_row.dbi, a_row.dbi):+.1f}%  "
            f"CHI {pct_chg(d_row.chi, a_row.chi):+.1f}%  "
            f"+ anomaly detection {d_row.anomaly_rate:.2f}%\n\n")

    f.write("5. PAPER-PASTE SNIPPETS\n\n")
    f.write(f"""Bootstrap CIs (Section 4 — Experimental Results):
  "The ZSH hybrid method achieves a Silhouette score of {sil_mu:.4f} (95% CI
   [{sil_lo:.4f}, {sil_hi:.4f}]), a Davies-Bouldin index of {dbi_mu:.4f} (95% CI
   [{dbi_lo:.4f}, {dbi_hi:.4f}]), and a Calinski-Harabasz index of {chi_mu:.0f}
   (95% CI [{chi_lo:.0f}, {chi_hi:.0f}]), computed over {N_BOOTSTRAP} bootstrap
   iterations of {N_BOOT_SAMPLE:,}-row stratified subsamples in the
   standardized Zeta-weighted feature space."

Permutation test (Section 4 — Statistical Significance):
  "The null hypothesis that cluster assignments are indistinguishable
   from random labelling is rejected with p {sig_str} (permutation
   test, n={N_PERMUTE} shuffles; observed Silhouette = {obs_sil:.4f};
   null mean = {perm_null.mean():.4f} ± {perm_null.std():.4f})."

Ablation (Section 4 — Component Analysis):
  "An ablation study (Table X) demonstrates that Zeta weighting
   contributes {pct_chg(b_row.silhouette, a_row.silhouette):+.1f}% Silhouette
   and {pct_chg(b_row.chi, a_row.chi):+.1f}% Calinski-Harabasz improvement
   over vanilla K-Means. {ablation_component_snippet}
   Isolation Forest provides independent anomaly detection at a
   {d_row.anomaly_rate:.2f}% rate without altering cluster structure."
""")

    f.write("\n6. POSITIONING / FRAMING GUIDANCE\n")
    f.write(f"   Safe claim     : {safe_claim}\n")
    f.write(f"   Avoid claim    : {avoid_claim}\n")
    f.write(f"   Baseline hook  : {primary_claim_line}\n\n")
    f.write(f"   UMAP note      : {umap_note}\n\n")
    f.write("   Discussion-ready paragraph:\n")
    f.write(f"   \"{discussion_text}\"\n\n")
    f.write("   Figure 10 caption rewrite:\n")
    f.write(f"   \"{fig10_caption}\"\n\n")
    f.write("   Limitations paragraph:\n")
    f.write(f"   \"{limitations_text}\"\n\n")
    f.write("   Abstract-ready paragraph:\n")
    f.write(f"   \"{abstract_snippet}\"\n")

with open(positioning_path, 'w', encoding='utf-8') as f:
    f.write("ZSH PAPER POSITIONING NOTES\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")
    f.write("HONEST VERDICT\n")
    f.write("ZSH is publishable, but the paper must be positioned as a profiling / interpretability framework, not as the best geometry-only clustering method.\n\n")
    f.write("SAFE CLAIM\n")
    f.write(safe_claim + "\n\n")
    f.write("DO NOT CLAIM\n")
    f.write(avoid_claim + "\n\n")
    f.write("BASELINE HOOK\n")
    f.write(primary_claim_line + "\n\n")
    f.write("ABSTRACT-READY PARAGRAPH\n")
    f.write(abstract_snippet + "\n\n")
    f.write("DISCUSSION PARAGRAPH\n")
    f.write(discussion_text + "\n\n")
    f.write("UMAP METHODS NOTE\n")
    f.write(umap_note + "\n\n")
    f.write("FIGURE 10 CAPTION REWRITE\n")
    f.write(fig10_caption + "\n\n")
    f.write("LIMITATIONS PARAGRAPH\n")
    f.write(limitations_text + "\n")

ts(f"  Report saved: {report_path}")
ts(f"  Positioning notes saved: {positioning_path}")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
ts("\n" + "=" * 70)
ts("STEP 7 COMPLETE")
ts(f"  Total runtime: {time.time()-_T0:.1f}s")
ts(f"  Evaluation space: {EVAL_SPACE}")
ts(f"\n  Bootstrap:    Sil={sil_mu:.4f} [{sil_lo:.4f},{sil_hi:.4f}]  "
   f"DBI={dbi_mu:.4f}  CHI={chi_mu:.0f}")
ts(f"  Permutation:  p-value = {sig_str}  "
   f"(obs={obs_sil:.4f}  null={perm_null.mean():.4f}±{perm_null.std():.4f})")
ts(f"\n  Outputs:")
for fname in ['fig9_bootstrap_ci.png', 'fig10_sota_comparison.png',
              'fig11_ablation_study.png', 'step7_stats_report.txt',
              'paper_positioning_notes.txt',
              'step7_sota_comparison.csv', 'step7_metric_winners.csv',
              'step7_ablation_study.csv',
              'step7_metrics.db']:
    fp = p(fname)
    ts(f"  {'OK' if os.path.exists(fp) else 'MISSING':6} {fp}")
ts("=" * 70)
ts(f"RESUMABILITY: delete .ckpt7_{RUN_VERSION}_*.done files to force any stage to re-run.")
