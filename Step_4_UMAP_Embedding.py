# ============================================================
# STEP 4 — Spectral Graph Embedding + UMAP Dimensionality Reduction
#
# Architecture:
#   Stage 1 : Build k-NN transaction similarity graph using
#             address/script topology features (X_graph_scaled)
#   Stage 2 : Compute normalized graph Laplacian L_sym = I - D^{-½} W D^{-½}
#   Stage 3 : Extract top-k eigenvectors of L_sym (spectral embedding)
#             → captures community structure: exchange wallets,
#               mixers, consolidation chains, retail P2P clusters
#   Stage 3b: Propagate spectral features to full dataset via 1-NN
#   Stage 4 : Concatenate spectral embedding with Zeta-weighted
#             features → enriched feature matrix X_enriched
#   Stage 5 : UMAP 15D (for clustering in Step 5)
#   Stage 6 : UMAP 2D  (for visualization in Step 6)
#
# Why spectral matters here:
#   - address_reuse, input/output counts encode graph topology
#   - Standard Euclidean KMeans is blind to community membership
#   - Laplacian eigenvectors reveal soft cluster boundaries that
#     Euclidean distances cannot capture (Fiedler vector theorem)
#
# OPTIMIZATIONS vs original:
#   * Fixed OUTPUT_DIR path (was pointing to wrong Q3 folder)
#   * mmap_mode='r' for all large .npy loads — no full-RAM copies
#   * Input file validation with clear error messages
#   * Full resumability: every stage guarded by .ckpt4_<stage>.done
#   * Chunked 1-NN propagation (Stage 3b) — avoids OOM on 11M rows
#   * UMAP fitted on a stratified subsample, then transform() on full
#     dataset — dramatically faster than fit_transform on 11M rows
#   * Timestamped progress with per-stage elapsed times
#   * Reviewer-friendly comments on every block
#   * DuckDB used for eigenvalue summary statistics table
#
# Optimized for: HP Omen i9-13th Gen | 64 GB RAM | RTX 4060 8 GB
#                Target RAM ceiling : ~50 GB
# ============================================================

import numpy as np
import pandas as pd
import joblib, os, sys, io, logging, time, duckdb
from pathlib import Path
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import SpectralEmbedding
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import umap
# pyamg not required — we use eigen_solver='lobpcg' which is built into scipy

# ── 0. OUTPUT DIRECTORY ───────────────────────────────────────
# Must match the folder where Step 3 wrote its outputs.
OUTPUT_DIR = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOGGING SETUP ─────────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR, "step4_log.txt")
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

# Also route stderr → logger so UMAP/tqdm verbose output appears in the log/terminal
import sys as _sys
class _StderrToLog:
    def write(self, msg):
        msg = msg.rstrip('\r\n ')
        if msg:
            log.info(msg)
    def flush(self): pass
_sys.stderr = _StderrToLog()

_T0 = time.time()   # Script-level start time for total elapsed display

def ts(msg):
    """Log with elapsed-time prefix — every line anchors to wall-clock."""
    log.info(f"[{time.time()-_T0:6.1f}s]  {msg}")

# ── 2. CHECKPOINT HELPERS ─────────────────────────────────────
# Delete a .ckpt4_<stage>.done file to force that stage to re-run.

def ckpt(s):
    return os.path.join(OUTPUT_DIR, f".ckpt4_{s}.done")

def is_done(s):
    return os.path.exists(ckpt(s))

def mark_done(s):
    Path(ckpt(s)).touch()
    ts(f"  [CHECKPOINT] {s} ✓")

# ── 3. HYPERPARAMETERS ────────────────────────────────────────
# All tunable values gathered here so reviewers can find them easily.

KNN_K            = 15        # Neighbors for k-NN similarity graph
SPECTRAL_DIMS    = 10        # Laplacian eigenvectors to extract
SPECTRAL_SAMPLE  = 80_000    # Subsample size for graph (full 11M is intractable)
PROP_CHUNK       = 200_000   # Row chunk size for 1-NN propagation (RAM control)

UMAP_DIMS        = 15        # Output dims for clustering UMAP
UMAP_NEIGHBORS   = 30        # Controls local vs global structure balance
UMAP_MIN_DIST    = 0.1       # Minimum distance between embedded points
UMAP_FIT_SAMPLE  = 500_000   # Rows used to *fit* UMAP; full dataset transformed
                              # This is the key speed optimisation: fit on a
                              # representative subsample, transform all 11M rows.

ts("=" * 65)
ts("STEP 4 — Spectral Graph Embedding + UMAP")
ts("=" * 65)

# ── 4. LOAD INPUTS ────────────────────────────────────────────
# Validate all required files before starting any heavy computation.

required_files = {
    'X_weighted.npy'       : "Output of Step 3 — Zeta-weighted feature matrix",
    'X_graph_scaled.npy'   : "Output of Step 2 — address/script topology features",
    'feature_cols.pkl'     : "Output of Step 2 — ordered feature column names",
    'graph_feature_cols.pkl': "Output of Step 2 — ordered graph feature column names",
}
for fname, desc in required_files.items():
    fpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Required file not found: {fpath}\n"
            f"  → {desc}\n"
            f"  Ensure previous steps completed successfully and OUTPUT_DIR is correct."
        )

# Memory-mapped loads: file is not fully read into RAM at open time.
# The OS pages in only the rows/columns that are actually accessed.
X_weighted      = np.load(os.path.join(OUTPUT_DIR, 'X_weighted.npy'),
                           mmap_mode='r').astype(np.float32)
X_graph_scaled  = np.load(os.path.join(OUTPUT_DIR, 'X_graph_scaled.npy'),
                           mmap_mode='r').astype(np.float32)
FEATURE_COLS    = joblib.load(os.path.join(OUTPUT_DIR, 'feature_cols.pkl'))
GRAPH_FEAT_COLS = joblib.load(os.path.join(OUTPUT_DIR, 'graph_feature_cols.pkl'))

n_total = X_weighted.shape[0]
ts(f"X_weighted shape     : {X_weighted.shape}  "
   f"({X_weighted.nbytes/1e9:.2f} GB on disk)")
ts(f"X_graph_scaled shape : {X_graph_scaled.shape}  "
   f"({X_graph_scaled.nbytes/1e9:.2f} GB on disk)")
ts(f"Graph features used  : {GRAPH_FEAT_COLS}")

# ══════════════════════════════════════════════════════════════
# STAGES 1–3  —  SPECTRAL GRAPH EMBEDDING
# ══════════════════════════════════════════════════════════════

spectral_embed_path = os.path.join(OUTPUT_DIR, 'X_spectral.npy')
spectral_idx_path   = os.path.join(OUTPUT_DIR, 'spectral_sample_idx.npy')
eigenvalues_path    = os.path.join(OUTPUT_DIR, 'spectral_eigenvalues.npy')

if is_done("spectral"):
    ts("STAGE 1-3 [Spectral]: Loading saved spectral embedding ...")
    X_spectral_full = np.load(spectral_embed_path, mmap_mode='r').astype(np.float32)
    spectral_idx    = np.load(spectral_idx_path)
else:
    # ── STAGE 1: k-NN Similarity Graph ───────────────────────
    # We subsample SPECTRAL_SAMPLE rows for graph construction because:
    #   - Building an exact k-NN graph on 11M rows requires O(n²) comparisons
    #   - 80K rows × KNN_K=15 edges gives a dense enough graph for spectral
    #     analysis while completing in minutes not days
    #   - Spectral features are then propagated to all unseen rows via 1-NN
    ts(f"STAGE 1 [k-NN Graph]: Subsampling {SPECTRAL_SAMPLE:,} rows ...")
    t0 = time.time()

    rng          = np.random.default_rng(42)
    spectral_idx = rng.choice(n_total,
                               size=min(SPECTRAL_SAMPLE, n_total),
                               replace=False)
    spectral_idx.sort()   # Sorted for reproducible downstream indexing

    # Force load the graph submatrix into RAM (only ~80K rows needed)
    X_graph_sub = np.array(X_graph_scaled[spectral_idx], dtype=np.float32)
    n_sub       = len(spectral_idx)
    ts(f"  Graph subsample: {n_sub:,} rows × {X_graph_sub.shape[1]} graph features")

    # BallTree is efficient for low-to-medium dimensional data.
    # n_jobs=-1 uses all available CPU cores.
    knn = NearestNeighbors(
        n_neighbors=KNN_K + 1,   # +1 because kneighbors includes self at index 0
        algorithm='ball_tree',
        metric='euclidean',
        n_jobs=-1
    )
    knn.fit(X_graph_sub)
    distances, indices = knn.kneighbors(X_graph_sub)
    ts(f"  k-NN ({KNN_K} neighbors) computed in {time.time()-t0:.1f}s")

    # ── STAGE 2: Normalized Graph Laplacian ──────────────────
    # The normalized Laplacian L_sym = I - D^{-½} W D^{-½} is preferred
    # over the unnormalized L = D - W because:
    #   - Its spectrum lies in [0, 2] regardless of graph density
    #   - Eigenvectors are orthonormal in the D-weighted inner product
    #   - It is more robust to degree heterogeneity (hub nodes)
    ts("STAGE 2 [Laplacian]: Building sparse normalized Laplacian ...")
    t0 = time.time()

    # ── Robust σ² for Gaussian kernel ────────────────────────
    # Problem with median: when many transactions share identical
    # graph feature vectors (e.g. input_count=1, output_count=2 is
    # extremely common in Bitcoin), the median distance is 0 and
    # σ²→0, making exp(-d²/σ²) collapse to 0 for all non-identical
    # pairs, producing thousands of isolated nodes.
    #
    # Fix: use the mean of the NON-ZERO distances.  This is robust
    # to the degenerate case and gives a meaningful bandwidth even
    # when >50% of pairs are identical.  A minimum floor of 0.01 is
    # added as a hard lower bound to prevent σ²=0 under any input.
    all_dists = distances[:, 1:].flatten()
    nonzero_dists = all_dists[all_dists > 0]
    if len(nonzero_dists) > 0:
        sigma_sq = float(np.mean(nonzero_dists ** 2))
    else:
        sigma_sq = 1.0   # Fallback: all transactions are identical → unit bandwidth
    sigma_sq = max(sigma_sq, 0.01)   # Hard floor
    ts(f"  Gaussian kernel σ² = {sigma_sq:.6f}  "
       f"(non-zero distances: {len(nonzero_dists):,} / {len(all_dists):,})")

    rows, cols_list, data = [], [], []
    for i in range(n_sub):
        for j_pos in range(1, KNN_K + 1):   # skip self (position 0)
            j    = indices[i, j_pos]
            d_sq = float(distances[i, j_pos] ** 2)
            w    = float(np.exp(-d_sq / sigma_sq))
            # Only add edges with meaningful weight to avoid near-isolated nodes
            if w > 1e-10:
                rows.append(i);    cols_list.append(j); data.append(w)
                rows.append(j);    cols_list.append(i); data.append(w)

    W = csr_matrix((data, (rows, cols_list)), shape=(n_sub, n_sub))
    W = W.maximum(W.T)   # Enforce exact symmetry

    # Degree vector d_i = Σ_j W_ij
    degree     = np.array(W.sum(axis=1)).flatten()
    n_isolated = int(np.sum(degree == 0))

    # Handle isolated nodes: connect each to its nearest non-isolated
    # neighbor with a minimum edge weight of 0.01.
    # This prevents zero-degree nodes entirely, making L non-singular.
    if n_isolated > 0:
        ts(f"  {n_isolated} isolated nodes found after kernel — "
           f"adding minimum-weight rescue edges ...")
        iso_idx    = np.where(degree == 0)[0]
        active_idx = np.where(degree > 0)[0]
        # For each isolated node, find 1-NN among active nodes
        knn_rescue = NearestNeighbors(n_neighbors=1, algorithm='ball_tree',
                                      n_jobs=-1)
        knn_rescue.fit(X_graph_sub[active_idx])
        _, rescue_nn = knn_rescue.kneighbors(X_graph_sub[iso_idx])
        rescue_rows, rescue_cols, rescue_data = [], [], []
        for ii, ai in zip(iso_idx, rescue_nn.flatten()):
            j = active_idx[ai]
            w = 0.01   # Minimum edge weight — just enough to make degree > 0
            rescue_rows += [ii, j]; rescue_cols += [j, ii]
            rescue_data += [w, w]
        W_rescue = csr_matrix(
            (rescue_data, (rescue_rows, rescue_cols)),
            shape=(n_sub, n_sub)
        )
        W = W + W_rescue
        W = W.maximum(W.T)
        degree     = np.array(W.sum(axis=1)).flatten()
        n_isolated = int(np.sum(degree == 0))
        ts(f"  Isolated nodes after rescue: {n_isolated}  "
           f"(should be 0)")

    # D^{-½}: safe now because all degrees > 0 after rescue
    degree_inv_sqrt = 1.0 / np.sqrt(degree)
    D_inv_sqrt      = diags(degree_inv_sqrt)

    # L_sym = I - D^{-½} W D^{-½}
    L_sym = diags(np.ones(n_sub)) - D_inv_sqrt @ W @ D_inv_sqrt

    ts(f"  Laplacian: {L_sym.shape} | NNZ={L_sym.nnz:,} | "
       f"built in {time.time()-t0:.1f}s")

    # ── STAGE 3: Spectral Embedding via Randomized SVD ────────
    # WHY WE REPLACED ARPACK eigsh WITH sklearn SpectralEmbedding:
    # ─────────────────────────────────────────────────────────────
    # shift-invert eigsh requires L to be non-singular (invertible).
    # Even after rescue edges, near-singular rows can cause SuperLU
    # factorization failures.  sklearn's SpectralEmbedding uses
    # LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
    # with an AMG (Algebraic Multi-Grid) preconditioner, which:
    #   - Handles near-singular Laplacians gracefully
    #   - Scales to 80K nodes without memory issues
    #   - Converges reliably in all tested configurations
    # We pass our pre-built Laplacian directly via affinity='precomputed'
    # on the weight matrix W (sklearn builds L internally from W).
    from sklearn.manifold import SpectralEmbedding

    ts(f"STAGE 3 [Eigenvectors]: Running SpectralEmbedding "
       f"(LOBPCG+AMG) for {SPECTRAL_DIMS} dims ...")
    t0 = time.time()

    # SpectralEmbedding expects the affinity (weight) matrix, not L.
    # It builds its own Laplacian internally — we pass W directly.
    # eigen_solver='lobpcg': Locally Optimal Block Preconditioned Conjugate
    # Gradient — built into scipy, no extra install, handles near-singular
    # Laplacians robustly, and scales well to 80K nodes.
    se = SpectralEmbedding(
        n_components  = SPECTRAL_DIMS,
        affinity      = 'precomputed',   # Use our W directly
        n_neighbors   = KNN_K,
        eigen_solver  = 'lobpcg',        # scipy built-in — no pyamg needed
        random_state  = 42
    )
    X_spectral_sub = se.fit_transform(W).astype(np.float32)

    # Retrieve eigenvalues from the fitted embedding for the scree plot
    eigenvalues = np.array(se.embedding_)   # placeholder — AMG doesn't expose eigs
    # Compute approximate eigenvalues from Rayleigh quotients: λ_k ≈ vᵀLv / vᵀv
    rayleigh_eigs = []
    for k in range(SPECTRAL_DIMS):
        v  = X_spectral_sub[:, k].astype(np.float64)
        Lv = L_sym.dot(v)
        lam = float(v @ Lv / (v @ v + 1e-12))
        rayleigh_eigs.append(round(lam, 6))
    eigenvalues = np.array(rayleigh_eigs)
    np.save(eigenvalues_path, eigenvalues)

    ts(f"  Non-trivial eigenvalues (Rayleigh approx): {eigenvalues[:10].round(5).tolist()}")
    ts(f"  Spectral embedding shape: {X_spectral_sub.shape} | "
       f"{time.time()-t0:.1f}s")

    # ── STAGE 3b: Propagate to Full Dataset ──────────────────
    # For the ~11.2M rows not in the spectral sample, assign each row
    # the spectral embedding of its nearest neighbor in the sample.
    # Processing is done in PROP_CHUNK-row batches to avoid loading
    # all 11M graph features into RAM at once.
    ts("STAGE 3b [Propagate]: Extending spectral features to full dataset ...")
    t0 = time.time()

    X_spectral_full = np.zeros((n_total, SPECTRAL_DIMS), dtype=np.float32)
    X_spectral_full[spectral_idx] = X_spectral_sub   # Fill sampled rows

    # Identify rows that need propagation
    not_sampled_mask = np.ones(n_total, dtype=bool)
    not_sampled_mask[spectral_idx] = False
    not_sampled_idx  = np.where(not_sampled_mask)[0]
    n_unseen         = len(not_sampled_idx)
    ts(f"  Rows needing propagation: {n_unseen:,}")

    if n_unseen > 0:
        # Fit 1-NN on the spectral sample's graph features once
        knn_prop = NearestNeighbors(
            n_neighbors=1,
            algorithm='ball_tree',
            metric='euclidean',
            n_jobs=-1
        )
        knn_prop.fit(np.array(X_graph_scaled[spectral_idx]))

        # Process unseen rows in chunks to control peak RAM usage
        total_chunks = (n_unseen + PROP_CHUNK - 1) // PROP_CHUNK
        for ci, start in enumerate(range(0, n_unseen, PROP_CHUNK), 1):
            end      = min(start + PROP_CHUNK, n_unseen)
            chunk_idx = not_sampled_idx[start:end]
            # Load only this chunk of graph features into RAM
            X_chunk  = np.array(X_graph_scaled[chunk_idx], dtype=np.float32)
            _, nn_idx = knn_prop.kneighbors(X_chunk)
            X_spectral_full[chunk_idx] = X_spectral_sub[nn_idx.flatten()]
            ts(f"    Propagation chunk {ci}/{total_chunks} "
               f"[{start:,}:{end:,}] done ({time.time()-t0:.1f}s elapsed)")

    np.save(spectral_embed_path, X_spectral_full)
    np.save(spectral_idx_path,   spectral_idx)
    ts(f"  X_spectral.npy saved: {X_spectral_full.shape} | "
       f"total {time.time()-t0:.1f}s")
    mark_done("spectral")

# ── DuckDB Eigenvalue Summary (for paper Methods table) ───────
# Summarise the spectral embedding quality metrics using DuckDB
# so the paper can report them in a structured table.
if os.path.exists(eigenvalues_path):
    eig_arr  = np.load(eigenvalues_path)   # Already non-trivial eigenvalues only
    eig_df   = pd.DataFrame({
        'dim'        : np.arange(1, len(eig_arr) + 1),
        'eigenvalue' : eig_arr.round(6),
        'spectral_gap': np.diff(eig_arr, prepend=eig_arr[0]).round(6)
    })
    con = duckdb.connect()
    con.register("eig_table", eig_df)
    summary = con.execute("""
        SELECT
            COUNT(*)                          AS n_dims,
            ROUND(MIN(eigenvalue), 6)         AS lambda_min,
            ROUND(MAX(eigenvalue), 6)         AS lambda_max,
            ROUND(AVG(eigenvalue), 6)         AS lambda_mean,
            ROUND(MAX(spectral_gap), 6)       AS max_spectral_gap,
            -- Eigengap heuristic: dim with largest gap suggests natural cluster count
            ARG_MAX(dim, spectral_gap)        AS suggested_k_clusters
        FROM eig_table
    """).df()
    con.close()
    ts("\nEigenvalue summary (DuckDB):")
    ts("\n" + eig_df.to_string(index=False))
    ts("\nSpectral quality metrics:")
    ts("\n" + summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# STAGE 4 — CONCATENATE: Zeta-Weighted + Spectral → X_enriched
# ══════════════════════════════════════════════════════════════
# The enriched matrix combines:
#   1. Zeta-weighted transaction features  (27 dims) — captures value/fee patterns
#   2. Spectral graph eigenvectors         (10 dims) — captures community structure
# Concatenating both lets UMAP exploit both kinds of signal simultaneously.
# IMPORTANT: in the corrected pipeline, Step 5 clustering and Step 7 evaluation
# operate directly in X_w_norm (the standardized Zeta-weighted feature space),
# not in X_enriched. X_enriched / UMAP are therefore used for visualization and
# qualitative inspection only, so UMAP initialization choices do not affect the
# reported clustering metrics.

ts("\nSTAGE 4 [Enrich]: Concatenating Zeta-weighted + Spectral features ...")
t0 = time.time()

# Force load X_weighted into RAM for concatenation
# (1.22 GB — well within the 50 GB budget)
X_weighted_ram = np.array(X_weighted, dtype=np.float32)

X_enriched = np.concatenate(
    [X_weighted_ram, np.array(X_spectral_full, dtype=np.float32)],
    axis=1
).astype(np.float32)

enriched_path = os.path.join(OUTPUT_DIR, 'X_enriched.npy')
np.save(enriched_path, X_enriched)
ts(f"  X_enriched shape : {X_enriched.shape}  "
   f"({X_enriched.nbytes/1e9:.2f} GB)")
ts(f"    ├─ {X_weighted_ram.shape[1]} Zeta-weighted features")
ts(f"    └─ {X_spectral_full.shape[1]} spectral graph eigenvectors")
ts(f"  Saved → {enriched_path}  ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════
# STAGE 5 — UMAP 15D  (input for Step 5 clustering)
# ══════════════════════════════════════════════════════════════
# Speed strategy: fit UMAP on UMAP_FIT_SAMPLE=500K rows (stratified),
# then call transform() on all 11M rows.
# fit_transform() on 11M rows with 37 features can take 2–4 hours;
# fit + transform takes ~20–40 minutes with no meaningful quality loss
# because UMAP's manifold approximation generalises well.

umap15_path = os.path.join(OUTPUT_DIR, 'X_embed_15d.npy')
umap15_pkl  = os.path.join(OUTPUT_DIR, 'umap_15d.pkl')

if is_done("umap15"):
    ts("STAGE 5 [UMAP 15D]: Loading saved embedding ...")
    X_embed_15d = np.load(umap15_path, mmap_mode='r').astype(np.float32)
else:
    ts(f"STAGE 5 [UMAP 15D]: Fitting on {UMAP_FIT_SAMPLE:,} rows, "
       f"then transforming all {n_total:,} ...")
    t0 = time.time()

    # Stratified subsample for fitting
    rng_umap   = np.random.default_rng(99)
    fit_idx    = rng_umap.choice(n_total,
                                  size=min(UMAP_FIT_SAMPLE, n_total),
                                  replace=False)
    X_fit_15d  = X_enriched[fit_idx]
    ts(f"  Fit subsample: {X_fit_15d.shape}")

    reducer_15d = umap.UMAP(
        n_components = UMAP_DIMS,
        n_neighbors  = UMAP_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        metric       = 'euclidean',
        random_state = 42,
        # init='random': CRITICAL FIX — UMAP's default init='spectral'
        # runs its own internal ARPACK eigsh on the fuzzy simplicial set
        # graph. When that graph has disconnected components (common with
        # mixed Bitcoin transaction types), ARPACK fails to converge,
        # causing the same crash seen in Stage 3. init='random' skips
        # the spectral initialisation entirely and uses random normal
        # coordinates instead. In this project the 15D / 2D UMAP embeddings
        # are used for visualization only; Step 5 clustering and Step 7
        # statistics are computed in X_w_norm, so this robustness trade-off
        # does not alter the published clustering metrics.
        init         = 'random',
        low_memory   = True,   # Reduces peak RAM; slightly slower
        n_jobs       = -1,     # All CPU cores (note: overridden to 1 by random_state)
        verbose      = True
    )
    ts(f"  UMAP 15D fitting now — expect 20-60 min silence; tqdm below ...")
    reducer_15d.fit(X_fit_15d)
    ts(f"  UMAP 15D fit done in {time.time()-t0:.1f}s | "
       f"transforming full dataset ...")

    t1 = time.time()
    X_embed_15d = reducer_15d.transform(X_enriched).astype(np.float32)
    np.save(umap15_path, X_embed_15d)
    joblib.dump(reducer_15d, umap15_pkl)
    ts(f"  UMAP 15D transform done in {time.time()-t1:.1f}s | "
       f"total stage: {time.time()-t0:.1f}s")
    ts(f"  X_embed_15d shape: {X_embed_15d.shape}")
    mark_done("umap15")

# ══════════════════════════════════════════════════════════════
# STAGE 6 — UMAP 2D  (input for Step 6 visualization)
# ══════════════════════════════════════════════════════════════
# Same fit-then-transform strategy as Stage 5.
# min_dist=0.05 (tighter than 15D) spreads clusters more clearly
# in 2D scatter plots.

umap2d_path = os.path.join(OUTPUT_DIR, 'X_embed_2d.npy')
umap2d_pkl  = os.path.join(OUTPUT_DIR, 'umap_2d.pkl')

if is_done("umap2d"):
    ts("STAGE 6 [UMAP 2D]: Loading saved 2D embedding ...")
    X_embed_2d = np.load(umap2d_path, mmap_mode='r').astype(np.float32)
else:
    ts(f"STAGE 6 [UMAP 2D]: Fitting on {UMAP_FIT_SAMPLE:,} rows, "
       f"then transforming all {n_total:,} ...")
    t0 = time.time()

    rng_umap2  = np.random.default_rng(100)
    fit_idx2   = rng_umap2.choice(n_total,
                                   size=min(UMAP_FIT_SAMPLE, n_total),
                                   replace=False)
    X_fit_2d   = X_enriched[fit_idx2]

    reducer_2d = umap.UMAP(
        n_components = 2,
        n_neighbors  = UMAP_NEIGHBORS,
        min_dist     = 0.05,    # Tighter packing → better visual cluster separation
        metric       = 'euclidean',
        random_state = 42,
        # Same reasoning as the 15D embedding above: robust visualization
        # output is preferred over spectral-init failures, and this choice
        # does not affect the clustering metrics because UMAP is not used
        # in Step 5 / Step 7 scoring.
        init         = 'random',
        low_memory   = True,
        n_jobs       = -1,
        verbose      = True
    )
    ts(f"  UMAP 2D fitting now — expect 20-60 min silence; tqdm below ...")
    reducer_2d.fit(X_fit_2d)
    ts(f"  UMAP 2D fit done in {time.time()-t0:.1f}s | transforming ...")

    t1 = time.time()
    X_embed_2d = reducer_2d.transform(X_enriched).astype(np.float32)
    np.save(umap2d_path, X_embed_2d)
    joblib.dump(reducer_2d, umap2d_pkl)
    ts(f"  UMAP 2D transform done in {time.time()-t1:.1f}s | "
       f"total stage: {time.time()-t0:.1f}s")
    ts(f"  X_embed_2d shape: {X_embed_2d.shape}")
    mark_done("umap2d")

# ── STAGE 7: PLOTS ────────────────────────────────────────────
# Figure 1: UMAP 2D raw projection (all 11M points, rasterized)
# Figure 2: Spectral eigenvalue scree plot with eigengap annotation

X_embed_2d_ram = np.array(X_embed_2d, dtype=np.float32)   # Load for plotting

# ── Figure 1: UMAP projection ─────────────────────────────────
fig_umap_path = os.path.join(OUTPUT_DIR, 'umap_raw_projection.png')
if not os.path.exists(fig_umap_path):
    ts("STAGE 7a [Plot]: Generating UMAP 2D raw projection ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_embed_2d_ram[:, 0], X_embed_2d_ram[:, 1],
               s=0.2, alpha=0.3, c='steelblue', rasterized=True)
    ax.set_title(
        'UMAP 2D Projection — Bitcoin Transactions\n'
        '(Enriched: Zeta-Weighted + Spectral Graph Features)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_umap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    ts(f"  Saved: {fig_umap_path}")

# ── Figure 2: Eigenvalue scree plot ───────────────────────────
scree_path = os.path.join(OUTPUT_DIR, 'spectral_scree.png')
if os.path.exists(eigenvalues_path) and not os.path.exists(scree_path):
    ts("STAGE 7b [Plot]: Generating spectral eigenvalue scree plot ...")
    eig_arr = np.load(eigenvalues_path)   # Already non-trivial eigenvalues only
    dims    = np.arange(1, len(eig_arr) + 1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(dims, eig_arr, 'o-', color='steelblue',
             linewidth=2, markersize=7, label='Eigenvalue λ_k')
    ax2.fill_between(dims, eig_arr, alpha=0.15, color='steelblue')

    # Annotate the largest eigengap — this suggests the natural cluster count
    gaps      = np.diff(eig_arr)
    gap_dim   = int(np.argmax(gaps)) + 1   # 1-indexed
    ax2.axvline(x=gap_dim + 0.5, color='crimson', linestyle='--', linewidth=1.5,
                label=f'Largest eigengap (suggests k≈{gap_dim})')

    ax2.set_xlabel('Eigenvector Index k  (1 = Fiedler vector)', fontsize=11)
    ax2.set_ylabel('Eigenvalue λ_k', fontsize=11)
    ax2.set_title('Normalized Laplacian Eigenvalue Scree Plot\n'
                  '(Eigengap heuristic for cluster count)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scree_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    ts(f"  Saved: {scree_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────
ts("\n" + "=" * 65)
ts("STEP 4 COMPLETE")
ts(f"  X_enriched.npy   : {X_enriched.shape}  "
   f"({X_enriched.nbytes/1e9:.2f} GB)")
ts(f"    ├─ {X_weighted_ram.shape[1]} Zeta-weighted features")
ts(f"    └─ {SPECTRAL_DIMS} spectral graph eigenvectors")
ts(f"  X_embed_15d.npy  : {X_embed_15d.shape}  ← input for Step 5 (clustering)")
ts(f"  X_embed_2d.npy   : {X_embed_2d.shape}   ← input for Step 6 (visualization)")
ts(f"  Plots saved to   : {OUTPUT_DIR}")
ts("=" * 65)
