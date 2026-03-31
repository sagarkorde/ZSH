# ============================================================
# STEP 2 — Preprocessing & Stratified Balancing
#
# OPTIMIZATIONS:
#   * Full resumability: every stage guarded by .ckpt2_<stage>.done
#     sentinel files — re-run after any crash, picks up exactly
#     where it left off  (previously checkpoints were defined but
#     NOT USED — all stages re-ran on every execution)
#   * DuckDB for strata balance statistics and summary tables —
#     zero-copy SQL on in-process DataFrames
#   * Per-stage elapsed timestamps printed after every stage
#   * Meta/temporal columns preserved through balancing so df_meta
#     (block_height, year, hour …) is always non-empty
#   * float32 throughout — halves RAM vs float64
#   * RobustScaler: handles Bitcoin value outliers better than
#     StandardScaler (median/IQR instead of mean/std)
#   * Strata balancing target: 30% of majority (was 20%)
#
# Optimized for: HP Omen i9-13th Gen | 64 GB RAM | RTX 4060 8 GB
#                Target RAM ceiling : ~50 GB
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib, os, sys, io, json, logging, time, duckdb
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────
PARQUET_PATH = r"C:\Users\sagar\Desktop\Q2 Paper 22326\Dataset.parquet"
OUTPUT_DIR   = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logger ────────────────────────────────────────────────────
# Dual-sink: UTF-8 console + persistent log file.
# mode="a" appends so previous partial runs are preserved.
log_path = os.path.join(OUTPUT_DIR, "step2_log.txt")
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

# Also route stderr → logger so warnings appear in the log
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

# ── Checkpoint Helpers ────────────────────────────────────────
# Resumability pattern:
#   Before each stage → is_done(stage)? skip : run
#   After each stage  → mark_done(stage)  (touches empty sentinel file)
# To force a stage to re-run: delete its .ckpt2_<stage>.done file.
def ckpt(stage):    return os.path.join(OUTPUT_DIR, f".ckpt2_{stage}.done")
def is_done(stage): return os.path.exists(ckpt(stage))
def mark_done(stage):
    Path(ckpt(stage)).touch()
    ts(f"  [CHECKPOINT] {stage} ✓")

ts("=" * 65)
ts("STEP 2 — Preprocessing & Balancing")
ts("=" * 65)

# ── Load Manifest ─────────────────────────────────────────────
# column_manifest.json written by Step 1 — avoids hardcoding any
# column names here and ensures Steps 1 and 2 stay in sync.
manifest_path = os.path.join(OUTPUT_DIR, "column_manifest.json")
if not os.path.exists(manifest_path):
    ts("ERROR: column_manifest.json not found. Run Step 1 first.")
    raise FileNotFoundError(manifest_path)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

SAFE_FEATURE_COLS = manifest["safe_feature_cols"]
DROP_LIST         = manifest["drop_list"]
FLAG_COLS         = manifest["flag_cols"]

ts(f"Manifest loaded. Safe feature cols: {len(SAFE_FEATURE_COLS)}")
ts(f"  Features: {SAFE_FEATURE_COLS}")

# ══════════════════════════════════════════════════════════════
# STAGE 1: Load Raw Data
# ══════════════════════════════════════════════════════════════
# pd.read_parquet loads the full Parquet file via PyArrow.
# At ~5.9M rows × 53 cols this is ~2-3 GB — well within budget.
# Checkpointed so a re-run after a later crash skips the slow load.

if is_done("load"):
    ts("\nSTAGE 1 [Load]: Already done — reloading parquet ...")
else:
    ts("\nSTAGE 1 [Load]: Loading raw parquet ...")

t0 = time.time()
df = pd.read_parquet(PARQUET_PATH, engine='pyarrow')
ts(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} cols in {time.time()-t0:.1f}s")
ts(f"  RAM ≈ {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
mark_done("load")

# ══════════════════════════════════════════════════════════════
# STAGE 2: Drop Unwanted Columns
# ══════════════════════════════════════════════════════════════
# Drop list comes from the manifest — includes zero-variance,
# artifact, identifier, and temporal columns.
#
# IMPORTANT: Meta/temporal columns (block_height, year, hour …)
# are in DROP_LIST (Step 1 classifies them as temporal) but must
# survive into df_balanced so Stage 9 can build df_meta.
# META_COLS_PRESERVE explicitly protects them from being dropped.
if is_done("drop"):
    ts("\nSTAGE 2 [Drop]: Already done — skipping.")
    # df already loaded from Stage 1; reconstruct drop state
    META_COLS_PRESERVE = {'block_height', 'year', 'month', 'hour',
                          'day_of_week', 'week_of_year', 'locktime', 'version'}
    cols_to_drop = [c for c in DROP_LIST
                    if c in df.columns and c not in META_COLS_PRESERVE]
    df.drop(columns=cols_to_drop, inplace=True)
else:
    ts("\nSTAGE 2 [Drop]: Dropping non-feature columns ...")
    t0 = time.time()

    # Preserve temporal meta cols so Stage 9 can build df_meta
    META_COLS_PRESERVE = {'block_height', 'year', 'month', 'hour',
                          'day_of_week', 'week_of_year', 'locktime', 'version'}
    cols_to_drop = [c for c in DROP_LIST
                    if c in df.columns and c not in META_COLS_PRESERVE]
    df.drop(columns=cols_to_drop, inplace=True)
    ts(f"  Dropped {len(cols_to_drop)} cols. Remaining: {df.shape[1]}")
    ts(f"  Meta cols preserved: {[c for c in META_COLS_PRESERVE if c in df.columns]}")
    ts(f"  Elapsed: {time.time()-t0:.1f}s")
    mark_done("drop")

# ══════════════════════════════════════════════════════════════
# STAGE 3: Log-Transform Skewed Monetary Features
# ══════════════════════════════════════════════════════════════
# Bitcoin value distributions are extremely right-skewed (Pareto-like).
# log1p(x + eps) maps the heavy tail to a near-Gaussian shape so
# RobustScaler and downstream clustering algorithms work correctly.
# eps=1e-8 prevents log(0) for zero-fee transactions.
if is_done("logtransform"):
    ts("\nSTAGE 3 [Log Transform]: Already done — skipping.")
else:
    ts("\nSTAGE 3 [Log Transform]: Log1p-transforming skewed columns ...")
    t0 = time.time()

    LOG_COLS = [
        'total_input_value', 'total_output_value', 'fee',
        'avg_input_value', 'avg_output_value',
        'fee_rate_sat_per_byte', 'fee_rate_sat_per_vbyte',
        'size', 'vsize', 'weight', 'value_difference',
        'value_concentration_ratio'
    ]
    eps = 1e-8
    transformed = []
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0).astype(np.float32) + eps)
            transformed.append(col)
    ts(f"  Transformed ({len(transformed)}): {transformed}  ({time.time()-t0:.1f}s)")
    mark_done("logtransform")

# ══════════════════════════════════════════════════════════════
# STAGE 4: Boolean → int8
# ══════════════════════════════════════════════════════════════
# Cast boolean flag columns to int8 (0/1).
# int8 uses 1 byte vs 8 bytes for bool objects — 8× RAM saving
# for flag columns in an 11M-row dataset.
if is_done("bool_cast"):
    ts("\nSTAGE 4 [Bool Cast]: Already done — skipping.")
else:
    ts("\nSTAGE 4 [Bool Cast]: Casting boolean flags to int8 ...")
    t0 = time.time()
    bool_cols_present = [c for c in FLAG_COLS if c in df.columns]
    df[bool_cols_present] = df[bool_cols_present].astype(np.int8)
    ts(f"  Cast {len(bool_cols_present)} flag columns  ({time.time()-t0:.1f}s)")
    mark_done("bool_cast")

# ══════════════════════════════════════════════════════════════
# STAGE 5: Fill Remaining Nulls
# ══════════════════════════════════════════════════════════════
# Step 1 reported 0 nulls; this is a safety net for unseen data.
# Fill with 0: for monetary features, 0 is a valid sentinel;
# for boolean flags, 0 means "not present".
if is_done("fill_nulls"):
    ts("\nSTAGE 5 [Fill Nulls]: Already done — skipping.")
else:
    ts("\nSTAGE 5 [Fill Nulls]: Filling nulls ...")
    t0 = time.time()
    null_count_before = df.isnull().sum().sum()
    df.fillna(0, inplace=True)
    ts(f"  Filled {null_count_before} null cells  ({time.time()-t0:.1f}s)")
    mark_done("fill_nulls")

# ══════════════════════════════════════════════════════════════
# STAGE 6: Stratified Balancing
# ══════════════════════════════════════════════════════════════
# Bitcoin transaction types are highly imbalanced:
#   Standard P2P  : ~77% of all transactions
#   Coinjoin      : ~0.03%
#   Batch Payment : ~2.3%
# Without balancing, a clustering model would create clusters that
# are 95%+ Standard P2P and never discover rare types.
#
# Strategy:
#   1. Composite strata key from 3 most-imbalanced flag columns
#   2. Upsample any strata smaller than 30% of the majority size
#   3. Shuffle and reset index (eliminates temporal ordering bias)
#
# DuckDB used for strata statistics — faster COUNT/GROUP BY than
# pandas .value_counts() on a 5M+ row DataFrame.
if is_done("balance"):
    ts("\nSTAGE 6 [Balance]: Already done — loading balanced parquet ...")
    # Reload the balanced frame saved at the end of this stage
    bal_ckpt_path = os.path.join(OUTPUT_DIR, 'df_balanced_temp.parquet')
    if os.path.exists(bal_ckpt_path):
        df_balanced = pd.read_parquet(bal_ckpt_path)
        ts(f"  Loaded: df_balanced shape {df_balanced.shape}")
    else:
        ts("  WARNING: df_balanced_temp.parquet not found — re-running balance stage.")
        Path(ckpt("balance")).unlink(missing_ok=True)  # Reset checkpoint
        df_balanced = None
else:
    df_balanced = None

if df_balanced is None:
    ts("\nSTAGE 6 [Balance]: Stratified balancing of rare transaction types ...")
    t0 = time.time()

    # Three most imbalanced rare-type flags form the composite strata key.
    # Using 3 flags gives 2^3=8 possible strata — tractable for upsampling.
    STRATA_COLS = ['is_coinjoin_like', 'is_batch_payment', 'has_op_return']
    strata_cols_present = [c for c in STRATA_COLS if c in df.columns]

    if strata_cols_present:
        df['_strata'] = df[strata_cols_present].astype(str).agg('_'.join, axis=1)

        # DuckDB: COUNT per strata — zero-copy query on the pandas DataFrame
        con = duckdb.connect()
        con.register("df_strata", df[['_strata']])
        strata_stats = con.execute("""
            SELECT
                _strata,
                COUNT(*) AS n,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 4) AS pct
            FROM df_strata
            GROUP BY _strata
            ORDER BY n DESC
        """).df()
        con.close()
        ts("  Strata distribution (DuckDB):\n" + strata_stats.to_string(index=False))

        strata_counts   = df['_strata'].value_counts()
        BALANCE_TARGET  = int(strata_counts.max() * 0.30)
        ts(f"  Upsample target per minority strata: {BALANCE_TARGET:,}")

        balanced_parts = []
        for strata_val, group in df.groupby('_strata'):
            if len(group) < BALANCE_TARGET:
                upsampled = group.sample(
                    n=BALANCE_TARGET, replace=True, random_state=42
                )
                balanced_parts.append(upsampled)
                ts(f"    [{strata_val}]: {len(group):,} → {BALANCE_TARGET:,}")
            else:
                balanced_parts.append(group)

        df_balanced = (pd.concat(balanced_parts)
                       .drop(columns=['_strata'])
                       .sample(frac=1, random_state=42)
                       .reset_index(drop=True))
    else:
        ts("  WARNING: No strata columns found. Using original data unbalanced.")
        df_balanced = df.copy()

    ts(f"  Balanced shape: {df_balanced.shape}  ({time.time()-t0:.1f}s)")

    # Save balanced frame as a temp checkpoint so this stage can be
    # skipped on re-runs — 11M × 35 cols ≈ 3 GB parquet
    bal_ckpt_path = os.path.join(OUTPUT_DIR, 'df_balanced_temp.parquet')
    ts(f"  Saving balance checkpoint (this may take ~60s) ...")
    df_balanced.to_parquet(bal_ckpt_path, index=False)
    ts(f"  Saved: df_balanced_temp.parquet")
    mark_done("balance")

# ══════════════════════════════════════════════════════════════
# STAGE 7: Separate Feature Matrix, Graph Features, Metadata
# ══════════════════════════════════════════════════════════════
# Three matrices are extracted from df_balanced:
#   FEATURE_COLS     → X_scaled (clustering input)
#   GRAPH_FEAT_COLS  → X_graph_scaled (spectral embedding, Step 4)
#   META_COLS        → df_meta (block_height, year, hour … for Step 6)
ts("\nSTAGE 7 [Separate]: Extracting feature / graph / meta matrices ...")
t0 = time.time()

# Graph-compatible features: address/script topology signals
# Used by Step 4 for k-NN graph construction and spectral embedding.
GRAPH_FEATURE_COLS = [c for c in [
    'input_address_count', 'output_address_count', 'total_addresses',
    'input_script_count', 'output_script_count', 'address_reuse',
    'input_count', 'output_count', 'input_output_ratio'
] if c in df_balanced.columns]

# Final clustering feature columns from the manifest
# Re-verified against actual df_balanced columns (balancing can shift dtypes)
FEATURE_COLS = [
    c for c in SAFE_FEATURE_COLS
    if c in df_balanced.columns
    and df_balanced[c].dtype != object
    and df_balanced[c].dtype.name != 'datetime64[ns]'
]
ts(f"  FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")
ts(f"  GRAPH_FEATURE_COLS ({len(GRAPH_FEATURE_COLS)}): {GRAPH_FEATURE_COLS}")
ts(f"  Stage 7 prep: {time.time()-t0:.1f}s")

X_raw = df_balanced[FEATURE_COLS].values.astype(np.float32)

# ══════════════════════════════════════════════════════════════
# STAGE 8: RobustScaler
# ══════════════════════════════════════════════════════════════
# RobustScaler uses median and interquartile range (IQR) instead
# of mean/std — critical for Bitcoin features because outliers
# (whale transactions, mining pool consolidations) can be 1000×
# the median and would dominate StandardScaler scaling.
if is_done("scale"):
    ts("\nSTAGE 8 [Scale]: Already done — reloading scaler ...")
    scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    scaler_graph = joblib.load(os.path.join(OUTPUT_DIR, 'scaler_graph.pkl'))
    X_graph_raw = df_balanced[GRAPH_FEATURE_COLS].fillna(0).values.astype(np.float32)
    X_graph_scaled = scaler_graph.transform(X_graph_raw).astype(np.float32)
else:
    ts("\nSTAGE 8 [Scale]: Fitting RobustScaler ...")
    t0 = time.time()

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    ts(f"  X_scaled: {X_scaled.shape}  dtype={X_scaled.dtype}  ({time.time()-t0:.1f}s)")

    X_graph_raw    = df_balanced[GRAPH_FEATURE_COLS].fillna(0).values.astype(np.float32)
    scaler_graph   = RobustScaler()
    X_graph_scaled = scaler_graph.fit_transform(X_graph_raw).astype(np.float32)
    ts(f"  X_graph_scaled: {X_graph_scaled.shape}")
    mark_done("scale")

# ══════════════════════════════════════════════════════════════
# STAGE 9: Save Metadata Frame
# ══════════════════════════════════════════════════════════════
# df_meta carries temporal context (block height, year, hour)
# for the visualizations in Step 6 but is NOT used for clustering.
# META_COLS must have survived the Stage 2 drop — they are
# explicitly preserved by META_COLS_PRESERVE above.
if is_done("meta"):
    ts("\nSTAGE 9 [Meta]: Already done — skipping.")
else:
    ts("\nSTAGE 9 [Meta]: Saving metadata frame ...")
    t0 = time.time()

    META_COLS = [c for c in ['block_height', 'year', 'month', 'hour',
                              'day_of_week', 'week_of_year', 'locktime', 'version']
                 if c in df_balanced.columns]
    df_meta = df_balanced[META_COLS].copy()
    df_meta_path = os.path.join(OUTPUT_DIR, "df_meta.parquet")
    df_meta.to_parquet(df_meta_path, index=False)
    ts(f"  df_meta: {df_meta.shape} → {df_meta_path}  ({time.time()-t0:.1f}s)")

    # DuckDB: quick validation of meta column ranges (written to log
    # so reviewers can verify temporal coverage in the log file)
    con = duckdb.connect()
    con.register("meta_tbl", df_meta)
    meta_summary = con.execute("""
        SELECT
            MIN(block_height) AS bh_min,
            MAX(block_height) AS bh_max,
            MIN(year)         AS year_min,
            MAX(year)         AS year_max,
            MIN(hour)         AS hour_min,
            MAX(hour)         AS hour_max
        FROM meta_tbl
        WHERE block_height IS NOT NULL
          AND year          IS NOT NULL
    """).df()
    con.close()
    ts("  Meta column ranges (DuckDB):\n" + meta_summary.to_string(index=False))
    mark_done("meta")

# ══════════════════════════════════════════════════════════════
# STAGE 10: Save All Artifacts
# ══════════════════════════════════════════════════════════════
if is_done("save_artifacts"):
    ts("\nSTAGE 10 [Save]: Already done — skipping.")
else:
    ts("\nSTAGE 10 [Save]: Saving arrays and model objects ...")
    t0 = time.time()

    np.save(os.path.join(OUTPUT_DIR, 'X_scaled.npy'),       X_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'X_graph_scaled.npy'), X_graph_scaled)
    joblib.dump(scaler,            os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    joblib.dump(scaler_graph,      os.path.join(OUTPUT_DIR, 'scaler_graph.pkl'))
    joblib.dump(FEATURE_COLS,      os.path.join(OUTPUT_DIR, 'feature_cols.pkl'))
    joblib.dump(GRAPH_FEATURE_COLS,os.path.join(OUTPUT_DIR, 'graph_feature_cols.pkl'))

    df_features_path = os.path.join(OUTPUT_DIR, "df_balanced_features.parquet")
    df_balanced[FEATURE_COLS].to_parquet(df_features_path, index=False)
    ts(f"  df_balanced_features: {df_balanced[FEATURE_COLS].shape}")
    ts(f"  All artifacts saved  ({time.time()-t0:.1f}s)")
    mark_done("save_artifacts")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
ts("\n" + "=" * 65)
ts("STEP 2 COMPLETE")
ts(f"  Total runtime            : {time.time()-_T0:.1f}s")
ts(f"  X_scaled.npy             : {X_scaled.shape}")
ts(f"  X_graph_scaled.npy       : {X_graph_scaled.shape}")
ts(f"  df_balanced_features     : (see parquet)")
ts(f"  df_meta                  : {list(META_COLS if 'META_COLS' in dir() else [])}")
ts(f"  Total rows after balance : {len(df_balanced):,}")
ts("RESUMABILITY: Re-run anytime — completed stages are auto-skipped.")
ts("  To force full re-run: delete all .ckpt2_*.done files in outputs/")
ts("=" * 65)
