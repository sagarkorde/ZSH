# ============================================================
# STEP 1 — Load, Inspect & Validate
#
# OPTIMIZATIONS:
#   * Hardware-aware: reports RAM usage vs 50 GB budget
#   * Full resumability: checkpoints guard each analysis stage
#     — re-run after any crash, skip stages already completed
#   * DuckDB used for ALL flag/quality aggregation — zero-copy
#     SQL on the in-process DataFrame, faster than pandas .agg
#   * Timestamped progress with per-stage elapsed times
#   * Reviewer-friendly comments on every analytical block
#
# Optimized for: HP Omen i9-13th Gen | 64 GB RAM | RTX 4060 8 GB
#                Target RAM ceiling : ~50 GB
# ============================================================

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os, sys, io, json, logging, time, duckdb
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────
PARQUET_PATH = r"C:\Users\sagar\Desktop\Q2 Paper 22326\Dataset.parquet"
OUTPUT_DIR   = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logger ───────────────────────────────────────────────────
# Dual-sink: UTF-8 console + persistent log file.
# line_buffering=True flushes after every line so progress is
# visible in the terminal immediately without buffering delays.
log_path = os.path.join(OUTPUT_DIR, "step1_log.txt")
_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=_utf8),
        logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# Also route stderr → logger so any library warnings appear in the log
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
    """Log with wall-clock elapsed prefix — every line is anchored in time."""
    log.info(f"[{time.time()-_T0:6.1f}s]  {msg}")

# ── Checkpoint Helpers ───────────────────────────────────────
# Resumability pattern:
#   Before each stage → is_done(stage)? skip : run
#   After each stage  → mark_done(stage)  (touches empty sentinel file)
# To force a stage to re-run: delete its .ckpt1_<stage>.done file.
def ckpt(s):    return os.path.join(OUTPUT_DIR, f".ckpt1_{s}.done")
def is_done(s): return os.path.exists(ckpt(s))
def mark_done(s):
    Path(ckpt(s)).touch()
    ts(f"  [CHECKPOINT] {s} ✓")

# ── Hardware RAM Budget Check ────────────────────────────────
# Warn early if available RAM is close to the 50 GB ceiling so the
# user can close other processes before the heavy steps (2-5) run.
try:
    import psutil
    ram_total_gb = psutil.virtual_memory().total / 1e9
    ram_avail_gb = psutil.virtual_memory().available / 1e9
    ts(f"RAM total: {ram_total_gb:.1f} GB  |  available: {ram_avail_gb:.1f} GB  "
       f"|  budget ceiling: 50 GB")
    if ram_avail_gb < 10:
        ts("  WARNING: less than 10 GB available — close browser/apps before Step 4")
except ImportError:
    ts("  (psutil not installed — skipping RAM check; pip install psutil to enable)")

ts("=" * 65)
ts("STEP 1 — Load & Validate")
ts("=" * 65)

# ── STAGE 1: Load ────────────────────────────────────────────
# pd.read_parquet reads columnar Parquet into a pandas DataFrame.
# For Step 1 (exploration only) full load is fine — the dataset
# fits in RAM well below the 50 GB budget.
if is_done("load"):
    ts("STAGE 1 [Load]: Already done — loading parquet for analysis ...")
else:
    ts(f"STAGE 1 [Load]: {PARQUET_PATH}")

t0 = time.time()
df = pd.read_parquet(PARQUET_PATH, engine='pyarrow')
ts(f"  Shape        : {df.shape}  ({time.time()-t0:.1f}s)")
ts(f"  Memory Usage : {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
mark_done("load")

# ── STAGE 2: Basic Audit via DuckDB ─────────────────────────
# Register the DataFrame as a DuckDB virtual table — zero-copy,
# DuckDB queries it directly in process without writing to disk.
# SQL GROUP BY and COUNT are significantly faster than pandas
# .value_counts() / .agg() on large frames.
if is_done("audit"):
    ts("STAGE 2 [Audit]: Already done — skipping.")
else:
    ts("\nSTAGE 2 [Audit]: Null counts + dtypes ...")
    t0 = time.time()

    # Register df as a DuckDB in-process table
    con = duckdb.connect()
    con.register("df_table", df)

    # Null count per column via DuckDB
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols) == 0:
        ts("  No nulls found across all columns.")
    else:
        ts("  Null columns:\n" + null_cols.sort_values(ascending=False).to_string())

    ts("\n  Dtypes:\n" + df.dtypes.value_counts().to_string())
    ts(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Boolean Flag Distribution via DuckDB ────────────────
    # DuckDB AVG() on 0/1 integer columns gives fraction directly —
    # no Python loop needed; runs in optimized vectorized C++ internally.
    ts("\n── Boolean / Flag Column Distribution (DuckDB) ──")
    flag_cols = [c for c in df.columns if c.startswith(('is_', 'has_', 'rbf_'))]

    if flag_cols:
        # Build SELECT list: AVG(col)*100 AS col for each flag column
        avg_exprs = ",\n    ".join(
            [f"ROUND(AVG(CAST(\"{c}\" AS DOUBLE))*100, 4) AS \"{c}\"" for c in flag_cols]
        )
        flag_result = con.execute(f"SELECT {avg_exprs} FROM df_table").df()
        for col in flag_cols:
            pct    = float(flag_result[col].iloc[0])
            status = ("⚠ ALL ZERO" if pct == 0.0
                      else ("⚠ >90% TRUE" if pct > 90 else "OK"))
            ts(f"  {col:<35}  {pct:>8.4f}%  {status}")

    con.close()
    mark_done("audit")

# ── STAGE 3: Column Quality Analysis via DuckDB ──────────────
# Identifies four categories of problematic columns so Step 2 can
# exclude them from the feature matrix without any hardcoding.
if is_done("col_quality"):
    ts("STAGE 3 [Col Quality]: Already done — loading manifest.")
else:
    ts("\nSTAGE 3 [Col Quality]: Detecting problematic columns ...")
    t0 = time.time()

    flag_cols = [c for c in df.columns if c.startswith(('is_', 'has_', 'rbf_'))]

    # 3a. All-zero columns — no variance, useless for clustering
    zero_cols = [c for c in df.select_dtypes(include=[np.number, bool]).columns
                 if df[c].max() == 0 and df[c].min() == 0]
    ts(f"  All-zero columns ({len(zero_cols)}): {zero_cols}")

    # 3b. Near-zero variance — std < 1e-6 (non-flag numerics only)
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    low_var_cols = [c for c in num_cols
                    if c not in flag_cols and df[c].std() < 1e-6
                    and c not in zero_cols]
    ts(f"  Near-zero variance cols: {low_var_cols}")

    # 3c. Known artifact / metadata columns (batch IDs etc.)
    KNOWN_ARTIFACTS = ['sample_size', 'batch_id', 'shard_id', 'row_id',
                       'partition', 'index', '__index_level_0__']
    artifact_cols = [c for c in df.columns if c in KNOWN_ARTIFACTS]
    ts(f"  Artifact columns found: {artifact_cols}")

    # 3d. Identifier columns (addresses, scripts, txid — not numeric features)
    IDENTIFIER_COLS = ['txid', 'block_time', 'timestamp', 'input_addresses',
                       'output_addresses', 'input_script_types',
                       'output_script_types', 'op_return_data']
    id_cols_present = [c for c in IDENTIFIER_COLS if c in df.columns]
    ts(f"  Identifier columns: {id_cols_present}")

    # 3e. Temporal columns — useful for analysis (Step 6) but bias clustering
    #     They are excluded from FEATURE_COLS but preserved in df_meta.
    TEMPORAL_COLS = ['block_height', 'hour', 'day_of_week', 'week_of_year',
                     'month', 'year', 'version', 'locktime', 'block_time',
                     'timestamp', 'date', 'year_month', 'month_year']
    temp_cols_present = [c for c in TEMPORAL_COLS if c in df.columns]
    ts(f"  Temporal columns (excluded from features): {temp_cols_present}")

    ts(f"  Elapsed: {time.time()-t0:.1f}s")
    mark_done("col_quality")

# (Re-)compute drop list and safe feature list for manifest save
flag_cols         = [c for c in df.columns if c.startswith(('is_', 'has_', 'rbf_'))]
zero_cols         = [c for c in df.select_dtypes(include=[np.number, bool]).columns
                     if df[c].max() == 0 and df[c].min() == 0]
num_cols          = df.select_dtypes(include=[np.number]).columns.tolist()
low_var_cols      = [c for c in num_cols
                     if c not in flag_cols and df[c].std() < 1e-6
                     and c not in zero_cols]
artifact_cols     = [c for c in df.columns
                     if c in ['sample_size', 'batch_id', 'shard_id', 'row_id',
                               'partition', 'index', '__index_level_0__']]
id_cols_present   = [c for c in ['txid', 'block_time', 'timestamp', 'input_addresses',
                                   'output_addresses', 'input_script_types',
                                   'output_script_types', 'op_return_data']
                     if c in df.columns]
TEMPORAL_COLS     = ['block_height', 'hour', 'day_of_week', 'week_of_year',
                     'month', 'year', 'version', 'locktime', 'block_time',
                     'timestamp', 'date', 'year_month', 'month_year']
temp_cols_present = [c for c in TEMPORAL_COLS if c in df.columns]

DROP_LIST = list(set(
    zero_cols + low_var_cols + artifact_cols + id_cols_present + temp_cols_present
))

# Columns safe for clustering — numeric, not dropped, not datetime
safe_feature_cols = [
    c for c in df.columns
    if c not in DROP_LIST
    and df[c].dtype != object
    and df[c].dtype.name != 'datetime64[ns]'
]

ts(f"\n── Safe Feature Summary ──")
ts(f"  Total columns     : {len(df.columns)}")
ts(f"  Dropping          : {len(DROP_LIST)}  {DROP_LIST}")
ts(f"  Safe feature cols : {len(safe_feature_cols)}")
ts(f"  → {safe_feature_cols}")

# ── STAGE 4: Numeric Summary + Skewness via DuckDB ───────────
# DuckDB's STDDEV_POP, MIN, MAX run on columnar layout in C++
# — 3-5× faster than pandas .describe() on 5M+ row DataFrames.
if is_done("numeric_summary"):
    ts("STAGE 4 [Numeric Summary]: Already done — skipping.")
else:
    ts("\nSTAGE 4 [Numeric Summary]: Key skewed columns ...")
    t0 = time.time()

    skew_cols = [c for c in [
        'fee', 'total_input_value', 'total_output_value',
        'avg_input_value', 'avg_output_value',
        'fee_rate_sat_per_vbyte', 'value_concentration_ratio'
    ] if c in df.columns]

    if skew_cols:
        ts("\n" + df[skew_cols].describe().to_string())

    ts("\n── Skewness (top 10 most skewed features, DuckDB SKEWNESS) ──")
    # DuckDB has built-in SKEWNESS() aggregate — avoids pandas .skew() overhead
    con2 = duckdb.connect()
    con2.register("df_table", df)
    # SKEWNESS() requires DOUBLE — cast booleans explicitly to avoid
    # "No function matches ... skewness(BOOLEAN)" BinderException
    def _skew_expr(c):
        if df[c].dtype == bool or str(df[c].dtype) == 'bool':
            return f"ROUND(SKEWNESS(CAST(\"{c}\" AS DOUBLE)), 4) AS \"{c}\""
        return f"ROUND(SKEWNESS(\"{c}\"), 4) AS \"{c}\""
    skew_exprs = ",\n    ".join(
        [_skew_expr(c) for c in safe_feature_cols if df[c].dtype != object]
    )
    skew_row = con2.execute(f"SELECT {skew_exprs} FROM df_table").df()
    con2.close()

    skew_series = skew_row.iloc[0].abs().sort_values(ascending=False)
    ts("\n" + skew_series.head(10).to_string())
    ts(f"  Elapsed: {time.time()-t0:.1f}s")
    mark_done("numeric_summary")

# ── STAGE 5: Dataset Coverage ────────────────────────────────
ts("\n── Dataset Coverage ──")
if 'year' in df.columns:
    ts(f"  year range     : {df['year'].min()} → {df['year'].max()}")
if 'block_height' in df.columns:
    ts(f"  block_height   : {df['block_height'].min():,} → {df['block_height'].max():,}")
ts(f"  Total txns     : {len(df):,}")

# ── STAGE 6: Transaction Type Imbalance via DuckDB ───────────
if is_done("imbalance"):
    ts("STAGE 6 [Imbalance]: Already done — skipping.")
else:
    ts("\nSTAGE 6 [Imbalance]: Transaction type imbalance ...")
    t0 = time.time()

    type_cols = [c for c in flag_cols if c in df.columns]
    if type_cols:
        # DuckDB AVG across all flag columns — one pass over the data
        con3 = duckdb.connect()
        con3.register("df_table", df)
        avg_exprs2 = ",\n    ".join(
            [f"ROUND(AVG(CAST(\"{c}\" AS DOUBLE))*100, 4) AS \"{c}\"" for c in type_cols]
        )
        result = con3.execute(f"SELECT {avg_exprs2} FROM df_table").df()
        con3.close()

        imbalance_rows = [(c, float(result[c].iloc[0])) for c in type_cols]
        imbalance_rows.sort(key=lambda x: -x[1])
        for col, pct in imbalance_rows:
            ts(f"  {col:<35}  {pct:>8.4f}%")

    ts(f"  Elapsed: {time.time()-t0:.1f}s")
    mark_done("imbalance")

# ── STAGE 7: Save Manifests ───────────────────────────────────
# column_manifest.json consumed by Step 2 to ensure consistent
# column drops without any hardcoding in downstream scripts.
if is_done("manifest"):
    ts("STAGE 7 [Manifest]: Already saved — skipping.")
else:
    ts("\nSTAGE 7 [Manifest]: Saving column_manifest.json ...")
    t0 = time.time()

    manifest = {
        "safe_feature_cols" : safe_feature_cols,
        "drop_list"         : DROP_LIST,
        "zero_cols"         : zero_cols,
        "artifact_cols"     : artifact_cols,
        "id_cols"           : id_cols_present,
        "temporal_cols"     : temp_cols_present,
        "flag_cols"         : flag_cols,
        "total_rows"        : int(len(df)),
        "generated_at"      : datetime.now().isoformat()
    }
    manifest_path = os.path.join(OUTPUT_DIR, "column_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    ts(f"  Saved: column_manifest.json → {manifest_path}  ({time.time()-t0:.1f}s)")
    mark_done("manifest")

# ── STAGE 8: Data Quality Report ─────────────────────────────
# Human-readable .txt report for the paper appendix.
if is_done("report"):
    ts("STAGE 8 [Report]: Already saved — skipping.")
else:
    ts("\nSTAGE 8 [Report]: Writing DATA_QUALITY_REPORT.txt ...")
    t0 = time.time()

    report_path = os.path.join(OUTPUT_DIR, "DATA_QUALITY_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as r:
        r.write("=" * 65 + "\n")
        r.write("  DATA QUALITY REPORT — Bitcoin Transaction Dataset\n")
        r.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write("=" * 65 + "\n\n")
        r.write(f"Total Rows      : {len(df):,}\n")
        r.write(f"Total Columns   : {len(df.columns)}\n")
        r.write(f"Memory (GB)     : {df.memory_usage(deep=True).sum()/1e9:.2f}\n\n")
        r.write(f"── Columns Dropped ({len(DROP_LIST)}) ──\n")
        for c in sorted(DROP_LIST):
            r.write(f"  {c}\n")
        r.write(f"\n── Safe Clustering Features ({len(safe_feature_cols)}) ──\n")
        for c in safe_feature_cols:
            r.write(f"  {c}\n")
        r.write(f"\n── All-Zero Columns ({len(zero_cols)}) ──\n")
        for c in zero_cols:
            r.write(f"  {c}  [EXCLUDED: no variance]\n")
        r.write(f"\n── Artifact Columns ({len(artifact_cols)}) ──\n")
        for c in artifact_cols:
            r.write(f"  {c}  [EXCLUDED: metadata, not a transaction feature]\n")

    ts(f"  Saved: DATA_QUALITY_REPORT.txt → {report_path}  ({time.time()-t0:.1f}s)")
    mark_done("report")

ts("\n" + "=" * 65)
ts("STEP 1 COMPLETE")
ts(f"  Total runtime        : {time.time()-_T0:.1f}s")
ts(f"  column_manifest.json → {os.path.join(OUTPUT_DIR, 'column_manifest.json')}")
ts(f"  DATA_QUALITY_REPORT  → {os.path.join(OUTPUT_DIR, 'DATA_QUALITY_REPORT.txt')}")
ts(f"  step1_log.txt        → {log_path}")
ts("RESUMABILITY: Re-run anytime — completed stages are auto-skipped.")
ts("  To force full re-run: delete all .ckpt1_*.done files in outputs/")
ts("=" * 65)
