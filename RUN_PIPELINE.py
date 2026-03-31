# ====================================================================
# RUN_PIPELINE.py  —  Master Pipeline Runner
# ====================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Runs all 6 pipeline steps (+ COMPARISON.py) in correct order
#   2. Each step uses its own internal checkpoint system → re-run
#      safely at any time; only incomplete stages re-execute
#   3. After COMPARISON.py (Step 5), bridges its output files to the
#      standard names that Step_6 expects
#   4. Verifies inter-step sync: checks all key arrays share the same
#      row count (n_rows = 11,303,526 for the full dataset)
#   5. Deletes stale / intermediate files from a previous algorithm
#      experiment (HDBSCAN, spectral, old K-Means, old optimization)
#
# USAGE (single command from PowerShell or terminal):
#   & C:\ProgramData\anaconda3\envs\gpu-env\python.exe "c:/Users/sagar/Desktop/Q2 Paper 22326/RUN_PIPELINE.py"
#
# TO FORCE RE-RUN A SPECIFIC STEP:
#   Step 1 → delete outputs/.ckpt1_*.done
#   Step 2 → delete outputs/.ckpt2_*.done
#   Step 3 → delete outputs/.ckpt3_*.done
#   Step 4 → delete outputs/.ckpt4_*.done
#   Step 5 → delete outputs/zsh_improved_labels.npy
#   Step 6 → delete outputs/.ckpt6_*.done
#
# HARDWARE TARGET:
#   HP Omen | Intel i9-13th Gen | 64 GB RAM | RTX 4060 8 GB
# ====================================================================

import subprocess, sys, os, time, shutil, logging, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Install packages that individual steps need but may be missing ────
# Uses pip --quiet so it skips silently if already installed.
# Only installs lightweight packages here; heavy ones (umap-learn,
# torch) are assumed present from the gpu-env conda environment.
def _ensure_packages():
    needed = ["seaborn", "duckdb", "pyarrow", "joblib"]
    for pkg in needed:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q",
             "--disable-pip-version-check"],
            check=False, capture_output=True
        )

_ensure_packages()

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION  — only edit this block if paths change
# ─────────────────────────────────────────────────────────────────────

PYTHON   = r"C:\ProgramData\anaconda3\envs\gpu-env\python.exe"
WORK_DIR = r"C:\Users\sagar\Desktop\Q2 Paper 22326"
OUT_DIR  = os.path.join(WORK_DIR, "outputs")

# Pipeline order: (step_number, display_label, script_filename)
STEPS = [
    (1, "Load & Explore",      "Step_1_Load_and_Explore.py"),
    (2, "Preprocess",          "Step_2_Preprocess.py"),
    (3, "Zeta Weighting",      "Step_3_Zeta_Weighting.py"),
    (4, "UMAP Embedding",      "Step_4_UMAP_Embedding.py"),
    (5, "ZSH Clustering",      "Step_5_ZSH_Clustering.py"), # novel algorithm (improved ZSH)
    (6, "Profile & Visualize", "Step_6_Profile_and_Visualize.py"),
]

# Minimum set of output files that must exist after each step.
# Master runner fails fast if any are missing — preventing cascading errors.
REQUIRED_AFTER_STEP = {
    1: ["column_manifest.json", "DATA_QUALITY_REPORT.txt",
        "step1_log.txt"],
    2: ["df_balanced_features.parquet", "df_meta.parquet",
        "feature_cols.pkl", "scaler.pkl",
        "X_scaled.npy", "X_graph_scaled.npy",
        "scaler_graph.pkl", "graph_feature_cols.pkl",
        "step2_log.txt"],
    3: ["X_weighted.npy", "feature_ranks.csv",
        "step3_log.txt"],
    4: ["X_embed_15d.npy", "X_embed_2d.npy",
        "umap_15d.pkl", "umap_2d.pkl",
        "step4_log.txt"],
    5: ["zsh_improved_labels.npy", "zsh_improved_profiles.npy",
        "zsh_improved_anomaly_scores.npy", "zsh_improved_anomaly_flags.npy",
        "zsh_improved_comparison.csv"],
    6: ["profile_statistics.csv", "cluster_assignments.csv",
        "step6_log.txt"],
}

# Key numpy arrays + parquets used for row-count sync verification
SYNC_ARRAYS = [
    "X_weighted.npy",
    "X_embed_2d.npy",
    "final_labels.npy",
    "final_profiles.npy",
    "outlier_scores.npy",
    "anomaly_flags.npy",
]
SYNC_PARQUETS = [
    "df_balanced_features.parquet",
    "df_meta.parquet",
]

# ─────────────────────────────────────────────────────────────────────
# FILES TO DELETE (stale from previous algorithm experiments)
# ─────────────────────────────────────────────────────────────────────
# These are from old HDBSCAN / spectral / optimization runs that are
# no longer part of the final pipeline.  Safe to delete.
STALE_FILES = [
    # ── Old HDBSCAN algorithm (fully replaced by ZSH Hybrid) ──────
    "hdb_labels_clean.npy", "hdb_labels_raw.npy", "hdb_labels_optimized.npy",
    "hdb_sample_idx.npy", "hdbscan_model.pkl", "hdbscan_model_optimized.pkl",
    "hdbscan_param_scatter.png", "hdbscan_param_search.log",
    "hdbscan_param_search.png", "hdbscan_param_search_results.csv",

    # ── Old plain K-Means (kept only as baseline inside COMPARISON.py) ──
    "km_labels.npy", "km_labels_optimized.npy",
    "kmeans_model.pkl", "kmeans_model_optimized.pkl",
    "optimal_k_search.png",

    # ── Old spectral clustering experiment ─────────────────────────
    "X_spectral.npy", "spectral_eigenvalues.npy", "spectral_sample_idx.npy",
    "spectral_scree.png",

    # ── Old "optimized" / "publication" experiment outputs ─────────
    "final_labels_optimized.npy", "outlier_scores_optimized.npy",
    "anomaly_flags_optimized.npy", "k_search_metrics.csv",
    "cluster_stability.csv", "optimal_k_optimized.csv",
    "optimal_k_optimized.png",
    "clustering_metrics_optimized.db", "clustering_metrics_publication.db",
    "step5_optimized_log.txt", "step5_publication_log.txt",

    # ── Intermediate-only arrays (not needed after pipeline completes) ──
    "proxy_labels.npy", "mi_scores.npy",
    "X_enriched.npy",          # before Zeta-weighting; X_weighted.npy is final

    # ── Old Step_5_ZSH_Clustering.py outputs (COMPARISON.py is the winner) ──
    "taxonomy_labels.npy", "sub_taxonomy_labels.npy",
    "gmm_labels.npy", "gmm_model.pkl", "gmm_probabilities.npy",
    "gmm_best_k.txt", "gmm_bic_search.csv", "pca_gmm.pkl",
    "baseline_km_labels.npy", "baseline_kmeans.pkl",
    "isolation_forest.pkl",
    "step5_metrics.db",
    "step5_log.txt",
    "STEP5_RESULTS.txt",
    "metrics_comparison.csv",
    "metrics_comparison.png",   # regenerated by Step_6

    # ── Misc stale artefacts ────────────────────────────────────────
    "anomaly_by_cluster.csv",
    "umap_raw_projection.png",
    "0",                        # empty file created by some failed runs
]

# ─────────────────────────────────────────────────────────────────────
# LOGGER  (console + persistent log)
# ─────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
_log_path = os.path.join(OUT_DIR, "pipeline_runner_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("runner")


def ts(msg: str):
    log.info(msg)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def step5_is_done() -> bool:
    """
    COMPARISON.py has no .ckpt files — consider it done if the output
    zsh_improved_labels.npy exists AND its row count matches X_weighted.npy.
    """
    labels_path = os.path.join(OUT_DIR, "zsh_improved_labels.npy")
    if not os.path.exists(labels_path):
        return False
    xw_path = os.path.join(OUT_DIR, "X_weighted.npy")
    if not os.path.exists(xw_path):
        return True  # X_weighted not produced yet; can't compare row counts
    try:
        n_labels = np.load(labels_path, mmap_mode="r").shape[0]
        n_xw     = np.load(xw_path,     mmap_mode="r").shape[0]
        return n_labels == n_xw
    except Exception:
        return False


def bridge_step5_outputs():
    """
    COMPARISON.py saves outputs with "zsh_improved_" prefix.
    Step_6 expects the standard names below.  This function copies
    and renames them — no data modification except for outlier_scores
    which is min-max scaled to [0,1] so that Step_6's 0.80 threshold
    correctly identifies the most anomalous ~5% of transactions.
    """
    ts("\n  ── Bridging COMPARISON.py → Step_6 file names ──────────────")

    # Direct copy (no transformation needed)
    simple_map = [
        ("zsh_improved_labels.npy",       "final_labels.npy"),
        ("zsh_improved_profiles.npy",      "final_profiles.npy"),
        ("zsh_improved_anomaly_flags.npy", "anomaly_flags.npy"),
    ]
    for src_name, dst_name in simple_map:
        src = os.path.join(OUT_DIR, src_name)
        dst = os.path.join(OUT_DIR, dst_name)
        shutil.copy2(src, dst)
        ts(f"    {src_name:<42}  →  {dst_name}")

    # outlier_scores.npy: Step_6 uses a hard threshold of 0.80.
    # We scale the raw anomaly scores to [0,1] so that the top-5%
    # (most anomalous) transactions have scores > 0.80, matching the
    # contamination=0.05 target used in COMPARISON.py.
    scores_src = os.path.join(OUT_DIR, "zsh_improved_anomaly_scores.npy")
    raw    = np.load(scores_src).astype(np.float32)
    rmin, rmax = float(raw.min()), float(raw.max())
    scaled = (raw - rmin) / (rmax - rmin + 1e-9)

    np.save(os.path.join(OUT_DIR, "outlier_scores.npy"), scaled)
    np.save(os.path.join(OUT_DIR, "anomaly_scores.npy"), scaled)  # alias

    ts(f"    zsh_improved_anomaly_scores.npy → outlier_scores.npy  "
       f"(min-max scaled [{rmin:.4f}, {rmax:.4f}] → [0, 1])")
    ts("  Bridge complete.\n")


def verify_sync() -> bool:
    """
    Check that all key pipeline outputs exist and share the same row count.
    Returns True if everything is in sync, False otherwise.
    """
    ts("\n" + "=" * 65)
    ts("PIPELINE SYNC VERIFICATION")
    ts("=" * 65)

    row_counts: dict[str, int] = {}
    all_ok = True

    # ── numpy arrays ──────────────────────────────────────────────
    for fname in SYNC_ARRAYS:
        fpath = os.path.join(OUT_DIR, fname)
        if not os.path.exists(fpath):
            ts(f"  MISSING  {fname}")
            all_ok = False
            continue
        try:
            # Object-dtype arrays (e.g. string profiles) cannot use mmap;
            # fall back to allow_pickle load just to get shape.
            try:
                arr = np.load(fpath, mmap_mode="r")
            except ValueError:
                arr = np.load(fpath, allow_pickle=True)
            n       = arr.shape[0]
            size_mb = os.path.getsize(fpath) / 1e6
            dtype   = arr.dtype
            row_counts[fname] = n
            ts(f"  OK  {fname:<40}  n={n:>11,}  {dtype}  {size_mb:>8.1f} MB")
        except Exception as exc:
            ts(f"  ERROR reading {fname}: {exc}")
            all_ok = False

    # ── parquet files (use pyarrow metadata for fast row count) ───
    try:
        import pyarrow.parquet as pq
        use_pq = True
    except ImportError:
        import pandas as pd
        use_pq = False

    for fname in SYNC_PARQUETS:
        fpath = os.path.join(OUT_DIR, fname)
        if not os.path.exists(fpath):
            ts(f"  MISSING  {fname}")
            all_ok = False
            continue
        try:
            if use_pq:
                n = pq.read_metadata(fpath).num_rows
            else:
                n = len(pd.read_parquet(fpath, columns=[
                    pd.read_parquet(fpath, columns=None).columns[0]
                ]))
            size_mb = os.path.getsize(fpath) / 1e6
            row_counts[fname] = n
            ts(f"  OK  {fname:<40}  n={n:>11,}         {size_mb:>8.1f} MB")
        except Exception as exc:
            ts(f"  ERROR reading {fname}: {exc}")
            all_ok = False

    # ── Row-count consistency ──────────────────────────────────────
    counts = list(row_counts.values())
    if len(set(counts)) > 1:
        ts("\n  ⚠  ROW COUNT MISMATCH — files are NOT in sync:")
        for fname, n in row_counts.items():
            ts(f"    {fname:<40}  n={n:,}")
        all_ok = False
    elif counts:
        ts(f"\n  ✓  All {len(row_counts)} files in sync: {counts[0]:,} rows")

    # ── Per-step required-file check ──────────────────────────────
    ts("\nRequired-file presence by step:")
    for snum, files in REQUIRED_AFTER_STEP.items():
        missing = [f for f in files
                   if not os.path.exists(os.path.join(OUT_DIR, f))]
        if missing:
            ts(f"  Step {snum}: MISSING → {missing}")
            all_ok = False
        else:
            ts(f"  Step {snum}: OK  ({len(files)} files)")

    # ── ZSH metric summary ────────────────────────────────────────
    cmp_path = os.path.join(OUT_DIR, "zsh_improved_comparison.csv")
    if os.path.exists(cmp_path):
        try:
            import pandas as pd
            cmp = pd.read_csv(cmp_path)
            ts("\nZSH vs K-Means results (from zsh_improved_comparison.csv):")
            for _, row in cmp.iterrows():
                better = "✓" if row.get("ZSH_Better", False) else "✗"
                ts(f"  {better}  {row['Metric']:<25} "
                   f"ZSH={row['Improved_ZSH']:.4f}  "
                   f"KM={row['Baseline_KMeans']:.4f}  "
                   f"Δ={row['Improvement_Pct']:+.1f}%")
        except Exception:
            pass

    ts("=" * 65)
    ts("SYNC: " + ("✓ ALL OK" if all_ok else "⚠ ISSUES FOUND — see above"))
    ts("=" * 65)
    return all_ok


def cleanup_stale():
    """Remove stale and intermediate output files (see STALE_FILES list)."""
    ts("\n" + "=" * 65)
    ts("CLEANUP — Removing stale / intermediate output files")
    ts("=" * 65)

    deleted_count = 0
    deleted_bytes = 0

    for fname in STALE_FILES:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            try:
                size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                if os.path.isdir(fpath):
                    shutil.rmtree(fpath)   # directory (e.g. file named "0")
                else:
                    os.remove(fpath)
                deleted_count += 1
                deleted_bytes += size
                ts(f"  DELETED  {fname:<45}  ({size/1e6:.1f} MB)")
            except Exception as exc:
                ts(f"  SKIPPED  {fname}  ({exc})")

    # Delete all stale .ckpt5_*.done files from the old
    # Step_5_ZSH_Clustering.py — COMPARISON.py is the new Step 5
    for f in Path(OUT_DIR).glob(".ckpt5_*.done"):
        f.unlink()
        deleted_count += 1
        ts(f"  DELETED  {f.name}  (stale Step_5_ZSH checkpoint)")

    # Remove COMPARISON.py if it still exists — Step_5_ZSH_Clustering.py
    # is now the canonical Step 5 (same algorithm, correct filename).
    old_comparison = Path(WORK_DIR) / "COMPARISON.py"
    if old_comparison.exists():
        old_comparison.unlink()
        deleted_count += 1
        ts(f"  DELETED  COMPARISON.py  (renamed → Step_5_ZSH_Clustering.py)")

    if deleted_count == 0:
        ts("  Nothing to delete — output folder is already clean.")
    else:
        freed_mb = deleted_bytes / 1e6
        ts(f"\n  Removed {deleted_count} files  ({freed_mb:.1f} MB freed)")

    ts("=" * 65)


def run_step(step_num: int, label: str, script_name: str) -> bool:
    """
    Launch a pipeline script as a subprocess.
    Output streams to the terminal in real-time (no buffering).
    Returns True if the script exited with code 0.
    """
    script_path = os.path.join(WORK_DIR, script_name)
    ts("\n" + "═" * 65)
    ts(f"STEP {step_num}  —  {label}")
    ts(f"Script : {script_name}")
    ts(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ts("═" * 65)

    t_start = time.time()

    # subprocess inherits stdout/stderr → output visible in terminal
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=WORK_DIR,
    )

    elapsed = time.time() - t_start

    if result.returncode != 0:
        ts(f"\n  ✗  Step {step_num} FAILED  (exit code {result.returncode})")
        return False

    ts(f"\n  ✓  Step {step_num} finished in {elapsed/60:.1f} min")
    return True


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts("=" * 65)
    ts("BITCOIN TRANSACTION CLUSTERING  —  MASTER PIPELINE RUNNER")
    ts(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ts(f"Work dir: {WORK_DIR}")
    ts(f"Output  : {OUT_DIR}")
    ts("=" * 65)

    # ── Quick pre-flight: which steps are already done? ──────────
    ts("\nPre-flight checkpoint status:")
    for snum, slabel, sscript in STEPS:
        if snum == 5:
            done = step5_is_done()
        elif snum == 6:
            done = bool(list(Path(OUT_DIR).glob(".ckpt6_results_summary.done")))
        else:
            done = bool(list(Path(OUT_DIR).glob(f".ckpt{snum}_*.done")))
        status = "DONE (will skip)" if done else "PENDING (will run)"
        ts(f"  Step {snum} [{slabel}]: {status}")

    ts("")
    wall_start = time.time()
    success    = True

    for step_num, label, script in STEPS:

        # ── Skip if already complete ──────────────────────────────
        # Each step is considered done when ALL its required output files
        # exist.  This prevents re-launching long steps (e.g. UMAP takes
        # 1-3 hours) when their outputs are already on disk.
        if step_num == 5:
            already_done = step5_is_done()
        else:
            required = REQUIRED_AFTER_STEP.get(step_num, [])
            already_done = bool(required) and all(
                os.path.exists(os.path.join(OUT_DIR, f)) for f in required
            )

        if already_done:
            ts(f"\nSTEP {step_num} [{label}]: SKIPPED — all required outputs present")
            if step_num == 5:
                # Still bridge if Step_6 standard filenames are missing
                fl = os.path.join(OUT_DIR, "final_labels.npy")
                if not os.path.exists(fl):
                    ts("  final_labels.npy missing — running bridge ...")
                    bridge_step5_outputs()
            continue

        # ── If Step_5 is about to re-run, clear Step_6 checkpoints ──
        # (Step_6 figures depend on Step_5 cluster assignments)
        if step_num == 5:
            stale_ckpt6 = list(Path(OUT_DIR).glob(".ckpt6_*.done"))
            if stale_ckpt6:
                ts(f"\n  Clearing {len(stale_ckpt6)} stale Step_6 checkpoint(s) "
                   f"because Step_5 is re-running ...")
                for f in stale_ckpt6:
                    f.unlink()

        # ── Run the step ──────────────────────────────────────────
        ok = run_step(step_num, label, script)
        if not ok:
            ts(f"\n  PIPELINE ABORTED at Step {step_num} — fix errors above and re-run")
            success = False
            break

        # ── Bridge Step_5 outputs to Step_6 expected file names ───
        if step_num == 5:
            bridge_step5_outputs()

        # ── Post-step file presence check ─────────────────────────
        missing = [
            f for f in REQUIRED_AFTER_STEP.get(step_num, [])
            if not os.path.exists(os.path.join(OUT_DIR, f))
        ]
        if missing:
            ts(f"\n  ⚠  WARNING: expected outputs missing after Step {step_num}:")
            for mf in missing:
                ts(f"     {mf}")

    # ── Sync verification ─────────────────────────────────────────
    if success:
        sync_ok = verify_sync()
        if not sync_ok:
            ts("\n⚠  Sync issues detected — inspect warnings above before use.")

    # ── Cleanup stale / intermediate files ────────────────────────
    if success:
        cleanup_stale()

    # ── Final summary ─────────────────────────────────────────────
    total_min = (time.time() - wall_start) / 60
    ts("\n" + "=" * 65)
    status_str = "✓ COMPLETE" if success else "✗ FAILED"
    ts(f"PIPELINE {status_str}")
    ts(f"Total wall-clock time : {total_min:.1f} min")
    ts(f"Runner log saved to   : {_log_path}")
    ts("=" * 65)
