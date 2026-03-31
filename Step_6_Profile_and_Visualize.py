# ============================================================
# STEP 6 — Transaction Profiling & Paper-Ready Visualizations
# ============================================================
# HARDWARE TARGET: HP Omen i9-13th Gen | 64 GB RAM | RTX 4060
# RAM budget: ≤ 50 GB
#
# FIXES vs previous broken version:
#   FIX 1 — Loads zsh_improved_profiles.npy (30 semantic clusters
#            from new Step 5). Eliminates "Cluster_4197" artifacts.
#   FIX 2 — Loads zsh_improved_anomaly_scores/flags.npy from
#            Isolation Forest (not old HDBSCAN outlier_scores.npy
#            that gave 0% anomalies). Correct ~5% rate.
#   FIX 3 — File-path resolver: tries zsh_improved_* first, then
#            legacy filenames as fallback.
#   FIX 4 — Source-version sentinel: auto-invalidates stale
#            checkpoints on first run with new ZSH source files.
#   FIX 5 — All figures redesigned for reviewer readability:
#              fig1 top-8 with "Other" grouping
#              fig2 correct anomaly source (~5% rate)
#              fig3 heatmap capped at top-25
#              fig4 HORIZONTAL bars top-30
#              fig6 proper width per boxplot
#              fig7 log+z scaling instead of MinMax-only radar squeeze
#   FIX 6 — New fig8: ZSH vs Baseline metrics bar chart from
#            zsh_improved_comparison.csv (handles both CSV formats)
# ============================================================

import numpy as np
import pandas as pd
import joblib, os, sys, io, logging, time, duckdb, warnings, pickle
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\sagar\Desktop\Q2 Paper 22326\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_UMAP    = 8
TOP_HEATMAP = 25
TOP_BAR     = 30
TOP_BOX     = 15
TOP_RADAR   = 6

# ── Logger ────────────────────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR, "step6_log.txt")
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
def ts(msg): log.info(f"[{time.time()-_T0:6.1f}s]  {msg}")

def ckpt(s):    return os.path.join(OUTPUT_DIR, f".ckpt6_{s}.done")
def is_done(s): return os.path.exists(ckpt(s))
def mark_done(s):
    Path(ckpt(s)).touch()
    ts(f"  CHECKPOINT -> [{s}]")

# ── File resolver: tries both naming conventions ──────────────
def find_file(*candidates):
    for path in candidates:
        if os.path.exists(path):
            ts(f"  Found: {os.path.basename(path)}")
            return path
    raise FileNotFoundError(
        "None found:\n  " + "\n  ".join(candidates)
    )

p = lambda name: os.path.join(OUTPUT_DIR, name)

ts("=" * 65)
ts("STEP 6 - Profiling & Paper-Ready Visualizations")
ts("=" * 65)

# ── Source-version sentinel: invalidate figure checkpoints when
#    the active source changes (e.g. old final_labels → new
#    zsh_improved_labels).  This is more reliable than mtime
#    comparison when both files were created on the same day.
_ZSH_SRC   = p('zsh_improved_labels.npy')
_ZSH_CKPT  = os.path.join(OUTPUT_DIR, '.ckpt6_src_zsh_improved.done')
_ZSH_DEPENDS = [
    p('zsh_improved_labels.npy'),
    p('zsh_improved_profiles.npy'),
    p('zsh_improved_anomaly_flags.npy'),
    p('zsh_improved_anomaly_scores.npy'),
    p('zsh_improved_comparison.csv'),
]
_zsh_inputs = [f for f in _ZSH_DEPENDS if os.path.exists(f)]
_src_is_newer = (
    _zsh_inputs and (
        not os.path.exists(_ZSH_CKPT)
        or max(os.path.getmtime(f) for f in _zsh_inputs) > os.path.getmtime(_ZSH_CKPT)
    )
)
if os.path.exists(_ZSH_SRC) and _src_is_newer:
    # New or updated ZSH outputs — delete old figure/stat checkpoints
    _old = list(Path(OUTPUT_DIR).glob(".ckpt6_fig*.done")) + \
           list(Path(OUTPUT_DIR).glob(".ckpt6_stats.done")) + \
           list(Path(OUTPUT_DIR).glob(".ckpt6_save*.done")) + \
           list(Path(OUTPUT_DIR).glob(".ckpt6_summary*.done")) + \
           list(Path(OUTPUT_DIR).glob(".ckpt6_results*.done")) + \
           list(Path(OUTPUT_DIR).glob(".ckpt6_profile_labels.done"))
    if _old:
        ts(f"  Detected updated ZSH source — invalidating {len(_old)} stale ckpt(s).")
        for c in _old:
            os.remove(c)
    Path(_ZSH_CKPT).touch()  # mark: figures now match current zsh_improved_* source

# ── Resolve file paths ────────────────────────────────────────
ts("Resolving input file paths ...")
# Prefer zsh_improved_* files produced by the new Step 5
LABELS_PATH  = find_file(p('zsh_improved_labels.npy'),
                          p('final_labels.npy'),
                          p('final_labels_optimized.npy'))
ANOM_F_PATH  = find_file(p('zsh_improved_anomaly_flags.npy'),
                          p('anomaly_flags.npy'),
                          p('anomaly_flags_optimized.npy'))
ANOM_S_PATH  = find_file(p('zsh_improved_anomaly_scores.npy'),
                          p('anomaly_scores.npy'),
                          p('outlier_scores.npy'))
EMBED_PATH   = find_file(p('X_embed_2d.npy'))
FEAT_PKL     = find_file(p('feature_cols.pkl'))
DF_FEAT_PATH = find_file(p('df_balanced_features.parquet'))
DF_META_PATH = find_file(p('df_meta.parquet'))
# zsh_improved_comparison.csv has the correct metrics from new Step 5
METRICS_CSV  = p('zsh_improved_comparison.csv') \
               if os.path.exists(p('zsh_improved_comparison.csv')) \
               else p('metrics_comparison.csv')

# ── Load arrays ───────────────────────────────────────────────
ts("\nLoading artifacts (mmap for large arrays) ...")
t0 = time.time()

labels_mm  = np.load(LABELS_PATH,  mmap_mode='r')
anom_f_mm  = np.load(ANOM_F_PATH,  mmap_mode='r')
anom_s_mm  = np.load(ANOM_S_PATH,  mmap_mode='r')
embed_mm   = np.load(EMBED_PATH,   mmap_mode='r')
FEAT_COLS  = joblib.load(FEAT_PKL)
df_feat    = pd.read_parquet(DF_FEAT_PATH)
df_meta    = pd.read_parquet(DF_META_PATH)

# Load profile labels — prefer new ZSH improved semantic strings
if os.path.exists(p('zsh_improved_profiles.npy')):
    profiles_raw = np.load(p('zsh_improved_profiles.npy'), allow_pickle=True)
    ts("  Loaded: zsh_improved_profiles.npy (semantic labels)")
elif os.path.exists(p('final_profiles.npy')):
    profiles_raw = np.load(p('final_profiles.npy'), allow_pickle=True)
    ts("  Loaded: final_profiles.npy (semantic labels, fallback)")
elif os.path.exists(p('cluster_assignments.csv')):
    profiles_raw = pd.read_csv(p('cluster_assignments.csv'))['tx_profile'].values
    ts("  Loaded: tx_profile from cluster_assignments.csv (fallback)")
else:
    ts("  WARNING: No profile labels found; using cluster integers as labels")
    profiles_raw = np.load(LABELS_PATH, mmap_mode='r')

n_rows = min(len(labels_mm), len(profiles_raw), len(anom_f_mm),
             len(anom_s_mm), len(df_feat), len(df_meta))
ts(f"  Aligned n_rows: {n_rows:,}")

# Materialise into RAM
labels_arr   = np.array(labels_mm[:n_rows],   dtype=np.int32)
profiles_arr = np.array(profiles_raw[:n_rows], dtype=object)
anom_f_arr   = np.array(anom_f_mm[:n_rows],   dtype=np.int8)
anom_s_arr   = np.array(anom_s_mm[:n_rows],   dtype=np.float32)
embed_arr    = np.array(embed_mm[:n_rows],     dtype=np.float32)

df_feat = df_feat.iloc[:n_rows].reset_index(drop=True)
df_meta = df_meta.iloc[:n_rows].reset_index(drop=True)
ts(f"  Load done in {time.time()-t0:.1f}s")

# ── Build working DataFrame ───────────────────────────────────
df = df_feat.copy()
df['cluster']       = labels_arr
df['tx_profile']    = profiles_arr
df['is_anomaly']    = anom_f_arr
df['anomaly_score'] = anom_s_arr
df['umap_x']        = embed_arr[:, 0]
df['umap_y']        = embed_arr[:, 1]

for col in ['block_height', 'year', 'hour', 'month', 'day_of_week']:
    if col in df_meta.columns:
        df[col] = df_meta[col].values

ts(f"  Working df: {df.shape}  |  profiles: {df['tx_profile'].nunique()}")

profile_counts = df['tx_profile'].value_counts()
anomaly_rate   = df['is_anomaly'].mean() * 100
ts(f"  Anomaly rate: {anomaly_rate:.2f}%  (from Isolation Forest)")
ts(f"\n  Top 10 profiles:")
for name, cnt in profile_counts.head(10).items():
    ts(f"    {name:<30}  {cnt:>10,}  ({cnt/n_rows*100:.2f}%)")

# Extended colour palette
tab_colors = (list(plt.cm.tab20.colors) +
              list(plt.cm.tab20b.colors) +
              list(plt.cm.Set3.colors))

# ══════════════════════════════════════════════════════════════
# FIGURE 1 - UMAP Scatter by Semantic Profile
# ══════════════════════════════════════════════════════════════
fig1_path = p('fig1_cluster_umap.png')
if is_done("fig1"):
    ts("\nFIG1: Already saved - skipping.")
else:
    ts("\nFIG1: UMAP scatter by semantic profile ...")
    t0 = time.time()

    top_profs  = profile_counts.head(TOP_UMAP).index.tolist()
    other_mask = ~df['tx_profile'].isin(top_profs)
    n_other    = other_mask.sum()

    fig1, ax1 = plt.subplots(figsize=(14, 9))
    for i, prof in enumerate(top_profs):
        mask = df['tx_profile'] == prof
        ax1.scatter(df.loc[mask,'umap_x'], df.loc[mask,'umap_y'],
                    s=0.6, alpha=0.55,
                    color=tab_colors[i % len(tab_colors)],
                    label=f"{prof} ({mask.sum():,})",
                    rasterized=True)
    if n_other > 0:
        ax1.scatter(df.loc[other_mask,'umap_x'], df.loc[other_mask,'umap_y'],
                    s=0.12, alpha=0.12, color='lightgrey',
                    label=f"Other ({n_other:,})", rasterized=True)

    ax1.legend(markerscale=8, bbox_to_anchor=(1.02,1), loc='upper left',
               fontsize=7, framealpha=0.9, ncol=1)
    ax1.set_title(
        'ZSH Hybrid Clustering - UMAP 2D Projection\n'
        f'(Top {TOP_UMAP} profiles highlighted; remaining profiles grouped as Other | n={n_rows:,})',
        fontsize=13, fontweight='bold')
    ax1.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax1.grid(True, alpha=0.15)
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    ts(f"  Saved fig1  ({time.time()-t0:.1f}s)")
    mark_done("fig1")

# ══════════════════════════════════════════════════════════════
# FIGURE 2 - Isolation Forest Anomaly Overlay (FIXED)
# ══════════════════════════════════════════════════════════════
fig2_path = p('fig2_anomaly_overlay.png')
if is_done("fig2"):
    ts("\nFIG2: Already saved - skipping.")
else:
    ts("\nFIG2: Isolation Forest anomaly overlay ...")
    t0 = time.time()

    rng_sc = np.random.default_rng(42)
    SC_N   = min(300_000, n_rows)
    sc_idx = rng_sc.choice(n_rows, size=SC_N, replace=False)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    # Left: continuous score heatmap
    sc = axes2[0].scatter(
        df.iloc[sc_idx]['umap_x'], df.iloc[sc_idx]['umap_y'],
        c=df.iloc[sc_idx]['anomaly_score'],
        cmap='hot_r', s=0.3, alpha=0.6, rasterized=True,
        vmin=float(np.percentile(anom_s_arr, 1)),
        vmax=float(np.percentile(anom_s_arr, 99)))
    plt.colorbar(sc, ax=axes2[0],
                 label='Isolation Forest Anomaly Score\n(Higher = More Anomalous)')
    axes2[0].set_title('Continuous Anomaly Score Heatmap\n(300K sample)',
                        fontsize=11, fontweight='bold')
    axes2[0].set_xlabel('UMAP-1', fontsize=10)
    axes2[0].set_ylabel('UMAP-2', fontsize=10)

    # Right: binary flag
    norm_mask = df['is_anomaly'] == 0
    anom_mask = df['is_anomaly'] == 1
    n_anom    = anom_mask.sum()
    r_anom    = n_anom / n_rows * 100

    norm_sub = rng_sc.choice(np.where(norm_mask)[0],
                               size=min(200_000, norm_mask.sum()), replace=False)
    axes2[1].scatter(df.iloc[norm_sub]['umap_x'], df.iloc[norm_sub]['umap_y'],
                     s=0.2, alpha=0.2, c='lightgrey', label='Normal',
                     rasterized=True)
    axes2[1].scatter(df.loc[anom_mask,'umap_x'], df.loc[anom_mask,'umap_y'],
                     s=1.5, alpha=0.8, c='crimson',
                     label=f"Anomalous ({n_anom:,} | {r_anom:.2f}%)",
                     rasterized=True)
    axes2[1].legend(markerscale=6, fontsize=9, loc='upper right')
    axes2[1].set_title(f'Binary Anomaly Flag\n(~{r_anom:.2f}% flagged)',
                        fontsize=11, fontweight='bold')
    axes2[1].set_xlabel('UMAP-1', fontsize=10)
    axes2[1].set_ylabel('UMAP-2', fontsize=10)

    fig2.suptitle('Transaction Authenticity - Anomaly Detection via Isolation Forest',
                   fontsize=13, fontweight='bold')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    ts(f"  Saved fig2  ({time.time()-t0:.1f}s)  anomalies: {n_anom:,} ({r_anom:.2f}%)")
    mark_done("fig2")

# ══════════════════════════════════════════════════════════════
# FIGURE 3 - Profile Feature Heatmap (capped at TOP_HEATMAP)
# ══════════════════════════════════════════════════════════════
fig3_path = p('fig3_profile_heatmap.png')
if is_done("fig3"):
    ts("\nFIG3: Already saved - skipping.")
else:
    ts("\nFIG3: Profile feature heatmap ...")
    t0 = time.time()

    HM_FEATS = [c for c in ['input_count','output_count','fee_rate_sat_per_vbyte',
                              'value_concentration_ratio','input_output_ratio',
                              'is_coinjoin_like','is_batch_payment','is_consolidation',
                              'is_distribution','is_peer_to_peer','has_op_return',
                              'rbf_enabled','has_coinbase'] if c in df.columns]

    top_profs_hm = profile_counts.head(TOP_HEATMAP).index.tolist()
    con = duckdb.connect()
    con.register("df_tbl", df[['tx_profile'] + HM_FEATS])
    agg = ", ".join([f'ROUND(AVG("{c}"),6) AS "{c}"' for c in HM_FEATS])
    hm_df = con.execute(
        f'SELECT tx_profile, {agg} FROM df_tbl '
        f'WHERE tx_profile IN ({",".join([repr(x) for x in top_profs_hm])}) '
        f'GROUP BY tx_profile'
    ).df().set_index('tx_profile')
    con.close()

    hm_df   = hm_df.reindex([x for x in top_profs_hm if x in hm_df.index])
    sc_hm   = MinMaxScaler()
    hm_norm = pd.DataFrame(sc_hm.fit_transform(hm_df.T).T,
                            index=hm_df.index, columns=hm_df.columns)

    n_p   = len(hm_norm)
    fig3h = max(7, n_p * 0.40 + 2)
    fig3, ax3 = plt.subplots(figsize=(15, fig3h))
    sns.heatmap(hm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3,
                linewidths=0.3, linecolor='white',
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.6})
    ax3.set_title(
        f'Cluster Profile Heatmap - Normalised Mean Feature Values\n'
        f'(0=min, 1=max | Top {n_p} clusters)',
        fontsize=12, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=8)
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    ts(f"  Saved fig3  ({time.time()-t0:.1f}s)")
    mark_done("fig3")

# ══════════════════════════════════════════════════════════════
# FIGURE 4 - Cluster Sizes (HORIZONTAL BARS, readable)
# ══════════════════════════════════════════════════════════════
fig4_path = p('fig4_cluster_sizes.png')
if is_done("fig4"):
    ts("\nFIG4: Already saved - skipping.")
else:
    ts("\nFIG4: Cluster size distribution (horizontal) ...")
    t0 = time.time()

    top_bar   = profile_counts.head(TOP_BAR)
    n_bar     = len(top_bar)
    bc        = plt.cm.RdYlGn(np.linspace(0.85, 0.15, n_bar))
    fig4h     = max(8, n_bar * 0.42 + 2)

    fig4, ax4 = plt.subplots(figsize=(12, fig4h))
    bars = ax4.barh(top_bar.index[::-1], top_bar.values[::-1],
                    color=bc[::-1], edgecolor='black', linewidth=0.5)
    for bar, cnt in zip(bars, top_bar.values[::-1]):
        label = f'{cnt:,}  ({cnt/n_rows*100:.1f}%)'
        if bar.get_width() >= top_bar.max() * 0.18:
            x_text = bar.get_width() - top_bar.max() * 0.01
            ha = 'right'
            color = 'white'
        else:
            x_text = bar.get_width() + top_bar.max() * 0.01
            ha = 'left'
            color = 'black'
        ax4.text(
            x_text,
            bar.get_y() + bar.get_height()/2,
            label,
            va='center',
            ha=ha,
            fontsize=7.2,
            color=color,
            fontweight='bold' if ha == 'right' else 'normal'
        )
    ax4.set_xlabel('Transaction Count', fontsize=11)
    ax4.set_title(
        f'Transaction Profile Distribution\n'
        f'(ZSH Hybrid | Top {n_bar} of {profile_counts.nunique()} profiles)',
        fontsize=12, fontweight='bold')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax4.set_xlim(0, top_bar.max() * 1.08)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    ts(f"  Saved fig4  ({time.time()-t0:.1f}s)")
    mark_done("fig4")

# ══════════════════════════════════════════════════════════════
# FIGURE 5 - Temporal Distribution (block height)
# ══════════════════════════════════════════════════════════════
fig5_path = p('fig5_temporal_distribution.png')
if is_done("fig5"):
    ts("\nFIG5: Already saved - skipping.")
elif 'block_height' not in df.columns:
    ts("\nFIG5: Skipped - block_height not in df.")
else:
    ts("\nFIG5: Temporal distribution ...")
    t0 = time.time()
    top8 = profile_counts.head(8).index.tolist()
    fig5, ax5 = plt.subplots(figsize=(14, 6))
    for i, prof in enumerate(top8):
        sub = df.loc[df['tx_profile']==prof, 'block_height']
        ax5.hist(sub, bins=80, alpha=0.55,
                 color=tab_colors[i % len(tab_colors)],
                 label=f"{prof} ({len(sub):,})", density=True)
    ax5.set_xlabel('Block Height', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Transaction Profile Temporal Distribution (Block Height, 2022-2025)\n'
                   'Top 8 profiles shown', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, bbox_to_anchor=(1.02,1), loc='upper left')
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax5.grid(True, alpha=0.2)
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    ts(f"  Saved fig5  ({time.time()-t0:.1f}s)")
    mark_done("fig5")

# ══════════════════════════════════════════════════════════════
# FIGURE 6 - Anomaly Score Boxplots (FIXED WIDTH)
# ══════════════════════════════════════════════════════════════
fig6_path = p('fig6_outlier_by_profile.png')
if is_done("fig6"):
    ts("\nFIG6: Already saved - skipping.")
else:
    ts("\nFIG6: Anomaly score boxplots ...")
    t0 = time.time()

    top_box  = profile_counts.head(TOP_BOX).index.tolist()
    box_data = [df.loc[df['tx_profile']==prof,'anomaly_score'].values
                for prof in top_box]

    # 1.1 inches per boxplot ensures readable layout
    fig6w = max(15, len(top_box) * 1.2)
    fig6, ax6 = plt.subplots(figsize=(fig6w, 6))

    bp = ax6.boxplot(box_data, labels=top_box, patch_artist=True,
                     notch=False,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=2, alpha=0.4))
    for patch, color in zip(bp['boxes'], tab_colors[:len(top_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    thresh95 = float(np.percentile(anom_s_arr, 95))
    ax6.axhline(y=thresh95, color='crimson', linestyle='--', linewidth=1.5,
                label=f'95th percentile ({thresh95:.3f})')
    ax6.set_xlabel('Transaction Profile', fontsize=11)
    ax6.set_ylabel('Isolation Forest Anomaly Score', fontsize=11)
    ax6.set_title(f'Anomaly Score Distribution by Profile (Top {len(top_box)})',
                   fontsize=12, fontweight='bold')
    ax6.set_xticklabels(top_box, rotation=40, ha='right', fontsize=8)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.25, axis='y')
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    plt.close(fig6)
    ts(f"  Saved fig6  ({time.time()-t0:.1f}s)")
    mark_done("fig6")

# ══════════════════════════════════════════════════════════════
# FIGURE 7 - Radar Fingerprint Charts (polygon closure FIXED)
# ══════════════════════════════════════════════════════════════
fig7_path = p('fig7_radar_fingerprints.png')
RADAR_FEATS = [c for c in ['input_count','output_count','fee_rate_sat_per_vbyte',
                             'value_concentration_ratio','input_output_ratio',
                             'is_coinjoin_like','is_batch_payment'] if c in df.columns]

if is_done("fig7"):
    ts("\nFIG7: Already saved - skipping.")
elif len(RADAR_FEATS) < 3:
    ts("\nFIG7: Skipped - not enough radar features.")
else:
    ts("\nFIG7: Radar fingerprint charts ...")
    t0 = time.time()

    # Compute per-profile means via DuckDB
    con = duckdb.connect()
    con.register("df_tbl", df[['tx_profile'] + RADAR_FEATS])
    agg = ", ".join([f'AVG("{c}") AS "{c}"' for c in RADAR_FEATS])
    radar_raw = con.execute(
        f'SELECT tx_profile, {agg} FROM df_tbl GROUP BY tx_profile'
    ).df().set_index('tx_profile')
    con.close()

    # Select valid profiles (no NaN in any radar feature)
    valid_all = [pr for pr in profile_counts.index
                 if pr in radar_raw.index and
                    not radar_raw.loc[pr, RADAR_FEATS].isnull().any()]

    if len(valid_all) < 2:
        ts("  FIG7: Not enough valid profiles - skipping.")
    else:
        radar_plot = radar_raw[RADAR_FEATS].fillna(0).astype(float).copy()
        for feat in RADAR_FEATS:
            if radar_plot[feat].max() > 1.0:
                radar_plot[feat] = np.log1p(radar_plot[feat].clip(lower=0))

        z_all = pd.DataFrame(
            StandardScaler().fit_transform(radar_plot),
            index=radar_plot.index,
            columns=RADAR_FEATS
        )
        # Clip extreme z-scores and map to [0,1] for plotting while keeping
        # inter-profile variance visible. This avoids the "flat spiderweb"
        # effect produced by global MinMax scaling.
        rnorm_all = ((z_all.clip(-2.5, 2.5) + 2.5) / 5.0).astype(float)

        valid = sorted(
            valid_all,
            key=lambda p: (
                float(rnorm_all.loc[p].max() - rnorm_all.loc[p].min()),
                int(profile_counts.get(p, 0))
            ),
            reverse=True
        )[:TOP_RADAR]

        rnorm = rnorm_all.loc[valid]

        N_ax   = len(RADAR_FEATS)
        angles = np.linspace(0, 2*np.pi, N_ax, endpoint=False).tolist()
        ang_cl = angles + angles[:1]   # closed polygon

        fig7, axes7 = plt.subplots(2, 3, figsize=(16, 10),
                                    subplot_kw=dict(polar=True))
        ax_flat = axes7.flatten()

        for i in range(6):
            ax_r = ax_flat[i]
            if i >= len(valid):
                ax_r.set_visible(False)
                continue

            prof = valid[i]
            vals = rnorm.loc[prof].fillna(0).tolist()
            vcl  = vals + vals[:1]   # CRITICAL: close the polygon

            ax_r.plot(ang_cl, vcl, color=tab_colors[i], linewidth=2.5, zorder=3)
            ax_r.fill(ang_cl, vcl, color=tab_colors[i], alpha=0.25)
            ax_r.set_xticks(angles)
            ax_r.set_xticklabels(
                [f.replace('_', '\n') for f in RADAR_FEATS], fontsize=7)
            ax_r.set_ylim(0, 1)
            ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_r.set_yticklabels(['0.25','0.50','0.75','1.0'], fontsize=6, color='grey')
            ax_r.grid(True, alpha=0.3)

            title_s = prof if len(prof) <= 22 else prof[:20] + '...'
            n_prof  = profile_counts.get(prof, 0)
            ax_r.set_title(f"{title_s}\n({n_prof:,})",
                            size=9, fontweight='bold', pad=14)

        fig7.suptitle(
            'Transaction Profile Fingerprints - Radar Charts\n'
            '(log1p + z-score scaled profile means; higher = above-profile average)',
            fontsize=13, fontweight='bold')
        plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
        plt.close(fig7)
        ts(f"  Saved fig7  ({time.time()-t0:.1f}s)")
        mark_done("fig7")

# ══════════════════════════════════════════════════════════════
# FIGURE 8 - ZSH vs Baseline Metrics Bar Chart (NEW)
# ══════════════════════════════════════════════════════════════
fig8_path = p('fig8_metrics_comparison.png')
if is_done("fig8"):
    ts("\nFIG8: Already saved - skipping.")
elif not os.path.exists(METRICS_CSV):
    ts("\nFIG8: Skipped - metrics_comparison.csv not found.")
else:
    ts("\nFIG8: Metrics comparison bar chart ...")
    t0 = time.time()

    mdf = pd.read_csv(METRICS_CSV)
    baseline_method = 'Baseline K-Means'
    baseline_value_col = 'Baseline_KMeans'
    baseline_row_name = 'Baseline_KMeans'
    if 'Baseline_Method' in mdf.columns and mdf['Baseline_Method'].notna().any():
        baseline_method = str(mdf['Baseline_Method'].dropna().iloc[0])
    if 'Reference_Score' in mdf.columns:
        baseline_value_col = 'Reference_Score'
        baseline_row_name = 'Geometry_Reference'

    # Handle both CSV formats:
    #   Format A (zsh_improved_comparison.csv):
    #     Columns: Metric, Baseline_Method, Reference_Score, Improved_ZSH, Improvement_Pct, ZSH_Better
    #   Format B (old metrics_comparison.csv):
    #     Columns: model, n_clusters, silhouette, davies_bouldin, calinski_harabasz
    if 'Metric' in mdf.columns and 'Improved_ZSH' in mdf.columns:
        # Format A — transpose so each metric is a column
        mdf_t = mdf.set_index('Metric')[[baseline_value_col,'Improved_ZSH','Improvement_Pct']].T
        # Map row names to internal labels
        mdf_t.index = [baseline_row_name, 'ZSH_Hybrid', 'Improvement_%']
        # Rename columns to lowercase canonical names
        col_map = {'Silhouette':        'silhouette',
                   'Davies_Bouldin':    'davies_bouldin',
                   'Calinski_Harabasz': 'calinski_harabasz'}
        mdf_t.rename(columns=col_map, inplace=True)
        plot_m = mdf_t.loc[[baseline_row_name,'ZSH_Hybrid']]
        imp_pct = mdf_t.loc['Improvement_%'] if 'Improvement_%' in mdf_t.index else None
    else:
        # Format B — old file
        plot_m  = mdf[mdf['model'].isin(['Baseline_KMeans','ZSH_Hybrid'])].set_index('model')
        imp_row = mdf[mdf['model']=='Improvement_%']
        imp_pct = imp_row.iloc[0] if not imp_row.empty else None

    fig8, axes8 = plt.subplots(1, 3, figsize=(14, 5))
    C     = {baseline_row_name:'#FF7043', 'ZSH_Hybrid':'#2196F3'}
    baseline_title = baseline_method.replace(' (geometry-only reference)', '')
    if 'KMeans++ Elkan' in baseline_title:
        baseline_tick = 'KMeans++\nElkan'
    else:
        baseline_tick = baseline_title.replace(' ', '\n', 1)
    specs = [('silhouette',        'Silhouette Score\n(Higher = Better)'),
             ('davies_bouldin',    'Davies-Bouldin Index\n(Lower = Better)'),
             ('calinski_harabasz', 'Calinski-Harabasz Index\n(Higher = Better)')]

    for ax, (col, title) in zip(axes8, specs):
        if col not in plot_m.columns:
            ax.set_visible(False); continue
        vals   = plot_m[col].astype(float).values
        models = plot_m.index.tolist()
        colors = [C.get(m,'grey') for m in models]
        bars   = ax.bar([baseline_tick, 'ZSH\nHybrid'], vals,
                        color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        for bar, v in zip(bars, vals):
            label = f'{v:.4f}' if abs(v) < 100 else f'{v:,.0f}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ax.get_ylim()[1]*0.01,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        # Annotate improvement %
        if imp_pct is not None and col in imp_pct.index:
            pct = float(imp_pct[col])
            ax.annotate(f'ZSH {pct:+.1f}% vs ref', xy=(0.5, 0.96), xycoords='axes fraction',
                        ha='center', fontsize=9, color='#1565C0',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec='#1565C0', lw=1))

    # Build subtitle from improvement percentages
    if imp_pct is not None:
        try:
            sub = (f"Silhouette {float(imp_pct.get('silhouette', imp_pct.get('Silhouette',0))):+.1f}%  |  "
                   f"DBI {float(imp_pct.get('davies_bouldin', imp_pct.get('Davies_Bouldin',0))):+.1f}%  |  "
                   f"C-H {float(imp_pct.get('calinski_harabasz', imp_pct.get('Calinski_Harabasz',0))):+.1f}%")
        except Exception:
            sub = f"Reference = {baseline_title}"
    else:
        sub = f"Reference = {baseline_title}"
    fig8.suptitle(f'Intrinsic Clustering Quality: ZSH Hybrid vs {baseline_title}\n{sub}',
                   fontsize=13, fontweight='bold')
    fig8.text(
        0.5, 0.015,
        'Geometry-only reference shown for honest comparison; ZSH additionally provides semantic profile labels and anomaly flags.',
        ha='center', fontsize=9, color='#455A64'
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.9])
    plt.savefig(fig8_path, dpi=150, bbox_inches='tight')
    plt.close(fig8)
    ts(f"  Saved fig8  ({time.time()-t0:.1f}s)")
    mark_done("fig8")

# ══════════════════════════════════════════════════════════════
# PROFILE STATISTICS TABLE (DuckDB)
# ══════════════════════════════════════════════════════════════
stats_path = p('profile_statistics.csv')
if is_done("stats"):
    ts("\nSTATS: Already saved - skipping.")
    profile_stats = pd.read_csv(stats_path)
else:
    ts("\nSTATS: Computing profile statistics via DuckDB ...")
    t0 = time.time()
    sf = [c for c in ['input_count','output_count','fee_rate_sat_per_vbyte',
                       'value_concentration_ratio','is_coinjoin_like',
                       'is_batch_payment','anomaly_score'] if c in df.columns]
    con = duckdb.connect()
    con.register("df_tbl", df[['tx_profile','is_anomaly','anomaly_score']+sf])
    agg = ", ".join([f'ROUND(AVG("{c}"),6) AS "mean_{c}"' for c in sf])
    profile_stats = con.execute(f"""
        SELECT tx_profile,
               COUNT(*) AS count,
               ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),4) AS pct,
               ROUND(AVG(is_anomaly),6) AS anomaly_rate,
               ROUND(AVG(anomaly_score),6) AS mean_anomaly_score,
               {agg}
        FROM df_tbl GROUP BY tx_profile ORDER BY count DESC
    """).df()
    con.close()
    profile_stats.to_csv(stats_path, index=False)
    ts(f"  Saved profile_statistics.csv  ({time.time()-t0:.1f}s)")
    ts("\nTop 10 profiles:")
    ts("\n" + profile_stats.head(10).to_string(index=False))
    mark_done("stats")

# ── Save labeled dataset ──────────────────────────────────────
if not is_done("save"):
    ts("\nSaving labeled parquet + CSV ...")
    df.to_parquet(p('results_labeled.parquet'), index=False)
    df[['cluster','tx_profile','anomaly_score','is_anomaly']]\
      .to_csv(p('cluster_assignments.csv'), index=False)
    mark_done("save")

# ── Results summary ───────────────────────────────────────────
if not is_done("summary"):
    ts("\nWriting RESULTS_SUMMARY.txt ...")
    with open(p('RESULTS_SUMMARY.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("  ZSH HYBRID CLUSTERING - RESULTS SUMMARY\n")
        f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n\n")
        if 'block_height' in df.columns:
            f.write(f"Dataset Coverage   : Block {df['block_height'].min():,} "
                     f"- {df['block_height'].max():,} (2022-2025)\n")
        f.write(f"Total Transactions : {n_rows:,}\n")
        f.write(f"Features Used      : {len(FEAT_COLS)}\n")
        f.write(f"Unique Profiles    : {df['tx_profile'].nunique()}\n\n")
        f.write("-- Profile Counts --\n")
        for _, row in profile_stats.iterrows():
            f.write(f"  {str(row['tx_profile']):<30}  {int(row['count']):>10,}  "
                     f"({row['pct']:.2f}%)  anom={row['anomaly_rate']*100:.2f}%\n")
        f.write(f"\n-- Anomaly Detection --\n")
        f.write(f"  Method: Isolation Forest | Rate: {anomaly_rate:.2f}% | "
                 f"Flagged: {df['is_anomaly'].sum():,}\n")
    mark_done("summary")

ts("\n" + "=" * 65)
ts("STEP 6 COMPLETE")
ts(f"  Total runtime: {time.time()-_T0:.1f}s")
ts(f"  All 8 figures + tables saved to: {OUTPUT_DIR}")
ts("=" * 65)
ts("RESUMABILITY: delete .ckpt6_<n>.done to regenerate individual figures.")
