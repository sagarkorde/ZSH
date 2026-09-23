"""Build the article's tables (CSV + Markdown) and figures (PNG, 600 dpi) from results/.

Each builder runs only if its inputs exist, so the script can be re-run as experiments finish.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from zsh import plotstyle as ps  # noqa: E402
from zsh.config import CFG, OUT, RESULTS, results_dir  # noqa: E402
from zsh.data import L1_NAMES, L2_NAMES, L3_NAMES, TAG_BITS  # noqa: E402

import os  # noqa: E402

SFX = os.environ.get("ZSH_FIG_SUFFIX", "")  # "_smoke" to test builders on smoke outputs
TAB = results_dir("tables" + SFX)
FIG = results_dir("figures" + SFX)


def R(exp):
    """Results folder of an experiment (E0 has no smoke variant)."""
    return RESULTS / (exp if exp == "E0" else exp + SFX)
NICE = {
    "input_count": "Input count", "output_count": "Output count", "vsize": "Virtual size",
    "size": "Serialised size", "total_input_value": "Total input value", "fee": "Fee",
    "fee_rate_sat_per_vbyte": "Fee rate (sat/vB)", "fee_rate_sat_per_byte": "Fee rate (sat/B)",
    "avg_output_value": "Mean output value", "input_output_ratio": "Input/output value ratio",
    "has_op_return": "OP_RETURN output", "rbf_enabled": "RBF signalled",
}


def load_json(p):
    exp, rest = p.split("/", 1)
    p = R(exp) / rest
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def save_table(df, name, floatfmt=None):
    df.to_csv(TAB / f"{name}.csv", index=False)
    with open(TAB / f"{name}.md", "w", encoding="utf-8") as fh:
        fh.write(df.to_markdown(index=False, floatfmt=floatfmt or ".3g"))
    print(f"table {name}: {df.shape}")


def savefig(fig, name):
    # Pass dpi and bbox explicitly rather than relying on plotstyle.apply() having run:
    # F13_bench was once written by a run that never applied the style, so it inherited
    # the matplotlib defaults (100 dpi, no tight bounding box) and its rotated tick
    # labels and row labels were cut off at the canvas edge.
    out = FIG / f"{name}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    return out
    plt.close(fig)
    print(f"figure {name}")


# ---------------------------------------------------------------------------
# tables from E0 / E1
# ---------------------------------------------------------------------------
def table_data():
    splits = pd.read_csv(R("E0") / "splits.csv")
    rows = []
    names = {0: "Development (fit)", 1: "Test (2024)", 9: "Excluded (after test end)"}
    for _, r in splits.iterrows():
        rows.append({"Data": "Bitcoin sample [dataset]", "Part": names[int(r.split)],
                     "Period": f"{r.first_time[:10]} to {r.last_time[:10]}",
                     "Transactions": int(r.rows), "Blocks": int(r.blocks)})
    fm = load_json("E0/future_manifest.json")
    if fm:
        m = fm["months"]
        rows.append({"Data": "Prospective sample (this study)", "Part": "Future (2024-10 to 2026-08)",
                     "Period": f"{m[0]['month']} to {m[-1]['month']}",
                     "Transactions": fm["rows"], "Blocks": fm["blocks_with_page"]})
    e8 = load_json("E8/summary.json")
    if e8 and "data" in e8:
        for k, v in e8["data"].items():
            rows.append({"Data": "Elliptic [Weber2019]", "Part": k, "Period": v["period"],
                         "Transactions": v["rows"], "Blocks": v.get("labelled", "")})
    save_table(pd.DataFrame(rows), "T1_data")


def table_features():
    sel = load_json("E0/feature_selection.json")
    rows = []
    for f in sel["candidates"]:
        rows.append({"Feature": f, "Most frequent value share": sel["mode_share"].get(f),
                     "Kept": "yes" if f in sel["selected"] else "no",
                     "Reason if removed": sel["removed"].get(f, "")})
    for f in CFG["features"]["removed_annotation_flags"]:
        rows.append({"Feature": f, "Most frequent value share": None, "Kept": "no",
                     "Reason if removed": "count rule; used only as annotation (L1)"})
    w = R("E1") / "weights.csv"
    df = pd.DataFrame(rows)
    if w.exists():
        ww = pd.read_csv(w)[["feature", "rank", "mi", "weight"]].rename(columns={"feature": "Feature"})
        df = df.merge(ww, on="Feature", how="left")
    save_table(df, "T2_features")


def table_annotations():
    p = pd.read_csv(R("E0") / "annotation_prevalence.csv")
    p = p[p.split.isin([0, 1])]
    wide = p.pivot_table(index=["annotation", "label"], columns="split", values="share").reset_index()
    wide.columns = ["Annotation", "Label", "Share DEV", "Share TEST"]
    fut = OUT / "future" / "d4_future.parquet"
    if fut.exists():
        d = pd.read_parquet(fut)
        w = d.design_weight.to_numpy()
        vals = []
        for _, r in wide.iterrows():
            a, lab = r.Annotation, r.Label
            if a == "L1":
                m = d.L1.to_numpy() == L1_NAMES.index(lab)
            elif a == "L2":
                m = d.L2.to_numpy() == L2_NAMES.index(lab)
            elif a == "L3":
                m = d.L3.to_numpy() == L3_NAMES.index(lab)
            elif a == "L4":
                m = (d.tags.to_numpy() & TAG_BITS[lab]) > 0
            else:
                vals.append(np.nan)
                continue
            vals.append(float(np.average(m, weights=w)))
        wide["Share FUTURE (weighted)"] = vals
        extra = pd.DataFrame([{"Annotation": "L5", "Label": "equal-output CoinJoin",
                               "Share DEV": np.nan, "Share TEST": np.nan,
                               "Share FUTURE (weighted)": float(np.average(d.eocj, weights=w))}])
        wide = pd.concat([wide, extra], ignore_index=True)
    save_table(wide, "T3_annotations", floatfmt=".4f")


def table_profiles():
    dev = pd.read_csv(R("E1") / "profiles_dev.csv")
    test = pd.read_csv(R("E1") / "profiles_test.csv")[["profile", "share", "runes_share"]]
    test.columns = ["profile", "share_test", "runes_share_test"]
    t = dev.merge(test, on="profile", how="left")
    fut = R("E6") / "profiles_future.csv"
    if fut.exists():
        f = pd.read_csv(fut)[["profile", "share", "eocj_share"]]
        f.columns = ["profile", "share_future", "eocj_share_future"]
        t = t.merge(f, on="profile", how="left")
    keep = ["profile", "share", "share_test"] + [c for c in ("share_future",) if c in t] + [
        "med_inputs", "med_outputs", "med_value_sat", "med_fee_rate", "med_vsize", "top_script",
        "top_script_share", "rbf_share", "opreturn_share", "runes_share_test", "exchange_tag_share",
        "top_rule", "top_rule_share"] + [c for c in ("eocj_share_future",) if c in t] + ["representative_txid"]
    save_table(t[keep], "T5_profiles")


# ---------------------------------------------------------------------------
# figures from E0 / E1
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig = plt.figure(figsize=(ps.FULL_W, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    steps = [("Transaction\nrecords", "12 features,\nsatoshi units"),
             ("Transform\nand scale", "signed log;\nmedian / IQR\nfrom DEV"),
             ("Rank-power\nweights", "MI with proxy\nK-means;\n$w_j \\propto r_j^{-1.5}$"),
             ("Hierarchical\ninitialisation", "160 micro-\nclusters; size-\nweighted Ward"),
             ("K-means and\nrefinement", "split clusters\n> 10% of rows\n(depth ≤ 3)"),
             ("Profiles", "nearest\ncentroid;\nannotations")]
    n = len(steps)
    left, right, gap = 0.01, 0.99, 0.022
    width = (right - left - gap * (n - 1)) / n
    for i, (title, sub) in enumerate(steps):
        x0 = left + i * (width + gap)
        box = FancyBboxPatch((x0, 0.05), width, 0.90, boxstyle="round,pad=0,rounding_size=0.02",
                             fc="#dbe9fb" if i in (2, 3, 4) else "#eef4fc", ec=ps.SERIES[0], lw=0.8)
        ax.add_patch(box)
        ax.text(x0 + width / 2, 0.76, title, ha="center", va="center", fontsize=7.5, weight="bold")
        ax.text(x0 + width / 2, 0.36, sub, ha="center", va="center", fontsize=6.8, color=ps.INK2,
                linespacing=1.15)
        if i < n - 1:
            ax.annotate("", xy=(x0 + width + gap, 0.50), xytext=(x0 + width, 0.50),
                        arrowprops=dict(arrowstyle="-|>", color=ps.MUTED, lw=0.8, shrinkA=0, shrinkB=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    savefig(fig, "F1_pipeline")


def fig_design():
    fm = load_json("E0/future_manifest.json")
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 1.6))
    spans = [("Development (fit)", "2022-07-13", "2024-01-01", ps.SERIES[0]),
             ("Test", "2024-01-01", "2024-09-08", ps.SERIES[1]),
             ("Prospective (collected after the freeze)", "2024-10-01", "2026-09-01", ps.SERIES[2])]
    for i, (lab, a, b, c) in enumerate(spans):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.barh(0, (b - a).days, left=a, height=0.45, color=c)
        ax.text(a + (b - a) / 2, 0.42, lab, ha="center", va="bottom", fontsize=7)
    for d, lab in (("2024-04-20", "Runes launch\n(block 840,000)"), ("2026-09-17", "code frozen\n(tag v2-frozen)")):
        ax.plot([pd.Timestamp(d)] * 2, [-0.5, 0.26], color=ps.INK2, lw=0.8, ls="--")
        ax.text(pd.Timestamp(d), -0.55, lab, ha="center", va="top", fontsize=6.3, color=ps.INK2)
    ax.set_ylim(-1.0, 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    savefig(fig, "F2_design")


def fig_weights():
    w = pd.read_csv(R("E1") / "weights.csv").sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(ps.HALF_W, 2.6))
    y = np.arange(len(w))
    ax.barh(y, w.weight, color=ps.SERIES[0], height=0.62)
    ax.set_yticks(y, [NICE.get(f, f) for f in w.feature])
    for yi, (wt, mi) in enumerate(zip(w.weight, w.mi)):
        ax.text(wt + 0.006, yi, f"{wt:.3f}  (MI {mi:.2f})", va="center", fontsize=6.3, color=ps.INK2)
    ax.set_xlim(0, w.weight.max() * 1.55)
    ax.set_xlabel("Weight $w_j$ (sums to 1)")
    ax.grid(axis="y", visible=False)
    savefig(fig, "F3_weights")


def fig_profiles():
    base = pd.read_parquet(OUT / "base" / "base.parquet",
                           columns=["split", "L2", "L3", "tags", "has_op_return", "rbf_enabled",
                                    "input_count", "output_count", "total_input_value",
                                    "fee_rate_sat_per_vbyte", "vsize"])
    dev = base[base.split == 0].reset_index(drop=True)
    test = base[base.split == 1].reset_index(drop=True)
    lab = np.load(OUT / "labels" / "dev_zsh.npy")
    labt = np.load(OUT / "labels" / "test_zsh.npy")
    k = int(lab.max()) + 1
    cols = {}
    g = dev.assign(p=lab).groupby("p")
    for f, name in (("input_count", "inputs"), ("output_count", "outputs"), ("total_input_value", "value"),
                    ("fee_rate_sat_per_vbyte", "fee rate"), ("vsize", "vsize")):
        med = np.log10(g[f].median().clip(lower=1e-9) + 1)
        cols[f"median {name}"] = (med - med.min()) / (med.max() - med.min())
    for code, name in enumerate(L2_NAMES):
        if name in ("other", "coinbase"):
            continue
        cols[name] = g.L2.apply(lambda s, c=code: (s == c).mean())
    cols["OP_RETURN"] = g.has_op_return.mean()
    cols["RBF"] = g.rbf_enabled.mean()
    cols["exchange tag"] = g.tags.apply(lambda s: ((s.to_numpy() & 1) > 0).mean())
    gt = test.assign(p=labt).groupby("p")
    cols["Runes (2024)"] = gt.L3.apply(lambda s: (s == 1).mean()).reindex(range(k))
    M = pd.DataFrame(cols).reindex(range(k))
    M.to_csv(TAB / "F4_profile_matrix.csv")
    share = pd.Series(np.bincount(lab, minlength=k) / len(lab))
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f7f9fc"] + ps.SEQ[1:])
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=40, ha="right")
    ax.set_yticks(range(k), [f"P{i:02d} ({share[i] * 100:.1f}%)" for i in range(k)], fontsize=6)
    ax.axvline(4.5, color="white", lw=2)
    ax.axvline(M.shape[1] - 1.5, color="white", lw=2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("column-scaled median (left block) or share of members", fontsize=6.5)
    savefig(fig, "F4_profiles")


# ---------------------------------------------------------------------------
# E2 / E3
# ---------------------------------------------------------------------------
def tlabel(t):
    if t == "L2:coinbase":
        return "Coinbase"
    kind, name = t.split(":", 1)
    return {"L2": "Inputs: ", "L3": "OP_RETURN: ", "L4": "Tag: ", "L5": "", "L1": "Rule: "}[kind] + {
        "runes": "Runes", "omni": "Omni", "other_opreturn": "other", "eocj": "Equal-output CoinJoin",
        "coinbase": "coinbase"}.get(name, name)


def table_methods():
    g = pd.read_csv(R("E2") / "methods_intrinsic_timing.csv")
    c = pd.read_csv(R("E2") / "methods_concentration_independent.csv")
    lift = c.groupby("method").ap_lift.agg(["median", "count"]).rename(
        columns={"median": "Median AP lift", "count": "Targets"})
    wins = c[c.method != "ZSH"].assign(better=lambda d: (d.d_ap_lift_hi < 0) & (d.d_ap_lift_p_holm < 0.05),
                                       worse=lambda d: (d.d_ap_lift_lo > 0) & (d.d_ap_lift_p_holm < 0.05))
    w = wins.groupby("method")[["better", "worse"]].sum().rename(
        columns={"better": "ZSH higher (Holm)", "worse": "ZSH lower (Holm)"})
    t = g.merge(lift, left_on="method", right_index=True, how="left").merge(
        w, left_on="method", right_index=True, how="left")
    cols = {"method": "Method", "k_fit": "K", "noise_share": "Noise share", "fit_seconds": "Fit (s)",
            "predict_test_seconds": "Assign TEST (s)", "common_silhouette": "Silhouette",
            "common_dbi": "Davies-Bouldin", "common_chi": "Calinski-Harabasz", "common_chi_dedup": "CH (dedup.)",
            "own_silhouette": "Silhouette (own space)", "test_max_share": "Max share (TEST)",
            "Median AP lift": "Median AP lift", "Targets": "Targets",
            "ZSH higher (Holm)": "ZSH higher (Holm)", "ZSH lower (Holm)": "ZSH lower (Holm)"}
    t = t[[k for k in cols if k in t]].rename(columns=cols)
    save_table(t, "T6_methods")


def fig_methods():
    c = pd.read_csv(R("E2") / "methods_concentration_independent.csv")
    targets = list(dict.fromkeys(c.target))
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 0.28 * len(targets) + 1.0))
    y = {t: i for i, t in enumerate(targets[::-1])}
    others = c[~c.method.isin(["ZSH", "KMeans++"])]
    ax.scatter(others.ap_lift, others.target.map(y), s=14, color=ps.MUTED, alpha=0.6, lw=0,
               label="other baselines", zorder=2)
    for m, col, dy in (("KMeans++", ps.SERIES[1], -0.14), ("ZSH", ps.SERIES[0], 0.14)):
        d = c[c.method == m]
        yy = d.target.map(y) + dy
        ax.errorbar(d.ap_lift, yy, xerr=[d.ap_lift - d.ap_lift_lo, d.ap_lift_hi - d.ap_lift],
                    fmt="o", ms=4, color=col, elinewidth=1, capsize=0, label=m if m == "ZSH" else "K-means++",
                    zorder=3)
    ax.axvline(1, color=ps.MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yticks(list(y.values()), [tlabel(t) for t in y])
    ax.set_xlabel("AP lift on TEST (average precision / base rate; log scale; 1 = no concentration)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13 - 1.2 / (0.28 * len(targets) + 1.0) * 0.1),
              ncol=3)
    ax.grid(axis="y", visible=False)
    savefig(fig, "F5_methods")


ARM_NAMES = {"A1": "ZSH (rank-power + refinement)", "A2": "uniform + refinement",
             "A3": "rank-power, no refinement", "A4": "uniform, no refinement",
             "A5": "uniform, no refinement (K_u)", "A6": "MI-direct + refinement",
             "A7": "Laplacian rank-power + refinement", "A9": "K-means++ (K*)"}


def table_factorial():
    g = pd.read_csv(R("E3") / "arms_geometry.csv")
    allc = pd.read_csv(R("E3") / "all_vs_A1_independent.csv")
    med = allc.groupby("method").ap_lift.median().rename("Median AP lift")
    t = g.merge(med, left_on="arm", right_index=True, how="left")
    t.insert(1, "Arm", t.arm.map(ARM_NAMES))
    t = t[["arm", "Arm", "k", "max_share_dev", "max_share_test", "common_silhouette", "common_dbi",
           "common_chi", "own_silhouette", "Median AP lift", "fit_seconds"]]
    save_table(t, "T7a_factorial_arms")
    rows = []
    for name, ref, other in (("H1 primary: rank-power vs uniform (no refinement, K*)", "A4", "A3"),
                             ("H1 secondary: rank-power vs uniform (with refinement)", "A2", "A1"),
                             ("H2 primary: refinement vs none (rank-power, K*)", "A3", "A1"),
                             ("H2 secondary: refinement vs none (uniform, K_u)", "A5", "A2")):
        key = {"A4": "H1_primary_A3_vs_A4", "A2": "H1_secondary_A1_vs_A2", "A3": "H2_primary_A1_vs_A3",
               "A5": "H2_secondary_A2_vs_A5"}[ref]
        for kind in ("independent", "structural"):
            f = R("E3") / f"{key}_{kind}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            a = d[d.method == other].set_index("target")
            b = d[d.method == ref].set_index("target")
            for tname in a.index:
                rows.append({"Contrast": name, "Annotation type": kind, "Target": tlabel(tname),
                             f"AP lift (test arm)": a.loc[tname, "ap_lift"],
                             f"AP lift (reference arm)": b.loc[tname, "ap_lift"],
                             "Difference": a.loc[tname, "d_ap_lift"],
                             "95% CI low": a.loc[tname, "d_ap_lift_lo"], "95% CI high": a.loc[tname, "d_ap_lift_hi"],
                             "p (Holm)": a.loc[tname].get("d_ap_lift_p_holm", np.nan)})
    save_table(pd.DataFrame(rows), "T7b_factorial_contrasts")


def fig_factorial():
    panels = (("H1_primary_A3_vs_A4", "A3", "Rank-power vs uniform weights\n(no refinement, same K)"),
              ("H2_primary_A1_vs_A3", "A1", "Refinement vs none\n(rank-power weights, same K)"))
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 3.3), sharey=True)
    for ax, (key, arm, title) in zip(axes, panels):
        d = pd.read_csv(R("E3") / f"{key}_independent.csv")
        d = d[d.method == arm].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        sig = d.d_ap_lift_p_holm < 0.05
        ax.errorbar(d.d_ap_lift, y, xerr=[d.d_ap_lift - d.d_ap_lift_lo, d.d_ap_lift_hi - d.d_ap_lift],
                    fmt="none", ecolor=ps.SERIES[0], elinewidth=1)
        ax.scatter(d.d_ap_lift[sig], y[sig], color=ps.SERIES[0], s=22, zorder=3, label="Holm p < 0.05")
        ax.scatter(d.d_ap_lift[~sig], y[~sig], facecolor="white", edgecolor=ps.SERIES[0], s=22, zorder=3,
                   label="not significant")
        ax.axvline(0, color=ps.MUTED, lw=0.8)
        ax.set_yticks(y, [tlabel(t) for t in d.target])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Difference in AP lift on TEST")
        ax.grid(axis="y", visible=False)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    savefig(fig, "F6_factorial")


# ---------------------------------------------------------------------------
# E4 stability
# ---------------------------------------------------------------------------
def table_stability():
    s = pd.read_csv(R("E4") / "summary_by_method.csv", keep_default_na=False, na_values=[""])
    p = pd.read_csv(R("E4") / "pairwise.csv", keep_default_na=False, na_values=[""])
    js = load_json("E4/summary.json")["clusterwise"]
    t = s.merge(p, on=["kind", "method"], how="outer")
    t["Clusters (Jaccard ≥ 0.75 / 0.5–0.75 / < 0.5)"] = t.apply(
        lambda r: (f"{js[r.method]['n_ge_0.75']} / {js[r.method]['n_0.5_to_0.75']} / {js[r.method]['n_lt_0.5']}"
                   if r.kind == "bootstrap" and r.method in js else ""), axis=1)
    t["Mean cluster Jaccard"] = t.apply(
        lambda r: js[r.method]["mean_jaccard"] if r.kind == "bootstrap" and r.method in js else np.nan, axis=1)
    kind_name = {"seed": "10 seeds, same data", "bootstrap": "30 block-bootstrap refits",
                 "null": "10 seeds, column-permuted data (no joint structure)"}
    t.insert(0, "Replicates", t.kind.map(kind_name))
    cols = ["Replicates", "method", "k_mean", "ari_mean", "ari_std", "ami_mean", "vi_bits_mean",
            "centroid_shift_mean", "pairwise_ari_mean", "pairwise_ari_min", "mi_rank_tau_mean",
            "Mean cluster Jaccard", "Clusters (Jaccard ≥ 0.75 / 0.5–0.75 / < 0.5)"]
    save_table(t[[c for c in cols if c in t]], "T8_stability")


def fig_stability():
    from itertools import combinations  # noqa: F401
    reps = pd.read_csv(R("E4") / "replicates.csv", keep_default_na=False, na_values=[""])
    jac = pd.read_csv(R("E4") / "clusterwise_jaccard_long.csv", keep_default_na=False, na_values=[""])
    methods = ["ZSH", "KMeans++ K*", "RPW K* (no refinement)"]
    names = {"ZSH": "ZSH", "KMeans++ K*": "K-means++", "RPW K* (no refinement)": "rank-power,\nno refinement"}
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for i, kind in enumerate(("seed", "bootstrap")):
        for j, m in enumerate(methods):
            v = reps[(reps.kind == kind) & (reps.method == m)].ari
            x = j + (i - 0.5) * 0.3
            jit = np.random.default_rng(j + 10 * i).uniform(-0.05, 0.05, len(v))
            if kind == "seed":
                ax.scatter(np.full(len(v), x) + jit, v, s=12, facecolor="white", edgecolor=ps.SERIES[j], lw=0.9,
                           label="10 seeds (hollow)" if j == 0 else None)
            else:
                ax.scatter(np.full(len(v), x) + jit, v, s=10, color=ps.SERIES[j], lw=0, alpha=0.85,
                           label="30 block-bootstrap refits (filled)" if j == 0 else None)
    ax.set_xticks(range(len(methods)), [names[m] for m in methods])
    ax.set_ylabel("ARI with the full-DEV fit")
    ax.set_ylim(0, 1)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec=ps.INK2, label="10 seeds"),
                       Line2D([], [], ls="", marker="o", color=ps.INK2, label="30 block-bootstrap refits")],
              loc="lower left")
    ax.set_title("(a) Agreement with full-DEV fit", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    b = jac[jac.kind == "bootstrap"].groupby(["method", "cluster"]).jaccard.mean().reset_index()
    for j, m in enumerate(methods):
        v = np.sort(b[b.method == m].jaccard.to_numpy())[::-1]
        ax.plot(np.arange(1, len(v) + 1), v, marker="o", ms=3, lw=1.2, color=ps.SERIES[j], label=names[m].replace("\n", " "))
    for yv in (0.5, 0.75):
        ax.axhline(yv, color=ps.MUTED, lw=0.7, ls=":")
    ax.set_xlabel("Clusters, sorted by mean Jaccard")
    ax.set_ylabel("Mean best Jaccard (30 refits)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.set_title("(b) Cluster-wise stability", loc="left")
    fig.tight_layout()
    savefig(fig, "F7_stability")


# ---------------------------------------------------------------------------
# E5 transfer
# ---------------------------------------------------------------------------
def table_transfer():
    s = load_json("E5/summary.json")
    rows = []
    for period in ("TEST", "FUTURE"):
        if period not in s:
            continue
        for m in ("ZSH", "KMeans++ K*"):
            a = s[period][m]
            rows.append({"Period": period, "Method": m, "Rows": s[period]["rows"], "Refit K": a["refit_k"],
                         "ARI transfer vs refit": a["ari"], "AMI": a["ami"], "VI (bits)": a["vi_bits"],
                         "Mean best Jaccard": a["mean_best_jaccard"],
                         "Profiles with Jaccard ≥ 0.75": a["n_best_jaccard_ge_0.75"],
                         "Profiles with Jaccard < 0.5": a["n_best_jaccard_lt_0.5"]})
    save_table(pd.DataFrame(rows), "T9a_transfer")
    rows = []
    for period in ("TEST", "FUTURE"):
        for key, lab in (("runes_transfer_zsh", "ZSH, transferred"), ("runes_refit_zsh", "ZSH, refitted"),
                         ("runes_transfer_kmu", "K-means++, transferred")):
            if period in s and key in s[period]:
                r = s[period][key]
                rows.append({"Period": period, "Partition": lab, "Runes transactions": r["runes"],
                             "Profiles holding 80%": r["profiles_for_80pct"],
                             "Profiles holding 95%": r["profiles_for_95pct"],
                             "Profiles ≥ 90% Runes": r["profiles_runes_ge_90pct"],
                             "Share of Runes in such profiles": r["runes_in_ge_90pct_profiles"]})
    save_table(pd.DataFrame(rows), "T9b_runes")


def fig_drift():
    m = pd.read_csv(R("E5") / "monthly_profile_shares.csv")
    piv = m.pivot(index="profile", columns="month", values="share").fillna(0)
    months = list(piv.columns)
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 4.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f7f9fc"] + ps.SEQ[1:])
    vmax = float(np.quantile(piv.to_numpy(), 0.99))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    step = max(1, len(months) // 16)
    ax.set_xticks(range(0, len(months), step), months[::step], rotation=45, ha="right")
    ax.set_yticks(range(len(piv)), [f"P{int(i):02d}" for i in piv.index], fontsize=6)
    for d, lab in (("2024-01", "test"), ("2024-04", "Runes"), ("2024-10", "prospective")):
        if d in months:
            x = months.index(d) - 0.5
            ax.axvline(x, color=ps.SERIES[1], lw=1)
            ax.text(x + 0.3, -0.9, lab, color=ps.SERIES[1], fontsize=6.5, va="bottom")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("monthly share of transactions (future months design-weighted)", fontsize=6.5)
    savefig(fig, "F8_drift")


# ---------------------------------------------------------------------------
# E6 concentration, E7 heuristic validity
# ---------------------------------------------------------------------------
def table_concentration():
    rows = []
    for period in ("test", "future"):
        f = R("E6") / f"{period}_independent.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        z = d[d.method == "ZSH"].set_index("target")
        k = d[d.method == "K-means++ (K*)"].set_index("target")
        for t in z.index:
            rows.append({"Period": period.upper(), "Target": tlabel(t), "Positives": int(z.loc[t, "positives"]),
                         "Base rate": z.loc[t, "base_rate"],
                         "ZSH AP lift": z.loc[t, "ap_lift"], "ZSH 95% CI": f"{z.loc[t, 'ap_lift_lo']:.2f}–{z.loc[t, 'ap_lift_hi']:.2f}",
                         "ZSH enrichment @25%": z.loc[t, "enrich@0.25"],
                         "Clusters @25%": int(z.loc[t, "clusters@0.25"]),
                         "K-means++ AP lift": k.loc[t, "ap_lift"],
                         "Difference (K-means++ − ZSH)": k.loc[t, "d_ap_lift"],
                         "Diff. 95% CI": f"{k.loc[t, 'd_ap_lift_lo']:.2f} to {k.loc[t, 'd_ap_lift_hi']:.2f}",
                         "p (Holm)": k.loc[t, "d_ap_lift_p_holm"]})
    save_table(pd.DataFrame(rows), "T10_concentration")


def _avg_curve(curves, grid):
    ys = []
    for c in curves:
        cov = np.asarray(c["coverage"])
        prec = np.asarray(c["precision"])
        idx = np.clip(np.searchsorted(cov, grid), 0, len(cov) - 1)
        ys.append(prec[idx] / c["base_rate"])
    return np.mean(ys, axis=0)


def fig_curves():
    s6 = load_json("E6/summary.json")
    panels = []
    for period, targets in (("TEST", ["L3:runes", "L4:exchange", "L2:P2WSH"]),
                            ("FUTURE", ["L5:eocj", "L3:runes", "L2:P2WSH"])):
        if s6 and period in s6:
            for t in targets:
                if t in s6[period]["curves"]:
                    panels.append((period, t, s6[period]["curves"][t]))
    if not panels:
        return
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ps.FULL_W, 2.2 * nrow), squeeze=False)
    grid = np.linspace(0.005, 1, 200)
    order = [("ZSH", ps.SERIES[0]), ("K-means++ (K*)", ps.SERIES[1]),
             ("rank-power K-means (K*)", ps.SERIES[2]), ("uniform + refinement", ps.SERIES[3])]
    for ax, (period, t, cur) in zip(axes.ravel(), panels):
        for m, col in order:
            if m in cur:
                ax.step(grid, _avg_curve(cur[m], grid), where="post", color=col, lw=1.3, label=m)
        ax.axhline(1, color=ps.MUTED, lw=0.7, ls=":")
        lo, hi = ax.get_ylim()
        if hi / max(lo, 1e-9) > 12:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{period}: {tlabel(t)}", loc="left")
        ax.set_xlabel("coverage of positives")
        ax.set_ylabel("precision / base rate")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=6.3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "F9_curves")


def table_heuristic():
    s = load_json("E7/summary.json")
    rows = []
    if s and "D1" in s:
        for k, v in s["D1"]["estimates"].items():
            rows.append({"Data": "D1 (2022–2024), stratified API sample", "Quantity": k,
                         "Estimate": v["estimate"], "95% CI": f"{v['ci'][0]:.4f}–{v['ci'][1]:.4f}"})
        rows.append({"Data": "D1", "Quantity": "GraphSense coinjoin-tag matches", "Estimate": s["D1"]["coinjoin_tag_matches"], "95% CI": ""})
    if s and "FUTURE" in s:
        for k, v in s["FUTURE"]["design_weighted"].items():
            rows.append({"Data": "Future sample (design-weighted)", "Quantity": k, "Estimate": v, "95% CI": ""})
        for k, v in s["FUTURE"]["confusion"].items():
            rows.append({"Data": "Future sample (counts)", "Quantity": k, "Estimate": v, "95% CI": ""})
        rows.append({"Data": "Future sample", "Quantity": "GraphSense coinjoin-tag matches",
                     "Estimate": s["FUTURE"]["coinjoin_tag_matches"], "95% CI": ""})
    save_table(pd.DataFrame(rows), "T11_heuristic", floatfmt=".4g")


# ---------------------------------------------------------------------------
# E8 Elliptic, E9 atypicality
# ---------------------------------------------------------------------------
def table_elliptic():
    c = pd.read_csv(R("E8") / "concentration.csv")
    s = load_json("E8/summary.json")
    rows = []
    for _, r in c.iterrows():
        rows.append({"Features": r.setting, "Method": r.method, "K": r.k, "AP lift": r.ap_lift,
                     "95% CI": f"{r.ap_lift_lo:.2f}–{r.ap_lift_hi:.2f}",
                     "Enrichment @10%": r["enrich@0.10"], "Enrichment @25%": r["enrich@0.25"],
                     "Enrichment @25% CI": f"{r['enrich@0.25_lo']:.2f}–{r['enrich@0.25_hi']:.2f}",
                     "Precision @25%": r["prec@0.25"], "Clusters @25%": r["clusters@0.25"],
                     "Diff. vs ZSH (AP lift)": r.get("d_ap_lift", np.nan),
                     "Diff. CI": (f"{r['d_ap_lift_lo']:.2f} to {r['d_ap_lift_hi']:.2f}"
                                  if pd.notna(r.get("d_ap_lift", np.nan)) else ""),
                     "p (Holm)": r.get("d_ap_lift_p_holm", np.nan)})
    t = pd.DataFrame(rows)
    rf = s["random_forest_reference"]
    t = pd.concat([t, pd.DataFrame([{"Features": "AF-165", "Method": "Random forest (supervised reference)",
                                     "AP lift": rf["ap"] / rf["base_rate"], "Precision @25%": np.nan,
                                     "95% CI": f"precision {rf['precision']:.3f}, recall {rf['recall']:.3f}, F1 {rf['f1']:.3f}"}])],
                  ignore_index=True)
    save_table(t, "T12_elliptic")


def fig_elliptic():
    s = load_json("E8/summary.json")
    ts = pd.read_csv(R("E8") / "per_timestep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.7))
    ax = axes[0]
    cur = s["AF-165"]["curves"]
    order = [("ZSH", ps.SERIES[0]), ("K-means++ (K)", ps.SERIES[1]), ("rank-power K-means (K)", ps.SERIES[2]),
             ("uniform + refinement", ps.SERIES[3])]
    grid = np.linspace(0.005, 1, 200)
    for m, col in order:
        if m in cur:
            ax.plot(grid, _avg_curve(cur[m], grid), color=col, lw=1.3, label=m)
    ax.axhline(1, color=ps.MUTED, lw=0.7, ls=":")
    ax.set_xlabel("coverage of test illicit transactions")
    ax.set_ylabel("precision / base rate (6.5%)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("(a) Enrichment at each coverage, test steps", loc="left")
    ax = axes[1]
    for m, col in order[:2]:
        d = ts[(ts.setting == "AF-165") & (ts.method == m)]
        ax.plot(d.timestep, d.precision, marker="o", ms=3, color=col, label=f"{m}: precision")
        ax.plot(d.timestep, d.recall, marker="s", ms=3, color=col, ls="--", label=f"{m}: recall")
    last = CFG["elliptic"]["train_last_timestep"]
    ax.axvline(43, color=ps.MUTED, lw=0.8, ls=":")
    ax.text(43.2, 0.95, "dark-market\nclosure", fontsize=6, color=ps.INK2, va="top")
    ax.set_xlabel("time step")
    ax.set_ylim(0, 1)
    ax.set_title("(b) Top clusters per test step", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.8)
    _ = last
    fig.tight_layout()
    savefig(fig, "F10_elliptic")


def table_atypicality():
    s = load_json("E9/summary.json")
    rows = []
    for name, v in s["elliptic"].items():
        rows.append({"Data": "Elliptic test (steps 35–49)", "Score / target": f"{name} / illicit",
                     "n": v["n"], "Positives": v["positives"], "ROC-AUC": v["roc_auc"],
                     "ROC-AUC 95% CI": f"{v['roc_auc_ci'][0]:.3f}–{v['roc_auc_ci'][1]:.3f}",
                     "PR-AUC": v["pr_auc"], "Base rate": v["base_rate"], "H6 decision": v["h6_decision"]})
    for period, d in s["bitcoin"].items():
        for tgt, v in d.items():
            rows.append({"Data": f"Bitcoin {period} (exploratory)", "Score / target": f"IF (ZSH space) / {tgt}",
                         "n": v["n"], "Positives": v["positives"], "ROC-AUC": v["roc_auc"],
                         "ROC-AUC 95% CI": f"{v['roc_auc_ci'][0]:.3f}–{v['roc_auc_ci'][1]:.3f}",
                         "PR-AUC": v["pr_auc"], "Base rate": v["base_rate"], "H6 decision": ""})
    save_table(pd.DataFrame(rows), "T13_atypicality")


def fig_atypicality():
    from sklearn.metrics import roc_curve
    sfx = "_smoke" if SFX else ""
    d = pd.read_parquet(OUT / ("labels" + sfx) / "elliptic_test_atypicality.parquet")
    fig, ax = plt.subplots(figsize=(ps.HALF_W, 3.9))
    cols = [c for c in d.columns if c not in ("timestep", "illicit")]
    nice = {"IF (rank-power space)": "Isolation Forest, ZSH space",
            "IF (unweighted space)": "Isolation Forest, unweighted",
            "LOF (rank-power space)": "local outlier factor, ZSH space",
            "distance to ZSH centroid": "distance to ZSH centroid"}
    for c, col in zip(cols, ps.SERIES):
        fpr, tpr, _ = roc_curve(d.illicit, d[c])
        ax.plot(fpr, tpr, color=col, lw=1.3, label=nice.get(c, c))
    ax.plot([0, 1], [0, 1], color=ps.MUTED, lw=0.7, ls=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6.3, ncol=1)
    savefig(fig, "F11_atypicality")


# ---------------------------------------------------------------------------
# E10 sensitivity
# ---------------------------------------------------------------------------
def table_sensitivity():
    g = pd.read_csv(R("E10") / "variants_geometry.csv")
    c = pd.read_csv(R("E10") / "variants_independent.csv")
    agg = c.groupby("method").agg(median_ap_lift=("ap_lift", "median"))
    diff = c[c.method != "reference"].groupby("method").apply(
        lambda d: pd.Series({"higher": int(((d.d_ap_lift_lo > 0) & (d.d_ap_lift_p_holm < 0.05)).sum()),
                             "lower": int(((d.d_ap_lift_hi < 0) & (d.d_ap_lift_p_holm < 0.05)).sum()),
                             "median_diff": d.d_ap_lift.median()}), include_groups=False)
    t = g.drop(columns=["weights"]).merge(agg, left_on="variant", right_index=True, how="left").merge(
        diff, left_on="variant", right_index=True, how="left")
    t = t.rename(columns={"variant": "Variant", "k": "K", "max_share_fit": "Max share (fit)",
                          "max_share_test": "Max share (TEST)", "common_silhouette": "Silhouette",
                          "common_dbi": "Davies-Bouldin", "common_chi": "Calinski-Harabasz",
                          "median_ap_lift": "Median AP lift", "higher": "Targets higher than reference (Holm)",
                          "lower": "Targets lower than reference (Holm)", "median_diff": "Median AP-lift difference",
                          "fit_seconds": "Fit (s)"})
    save_table(t, "T14a_sensitivity")
    loo = pd.read_csv(R("E10") / "loo_seeded.csv")
    loo = loo[["family", "feature_removed", "method", "positives", "ap_lift", "ap_lift_lo", "ap_lift_hi",
               "d_ap_lift", "d_ap_lift_lo", "d_ap_lift_hi", "d_ap_lift_p"]]
    save_table(loo, "T14b_loo")


def fig_sensitivity():
    """Heat map: log2 of each variant's AP lift relative to the reference variant, all annotations."""
    c = pd.read_csv(R("E10") / "variants_independent.csv")
    g = pd.read_csv(R("E10") / "variants_geometry.csv").set_index("variant")
    order = ["reference", "s=0.5", "s=1.0", "s=2.0", "s=3.0", "K0=10", "K0=20", "K0=40", "K0=60",
             "cap=0.05", "cap=0.15", "cap=0.2", "cap=None", "depth=6",
             "upsampled (v1 style)", "sample-weighted", "init=k-means++", "init=semantic seeds",
             "init=seed-Ward blend"]
    names = {"reference": "reference", "s=0.5": "s = 0.5", "s=1.0": "s = 1", "s=2.0": "s = 2", "s=3.0": "s = 3",
             "K0=10": "K₀ = 10", "K0=20": "K₀ = 20", "K0=40": "K₀ = 40", "K0=60": "K₀ = 60",
             "cap=0.05": "cap 5%", "cap=0.15": "cap 15%", "cap=0.2": "cap 20%", "cap=None": "no refinement",
             "depth=6": "depth 6", "upsampled (v1 style)": "upsampled corpus", "sample-weighted": "sample weights",
             "init=k-means++": "K-means++ start", "init=semantic seeds": "semantic seeds",
             "init=seed-Ward blend": "seed–Ward blend"}
    targets = list(dict.fromkeys(c.target))
    piv = c.pivot(index="method", columns="target", values="ap_lift")[targets]
    rel = np.log2(piv.loc[order] / piv.loc["reference"])
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 4.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [ps.DIV_NEG, ps.DIV_MID, ps.DIV_POS])
    lim = 1.5
    im = ax.imshow(rel.clip(-lim, lim).to_numpy(), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = piv.loc[order[i], targets[j]]
            txt = f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color=ps.INK if abs(rel.iloc[i, j]) < 1.0 else "white",
                    weight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(targets)), [tlabel(t) for t in targets], rotation=35, ha="right")
    ax.set_yticks(range(len(order)), [f"{names[v]} (K = {int(g.loc[v, 'k'])})" for v in order])
    for y in (0.5, 4.5, 8.5, 13.5, 15.5):
        ax.axhline(y, color="white", lw=2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("log₂(AP lift / reference), clipped at ±1.5", fontsize=6.5)
    savefig(fig, "F12_sensitivity")


# ---------------------------------------------------------------------------
# E17 benchmark across clustering families (added after the freeze)
# ---------------------------------------------------------------------------
BENCH_ORDER = ["ZSH", "K-means++ (K*)", "MiniBatchKMeans", "GMM (diag)", "BIRCH",
               "Ward (sample + NC)", "VKV-partial (K*)"]
BENCH_LABEL = {"ZSH": "ZSH", "K-means++ (K*)": "K-means++", "MiniBatchKMeans": "mini-batch K-means",
               "GMM (diag)": "Gaussian mixture", "BIRCH": "BIRCH", "Ward (sample + NC)": "Ward",
               "VKV-partial (K*)": "VKV-partial"}


def fig_bench():
    """Heat map: share of the attainable concentration reached by each family, test period."""
    c = pd.read_csv(R("E17") / "test_independent.csv")
    methods = [m for m in BENCH_ORDER if m in set(c.method)]
    targets = list(dict.fromkeys(c.target))
    piv = c.pivot(index="method", columns="target", values="ap").loc[methods, targets] * 100
    base = c.groupby("target").base_rate.first()[targets]
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ps.SEQ)
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color=ps.INK if v < 55 else "white", weight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(targets)),
                  [f"{tlabel(t)}\n{100 * base[t]:.2f}%" for t in targets], rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), [BENCH_LABEL[m] for m in methods])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("attained share of the ceiling (%)", fontsize=6.5)
    savefig(fig, "F13_bench")


BUILDERS = [table_data, table_features, table_annotations, table_profiles,
            fig_pipeline, fig_design, fig_weights, fig_profiles,
            table_methods, fig_methods, table_factorial, fig_factorial,
            table_stability, fig_stability, table_transfer, fig_drift,
            table_concentration, fig_curves, table_heuristic,
            table_elliptic, fig_elliptic, table_atypicality, fig_atypicality, fig_bench,
            table_sensitivity, fig_sensitivity]


def main():
    ps.apply()
    only = set(sys.argv[1:])
    for b in BUILDERS:
        if only and b.__name__ not in only:
            continue
        try:
            b()
        except FileNotFoundError as e:
            print(f"skip {b.__name__}: {e}")


if __name__ == "__main__":
    main()
