"""Compact, formatted tables for the article (Markdown), built from results/.

Presentation only: reads saved results and writes results/manuscript/<name>.md.
Each builder runs only if its inputs exist. Captions are written in the article.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.config import CFG, OUT, RESULTS, results_dir  # noqa: E402
from zsh.data import L1_NAMES, L2_NAMES, L3_NAMES, TAG_BITS  # noqa: E402

MS = results_dir("manuscript")

NICE = {
    "input_count": "Input count", "output_count": "Output count", "vsize": "Virtual size",
    "size": "Serialised size", "total_input_value": "Total input value", "fee": "Fee",
    "fee_rate_sat_per_vbyte": "Fee rate per virtual byte", "fee_rate_sat_per_byte": "Fee rate per byte",
    "avg_output_value": "Mean output value", "input_output_ratio": "Input-to-output value ratio",
    "has_op_return": "OP_RETURN output (0/1)", "rbf_enabled": "Replace-by-fee signalled (0/1)",
}
TARGET = {
    "L2:coinbase": "Coinbase", "L2:P2PKH": "P2PKH inputs", "L2:P2SH": "P2SH inputs",
    "L2:P2WPKH": "P2WPKH inputs", "L2:P2WSH": "P2WSH inputs", "L2:P2TR": "P2TR inputs",
    "L2:mixed": "Mixed-script inputs", "L3:runes": "Runes", "L3:omni": "Omni",
    "L3:other_opreturn": "Other OP_RETURN", "L4:exchange": "Exchange tag", "L4:miner": "Miner tag",
    "L4:coinjoin": "CoinJoin tag", "L5:eocj": "Equal-output CoinJoin",
}
TARGET_SHORT = {
    "L2:coinbase": "CB", "L2:P2PKH": "PKH", "L2:P2SH": "SH", "L2:P2WPKH": "WPKH", "L2:P2WSH": "WSH",
    "L2:P2TR": "TR", "L2:mixed": "Mix", "L3:runes": "Run", "L3:omni": "Omni", "L3:other_opreturn": "OR",
    "L4:exchange": "Exch",
}


def R(exp):
    return RESULTS / exp


def js(p):
    p = RESULTS / p
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def n(x):
    return "" if pd.isna(x) else f"{int(round(x)):,}"


def f(x, d=2):
    if x is None or pd.isna(x):
        return ""
    t = f"{x:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    return t.replace("-", "−")


def fa(x, d=1):
    """d decimals, or two when |x| < 1."""
    return f(x, 2 if abs(x) < 1 else d)


def pct(x, d=1):
    return "" if x is None or pd.isna(x) else f"{100 * x:.{d}f}"


def ci(lo, hi, d=2):
    return f"{f(lo, d)}–{f(hi, d)}" if lo >= 0 else f"{f(lo, d)} to {f(hi, d)}"


def pval(p):
    if pd.isna(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def save(df, name, align=None):
    cols = list(df.columns)
    align = align or ["l"] + ["r"] * (len(cols) - 1)
    # dash counts set the relative column widths when pandoc wraps a wide table
    widths = []
    for i, c in enumerate(cols):
        cells = ["" if pd.isna(v) else str(v) for v in df.iloc[:, i]]
        longest_word = max(len(w) for w in str(c).split())
        longest_cell = max([len(x) for x in cells] + [0])
        if align[i] == "L":                      # wide text column
            widths.append(min(max(longest_word + 2, longest_cell + 1), 44))
        elif align[i] == "l":
            widths.append(min(max(longest_word + 2, min(longest_cell + 1, 14)), 14))
        else:
            widths.append(max(longest_word + 2, min(longest_cell + 3, 18), 9))
    sep = [("-" * (w - 1) + ":") if a == "r" else (":" + "-" * (w - 2) + ":" if a == "c" else ":" + "-" * (w - 1))
           for a, w in zip(align, widths)]
    align = ["l" if a == "L" else a for a in align]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(sep) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    (MS / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {df.shape}")


# ---------------------------------------------------------------------------
def t_data():
    s = pd.read_csv(R("E0") / "splits.csv").set_index("split")
    rows = [
        ["Bitcoin sample, development", f"{s.loc[0, 'first_time'][:10]} to {s.loc[0, 'last_time'][:10]}",
         n(s.loc[0, "rows"]), n(s.loc[0, "blocks"]), "fitting all models"],
        ["Bitcoin sample, test", f"{s.loc[1, 'first_time'][:10]} to {s.loc[1, 'last_time'][:10]}",
         n(s.loc[1, "rows"]), n(s.loc[1, "blocks"]), "evaluation"],
    ]
    fm = js("E0/future_manifest.json")
    if fm:
        rows.append(["Prospective sample", f"{fm['months'][0]['month']} to {fm['months'][-1]['month']}",
                     n(fm["rows"]), n(fm["blocks_with_page"]), "evaluation"])
    ell = OUT / "labels" / "elliptic_counts.json"
    if ell.exists():
        e = json.load(open(ell))
        rows.append(["Elliptic, training steps", "time steps 1–34", n(e["train_rows"]),
                     f"{e['train_labelled']:,} labelled", "fitting; ranking clusters"])
        rows.append(["Elliptic, test steps", "time steps 35–49", n(e["test_rows"]),
                     f"{e['test_labelled']:,} labelled", "evaluation"])
    save(pd.DataFrame(rows, columns=["Data", "Period", "Transactions", "Blocks or labels", "Use"]), "T_data",
         ["l", "l", "r", "r", "l"])


def elliptic_counts():
    """Small helper output for T_data (row counts of the Elliptic split)."""
    from zsh.io import elliptic_train_mask, load_elliptic
    p = OUT / "labels" / "elliptic_counts.json"
    if p.exists():
        return
    df = load_elliptic()
    tr = elliptic_train_mask(df)
    lab = df.label.to_numpy() >= 0
    out = {"train_rows": int(tr.sum()), "test_rows": int((~tr).sum()),
           "train_labelled": int((tr & lab).sum()), "test_labelled": int((~tr & lab).sum()),
           "train_illicit": int((tr & (df.label.to_numpy() == 1)).sum()),
           "test_illicit": int((~tr & (df.label.to_numpy() == 1)).sum())}
    json.dump(out, open(p, "w"), indent=1)


def t_rules():
    p = pd.read_csv(R("E0") / "annotation_prevalence.csv")
    p = p[p.annotation == "L1-rule"].pivot_table(index="label", columns="split", values="share")
    der = js("E0/rule_derivability.json")
    defs = {"ManyInManyOut": ("coinjoin-like", "inputs > 3 and outputs > 3"),
            "SingleInFanOut": ("batch payment", "inputs = 1 and outputs > 5"),
            "FanIn": ("consolidation", "inputs > outputs and inputs > 2"),
            "FanOut": ("distribution", "outputs > inputs and outputs > 2"),
            "OneInOneOut": ("peer-to-peer", "inputs = 1 and outputs = 1")}
    rows = [[old, rule, pct(p.loc[k, 0]), pct(p.loc[k, 1]), f(100 * der[k]["accuracy"], 2)]
            for k, (old, rule) in defs.items()]
    save(pd.DataFrame(rows, columns=["Flag in the published sample", "Definition", "Development (%)", "Test (%)",
                                     "Recovered by depth-3 tree (%)"]), "T_rules", ["l", "l", "r", "r", "r"])


def t_features():
    w = pd.read_csv(R("E1") / "weights.csv").sort_values("rank")
    rows = [[int(r["rank"]), NICE.get(r.feature, r.feature), f(r.mi, 3), f(r.weight, 3)] for _, r in w.iterrows()]
    save(pd.DataFrame(rows, columns=["Rank", "Feature", "Mutual information", "Weight"]), "T_features",
         ["r", "l", "r", "r"])


def t_annotations():
    p = pd.read_csv(R("E0") / "annotation_prevalence.csv")
    p = p[p.annotation.isin(["L2", "L3", "L4"]) & p.split.isin([0, 1])]
    wide = p.pivot_table(index=["annotation", "label"], columns="split", values="share").reset_index()
    fut = OUT / "future" / "d4_future.parquet"
    d = pd.read_parquet(fut) if fut.exists() else None
    order = [("L2", x) for x in ("P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR", "mixed", "coinbase")] + \
            [("L3", x) for x in ("runes", "omni", "other_opreturn")] + [("L4", x) for x in ("exchange", "miner", "coinjoin")]
    rows = []
    for a, lab in order:
        r = wide[(wide.annotation == a) & (wide.label == lab)]
        row = [a, TARGET[f"{a}:{lab}"], pct(r[0].iloc[0], 2), pct(r[1].iloc[0], 2)]
        if d is not None:
            col = {"L2": d.L2.to_numpy() == L2_NAMES.index(lab) if a == "L2" else None,
                   "L3": d.L3.to_numpy() == L3_NAMES.index(lab) if a == "L3" else None,
                   "L4": (d.tags.to_numpy() & TAG_BITS[lab]) > 0 if a == "L4" else None}[a]
            row.append(pct(np.average(col, weights=d.design_weight), 2))
        rows.append(row)
    cols = ["Type", "Annotation", "Development (%)", "Test (%)"]
    if d is not None:
        rows.append(["L5", TARGET["L5:eocj"], "", "", pct(np.average(d.eocj, weights=d.design_weight), 2)])
        cols.append("Prospective (%, weighted)")
    save(pd.DataFrame(rows, columns=cols), "T_annotations", ["l", "l"] + ["r"] * (len(cols) - 2))


def t_profiles():
    dev = pd.read_csv(R("E1") / "profiles_dev.csv")
    test = pd.read_csv(R("E1") / "profiles_test.csv").set_index("profile")
    futp = R("E6") / "profiles_future.csv"
    fut = pd.read_csv(futp).set_index("profile") if futp.exists() else None
    rows = []
    for _, r in dev.iterrows():
        p = int(r.profile)
        row = [f"P{p:02d}", pct(r.share), pct(test.loc[p, "share"]) if p in test.index else ""]
        if fut is not None:
            row.append(pct(fut.loc[p, "share"]) if p in fut.index else "")
        row += [f"{r.med_inputs:g}", f"{r.med_outputs:g}", n(r.med_value_sat), f(r.med_fee_rate, 1),
                f"{r.top_script} {pct(r.top_script_share, 0)}", pct(r.rbf_share, 0),
                pct(r.exchange_tag_share, 1),
                pct(test.loc[p, "runes_share"], 0) if p in test.index else ""]
        rows.append(row)
    cols = ["Profile", "Dev. (%)", "Test (%)"] + (["Prosp. (%)"] if fut is not None else []) + [
        "In", "Out", "Value (sat)", "Fee rate", "Main input script (%)", "RBF (%)", "Exch. tag (%)",
        "Runes, test (%)"]
    save(pd.DataFrame(rows, columns=cols), "T_profiles", ["l"] + ["r"] * (len(cols) - 1))


def t_methods():
    g = pd.read_csv(R("E2") / "methods_intrinsic_timing.csv")
    c = pd.read_csv(R("E2") / "methods_concentration_independent.csv")
    name = {"ZSH": "ZSH", "KMeans++": "K-means++", "MiniBatchKMeans": "Mini-batch K-means",
            "GMM (diag)": "Gaussian mixture (diag.)", "BIRCH": "BIRCH", "Ward (sample + NC)": "Ward (sample)",
            "VKV-partial (k=5)": "Vlahavas et al., partial (k = 5)",
            "VKV-partial (k=31)": "Vlahavas et al., partial (k = K)",
            "HDBSCAN (own sample)": "HDBSCAN (own sample)"}
    if "noise_share" in g:
        hd = g.method == "HDBSCAN (own sample)"
        if hd.any():
            name["HDBSCAN (own sample)"] = f"HDBSCAN (own sample, {pct(g.loc[hd, 'noise_share'].iloc[0], 0)}% noise)"
    rows = []
    for _, r in g.iterrows():
        cc = c[c.method == r.method]
        hi = lo = ""
        if r.method != "ZSH" and len(cc):
            hi = int(((cc.d_ap_lift_hi < 0) & (cc.d_ap_lift_p_holm < 0.05)).sum())
            lo = int(((cc.d_ap_lift_lo > 0) & (cc.d_ap_lift_p_holm < 0.05)).sum())
        rows.append([name.get(r.method, r.method), int(r.k_fit), f(r.fit_seconds, 1), f(r.common_silhouette, 3),
                     f(r.common_dbi, 2), f(r.common_chi / 1000, 1), f(r.own_silhouette, 3),
                     pct(r.test_max_share), f(cc.ap_lift.median(), 2) if len(cc) else "",
                     f"{hi} / {lo}" if hi != "" else ""])
    cols = ["Method", "K", "Fit (s)", "Silh.", "DBI", "CH (10³)", "Silh. own", "Largest (%)",
            "Median lift", "Higher / lower"]
    save(pd.DataFrame(rows, columns=cols), "T_methods", ["l"] + ["r"] * (len(cols) - 1))


ARMS = {"A1": ("rank-power", "yes", "hierarchical"), "A2": ("uniform", "yes", "hierarchical"),
        "A3": ("rank-power", "no", "hierarchical"), "A4": ("uniform", "no", "hierarchical"),
        "A5": ("uniform", "no, K of A2", "hierarchical"), "A6": ("MI-proportional", "yes", "hierarchical"),
        "A7": ("Laplacian rank-power", "yes", "hierarchical"), "A9": ("uniform", "no", "K-means++ (10 starts)")}


def t_factorial():
    g = pd.read_csv(R("E3") / "arms_geometry.csv").set_index("arm")
    allc = pd.read_csv(R("E3") / "all_vs_A1_independent.csv")
    med = allc.groupby("method").ap_lift.median()
    rows = []
    for a, (w, ref, init) in ARMS.items():
        if a not in g.index:
            continue
        r = g.loc[a]
        rows.append([a + (" (ZSH)" if a == "A1" else ""), w, ref, init, int(r.k), pct(r.max_share_dev),
                     f(r.common_silhouette, 3), f(med.get(a, np.nan), 2)])
    cols = ["Arm", "Weights", "Refinement", "Initialisation", "K", "Largest cluster (%)", "Silh.",
            "Median AP lift"]
    save(pd.DataFrame(rows, columns=cols), "T_factorial", ["l", "l", "l", "l", "r", "r", "r", "r"])


def t_contrasts():
    rows = []
    for key, test_arm, lab in (("H1_primary_A3_vs_A4", "A3", "H1: A3 − A4"),
                               ("H2_primary_A1_vs_A3", "A1", "H2: A1 − A3")):
        d = pd.read_csv(R("E3") / f"{key}_independent.csv")
        a = d[d.method == test_arm]
        for _, r in a.iterrows():
            rows.append([lab, TARGET.get(r.target, r.target), f(r.d_ap_lift, 2),
                         ci(r.d_ap_lift_lo, r.d_ap_lift_hi), pval(r.d_ap_lift_p_holm)])
    save(pd.DataFrame(rows, columns=["Contrast", "Annotation", "Difference in AP lift", "95% CI", "p (Holm)"]),
         "T_contrasts", ["l", "l", "r", "r", "r"])


def t_stability():
    s = pd.read_csv(R("E4") / "summary_by_method.csv", keep_default_na=False, na_values=[""])
    p = pd.read_csv(R("E4") / "pairwise.csv", keep_default_na=False, na_values=[""])
    cw = js("E4/summary.json")["clusterwise"]
    t = s.merge(p, on=["kind", "method"], how="outer")
    kind = {"seed": "10 seeds", "bootstrap": "30 block bootstraps", "null": "10 seeds, permuted columns"}
    name = {"ZSH": "ZSH", "KMeans++ K*": "K-means++", "RPW K* (no refinement)": "Rank-power, no refinement"}
    order = [("bootstrap", "ZSH"), ("bootstrap", "KMeans++ K*"), ("bootstrap", "RPW K* (no refinement)"),
             ("seed", "ZSH"), ("seed", "KMeans++ K*"), ("seed", "RPW K* (no refinement)"), ("null", "ZSH")]
    rows = []
    for k, m in order:
        r = t[(t.kind == k) & (t.method == m)]
        if not len(r):
            continue
        r = r.iloc[0]
        jac = ""
        if k == "bootstrap" and m in cw:
            jac = f"{cw[m]['n_ge_0.75']} / {cw[m]['n_0.5_to_0.75']} / {cw[m]['n_lt_0.5']}"
        rows.append([kind[k], name[m], f(r.k_mean, 1),
                     f"{f(r.ari_mean, 2)} ± {f(r.ari_std, 2)}" if pd.notna(r.ari_mean) else "",
                     f(r.ami_mean, 2), f(r.vi_bits_mean, 2), f"{f(r.pairwise_ari_mean, 2)} ({f(r.pairwise_ari_min, 2)})",
                     f(r.centroid_shift_mean, 2), f(r.get("mi_rank_tau_mean", np.nan), 2), jac])
    cols = ["Replicates", "Method", "K", "ARI", "AMI", "VI", "Pair. ARI", "Shift", "τ", "Jaccard groups"]
    save(pd.DataFrame(rows, columns=cols), "T_stability", ["l", "l"] + ["r"] * 8)


def t_transfer():
    s = js("E5/summary.json")
    wd = js("E5/weight_drift.json") or {}
    rows = []
    for period, lab in (("TEST", "Test"), ("FUTURE", "Prospective")):
        if period not in s:
            continue
        for m, mn in (("ZSH", "ZSH"), ("KMeans++ K*", "K-means++")):
            a = s[period][m]
            key = "runes_transfer_zsh" if m == "ZSH" else "runes_transfer_kmu"
            ru = s[period].get(key)
            rr = s[period].get("runes_refit_zsh") if m == "ZSH" else None
            rows.append([lab, mn, a["transfer_profiles_used"], a["refit_k"], f(a["ari"], 2), f(a["ami"], 2),
                         f(a["mean_best_jaccard"], 2),
                         f"{a['n_best_jaccard_ge_0.75']} / {a['n_best_jaccard_lt_0.5']}",
                         f(wd.get(period, {}).get("kendall_tau", np.nan), 2) if m == "ZSH" else "",
                         ru["profiles_for_80pct"] if ru else "",
                         f"{rr['profiles_runes_ge_90pct']} ({pct(rr['runes_in_ge_90pct_profiles'], 0)}%)" if rr else ""])
    cols = ["Period", "Method", "Profiles", "Refit K", "ARI", "AMI", "Jaccard", "≥ 0.75 / < 0.5",
            "τ", "Runes in 80%", "Refit ≥ 90% Runes"]
    save(pd.DataFrame(rows, columns=cols), "T_transfer", ["l", "l"] + ["r"] * 9)


def t_concentration():
    rows = []
    for period, lab in (("test", "Test"), ("future", "Prospective")):
        p = R("E6") / f"{period}_independent.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        z = d[d.method == "ZSH"].set_index("target")
        k = d[d.method == "K-means++ (K*)"].set_index("target")
        for t in z.index:
            a = z.loc[t]
            b = k.loc[t]
            rows.append([lab, TARGET.get(t, t), n(a.positives), pct(a.base_rate, 2),
                         f"{f(a.ap_lift, 1)} ({ci(a.ap_lift_lo, a.ap_lift_hi, 1)})",
                         f(1 / a.base_rate, 1) if a.base_rate > 0.01 else n(1 / a.base_rate),
                         pct(a.ap, 1), pct(a["prec@0.25"], 1), int(a["clusters@0.25"]),
                         f(b.ap_lift, 1), f"{fa(-b.d_ap_lift)} ({ci(-b.d_ap_lift_hi, -b.d_ap_lift_lo, 2 if abs(b.d_ap_lift) < 1 else 1)})",
                         pval(b.d_ap_lift_p_holm)])
    cols = ["Period", "Annotation", "Positives", "Base (%)", "ZSH AP lift (95% CI)", "Maximum lift",
            "Attained (%)", "Prec. (%)", "Profiles", "KM++", "Difference (95% CI)", "p"]
    save(pd.DataFrame(rows, columns=cols), "T_concentration", ["l", "l"] + ["r"] * 10)


def t_heuristic():
    s = js("E7/summary.json")
    if not s:
        return
    rows = []
    if "D1" in s:
        e = s["D1"]["estimates"]
        rows += [["2022–2024 sample (stratified, n = "
                  f"{s['D1']['n_flagged']:,} + {s['D1']['n_unflagged']:,})",
                  "Precision of the count rule", f"{pct(e['precision_rule']['estimate'])} "
                  f"({pct(e['precision_rule']['ci'][0])}–{pct(e['precision_rule']['ci'][1])})"],
                 ["", "Recall of the count rule", f"{pct(e['recall_rule']['estimate'])} "
                  f"({pct(e['recall_rule']['ci'][0])}–{pct(e['recall_rule']['ci'][1])})"],
                 ["", "Prevalence of equal-output CoinJoins", f"{pct(e['prevalence']['estimate'], 2)} "
                  f"({pct(e['prevalence']['ci'][0], 2)}–{pct(e['prevalence']['ci'][1], 2)})"],
                 ["", "Transactions with a GraphSense coinjoin tag", str(s["D1"]["coinjoin_tag_matches"])]]
    if "FUTURE" in s:
        fw = s["FUTURE"]["design_weighted"]
        lab = f"Prospective sample (all {s['FUTURE']['rows']:,} transactions, weighted)"
        first = True
        for k, name in (("precision_rule", "Precision of the count rule"), ("recall_rule", "Recall of the count rule"),
                        ("prevalence_eocj", "Prevalence of equal-output CoinJoins"),
                        ("prevalence_rule", "Share meeting the count rule")):
            if k in fw:
                d = 2 if k.startswith("prevalence") else 1
                ci_ = s["FUTURE"].get("design_weighted_ci", {}).get(k)
                txt = pct(fw[k], d) + (f" ({pct(ci_[0], d)}–{pct(ci_[1], d)})" if ci_ else "")
                rows.append([lab if first else "", name, txt])
                first = False
        rows.append(["", "Transactions with a GraphSense coinjoin tag", str(s["FUTURE"]["coinjoin_tag_matches"])])
    save(pd.DataFrame(rows, columns=["Data", "Quantity", "Estimate, % (95% CI)"]), "T_heuristic", ["l", "l", "r"])


def t_elliptic():
    c = pd.read_csv(R("E8") / "concentration.csv")
    s = js("E8/summary.json")
    name = {"ZSH": "ZSH", "K-means++ (K)": "K-means++", "rank-power K-means (K)": "Rank-power, no refinement",
            "uniform + refinement": "Uniform + refinement"}
    rows = []
    for _, r in c.iterrows():
        rows.append([r.setting, name.get(r.method, r.method), int(r.k),
                     f"{f(r.ap_lift)} ({ci(r.ap_lift_lo, r.ap_lift_hi)})", pct(r.ap, 1),
                     f"{f(r['enrich@0.25'])} ({ci(r['enrich@0.25_lo'], r['enrich@0.25_hi'])})",
                     pct(r["prec@0.25"]),
                     "" if pd.isna(r.get("d_ap_lift")) else
                     f"{f(-r.d_ap_lift)} ({ci(-r.d_ap_lift_hi, -r.d_ap_lift_lo)})",
                     pval(r.get("d_ap_lift_p_holm", np.nan))])
    for m in s.get("failed_baselines", []):
        rows.append([m.split(", ")[-1], "Gaussian mixture (diag.)", "", "failed to fit", "", "", "", "", ""])
    rf = s["random_forest_reference"]
    rows.append(["AF-165", "Random forest (supervised)", "", f(rf["ap"] / rf["base_rate"]), pct(rf["ap"], 1), "",
                 f"{pct(rf['precision'])} (recall {pct(rf['recall'])})", "", ""])
    cols = ["Features", "Method", "K", "AP lift (95% CI)", "Attained (%)", "Enrichment (95% CI)",
            "Prec. (%)", "Difference (95% CI)", "p"]
    save(pd.DataFrame(rows, columns=cols), "T_elliptic", ["l", "l", "r", "r", "r", "r", "r", "r", "r"])


def t_atypicality():
    s = js("E9/summary.json")
    rows = []
    sname = {"IF (rank-power space)": "Isolation Forest, ZSH space",
             "IF (unweighted space)": "Isolation Forest, unweighted",
             "LOF (rank-power space)": "Local outlier factor, ZSH space",
             "distance to ZSH centroid": "Distance to ZSH centroid"}
    dec = {"supported (not positively associated)": "H6 supported",
           "contradicted (positively associated)": "H6 contradicted", "inconclusive": "inconclusive"}
    for k, v in s["elliptic"].items():
        rows.append(["Elliptic test steps", sname.get(k, k), "illicit", n(v["positives"]),
                     f"{f(v['roc_auc'], 3)} ({ci(*v['roc_auc_ci'], 3)})", f(v["pr_auc"], 3),
                     pct(v["base_rate"]), dec.get(v["h6_decision"], v["h6_decision"])])
    tname = {"runes": "Runes", "exchange_tag": "exchange tag", "P2WSH_inputs": "P2WSH inputs",
             "eocj": "equal-output CoinJoin"}
    for period, d in s["bitcoin"].items():
        for t, v in d.items():
            rows.append([{"TEST": "Bitcoin test", "FUTURE": "Bitcoin prospective"}[period],
                         "Isolation Forest, ZSH space", tname.get(t, t), n(v["positives"]),
                         f"{f(v['roc_auc'], 3)} ({ci(*v['roc_auc_ci'], 3)})", f(v["pr_auc"], 3),
                         pct(v["base_rate"]), "exploratory"])
    cols = ["Data", "Score", "Target", "Positives", "ROC-AUC (95% CI)", "PR-AUC", "Base (%)", "Reading"]
    save(pd.DataFrame(rows, columns=cols), "T_atypicality", ["l", "l", "l", "r", "r", "r", "r", "l"])


VARIANT = {"reference": "Reference (primary settings)", "s=0.5": "s = 0.5", "s=1.0": "s = 1", "s=2.0": "s = 2",
           "s=3.0": "s = 3", "K0=10": "K₀ = 10", "K0=20": "K₀ = 20", "K0=40": "K₀ = 40", "K0=60": "K₀ = 60",
           "cap=0.05": "c = 5%", "cap=0.15": "c = 15%", "cap=0.2": "c = 20%", "cap=None": "no refinement",
           "depth=6": "depth 6", "upsampled (v1 style)": "upsampled corpus", "sample-weighted": "sample weights",
           "init=k-means++": "K-means++ start", "init=semantic seeds": "semantic seeds",
           "init=seed-Ward blend": "seed–Ward blend"}


def t_sensitivity():
    g = pd.read_csv(R("E10") / "variants_geometry.csv")
    c = pd.read_csv(R("E10") / "variants_independent.csv")
    rows = []
    for _, r in g.iterrows():
        d = c[c.method == r.variant]
        hi = lo = ""
        if r.variant != "reference":
            hi = int(((d.d_ap_lift_lo > 0) & (d.d_ap_lift_p_holm < 0.05)).sum())
            lo = int(((d.d_ap_lift_hi < 0) & (d.d_ap_lift_p_holm < 0.05)).sum())
        rows.append([VARIANT.get(r.variant, r.variant), int(r.k), pct(r.max_share_fit), pct(r.max_share_test),
                     f(r.common_silhouette, 3), f(d.ap_lift.median(), 2),
                     f"{hi} / {lo}" if hi != "" else ""])
    cols = ["Variant", "K", "Largest, fit (%)", "Largest, test (%)", "Silh.", "Median lift",
            "Higher / lower"]
    save(pd.DataFrame(rows, columns=cols), "T_sensitivity", ["l"] + ["r"] * 6)
    # appendix: AP lift per annotation and variant
    piv = c.pivot(index="method", columns="target", values="ap_lift")
    piv = piv.reindex([v for v in g.variant if v in piv.index])
    tcols = [t for t in TARGET_SHORT if t in piv.columns]
    rows = [[VARIANT.get(v, v)] + [f(piv.loc[v, t], 1) for t in tcols] for v in piv.index]
    save(pd.DataFrame(rows, columns=["Variant"] + [TARGET_SHORT[t] for t in tcols]), "T_sensitivity_targets")


def t_loo():
    loo = pd.read_csv(R("E10") / "loo_seeded.csv")
    arm = {"unseeded ZSH": "unseeded", "seeded, all families": "all seeds",
           "seeded, family withheld": "family withheld"}
    rows = []
    fam_name = {"Coinbase": "Coinbase", "ManyInManyOut": "Many inputs and outputs",
                "SingleInFanOut": "Single input, fan-out", "FanIn": "Fan-in", "FanOut": "Fan-out",
                "OneInOneOut": "One input, one output", "OpReturn": "OP_RETURN", "RBF": "Replace-by-fee"}
    feats = set(CFG["features"]["selected"])
    for fam, d in loo.groupby("family", sort=False):
        d = d.set_index("method")
        fr = d.feature_removed.fillna("").iloc[0]
        r = [fam_name.get(fam, fam), (NICE.get(fr, fr) if fr in feats else "none (not an input)") if fr else "none"]
        for a in arm:
            v = d.loc[a]
            r.append(f"{f(v.ap_lift, 2)}")
        w, s_ = d.loc["seeded, family withheld"], d.loc["seeded, all families"]
        r.append(f"{f(w.ap_lift - s_.ap_lift, 2)}")
        rows.append(r)
    cols = ["Rule family", "Feature removed", "Unseeded", "All seeds", "Family withheld",
            "Withheld − all seeds"]
    save(pd.DataFrame(rows, columns=cols), "T_loo", ["l", "l", "r", "r", "r", "r"])


def t_representatives():
    """Appendix: the transaction closest to each profile centroid, with the profile descriptor."""
    d = pd.read_csv(R("E1") / "profiles_dev.csv")
    rows = []
    for _, r in d.iterrows():
        t = str(r.representative_txid)
        rows.append([f"P{int(r.profile):02d}", pct(r.share), r.descriptor, t[:32] + " " + t[32:]])
    save(pd.DataFrame(rows, columns=["Profile", "Share (%)", "Description of its members",
                                     "Transaction closest to the centroid"]),
         "T_representatives", ["l", "r", "L", "L"])


def t_proxy_k():
    """Sensitivity to the size of the proxy partition (E11)."""
    g = pd.read_csv(R("E11") / "variants_geometry.csv")
    c = pd.read_csv(R("E11") / "variants_independent.csv")
    ref = f"K_p={js('E11/summary.json')['proxy_k_reference']} (reference)"
    rows = []
    for _, r in g.sort_values("proxy_k").iterrows():
        d = c[c.method == r.variant]
        hi = lo = ""
        if r.variant != ref:
            hi = int(((d.d_ap_lift_lo > 0) & (d.d_ap_lift_p_holm < 0.05)).sum())
            lo = int(((d.d_ap_lift_hi < 0) & (d.d_ap_lift_p_holm < 0.05)).sum())
        rows.append([str(int(r.proxy_k)) + (" (reference)" if r.variant == ref else ""),
                     int(r.k), f(r.kendall_tau_vs_reference, 2), NICE.get(r.top_feature, r.top_feature),
                     f(r.common_silhouette, 3), f(d.ap_lift.median(), 2),
                     f(d[d.target == "L2:P2PKH"].ap_lift.iloc[0], 1),
                     f(d[d.target == "L3:omni"].ap_lift.iloc[0], 1),
                     f"{hi} / {lo}" if hi != "" else ""])
    cols = ["Proxy clusters", "K", "τ", "Top-ranked feature", "Silh.", "Median lift", "P2PKH", "Omni",
            "Higher / lower"]
    save(pd.DataFrame(rows, columns=cols), "T_proxy_k", ["l", "r", "r", "l", "r", "r", "r", "r", "r"])


def t_support():
    """Appendix: share of each profile per period with block-bootstrap intervals, and Jaccard spread."""
    d = pd.read_csv(R("E12") / "profile_support.csv")
    rows = []
    for _, r in d.iterrows():
        row = [f"P{int(r.profile):02d}"]
        for key in ("deve", "test", "pros"):
            if f"{key}_share" in d.columns:
                row.append(f"{pct(r[f'{key}_share'], 2)} ({pct(r[f'{key}_lo'], 2)}–{pct(r[f'{key}_hi'], 2)})")
        row.append(f"{f(r.jaccard_mean, 2)} ({f(r.jaccard_p05, 2)}–{f(r.jaccard_p95, 2)})")
        rows.append(row)
    cols = ["Profile", "Development (%)", "Test (%)", "Prospective (%)", "Jaccard (5th–95th)"]
    save(pd.DataFrame(rows, columns=cols[:len(rows[0])]), "T_support",
         ["l"] + ["r"] * (len(rows[0]) - 1))


def t_cjsource():
    """External CoinJoin labels against the two rules (E14)."""
    d = pd.read_csv(R("E14") / "coinjoin_source.csv")
    rows = []
    for _, r in d.iterrows():
        cj = int(r.known_coinjoins)
        rows.append([r.period, n(r.transactions), str(cj),
                     "" if not cj else f"{pct(r.count_rule_recall)} ({pct(r.count_rule_recall_lo)}–"
                                       f"{pct(r.count_rule_recall_hi)})",
                     "" if pd.isna(r.get("equal_output_recall")) else
                     f"{pct(r.equal_output_recall)} ({pct(r.equal_output_recall_lo)}–"
                     f"{pct(r.equal_output_recall_hi)})"])
    save(pd.DataFrame(rows, columns=["Period", "Transactions", "On the external list",
                                     "Count rule recall (%)", "Equal-output recall (%)"]),
         "T_cjsource", ["l", "r", "r", "r", "r"])


def t_matching():
    """Can profiles be followed across refits (E16)?"""
    d = pd.read_csv(R("E16") / "matching.csv")
    rows = [[r.period, r.method, int(r.profiles), int(r.refit_clusters), f(r.median_jaccard, 2),
             f(r.mean_jaccard, 2), str(int(r["matched_0.5"])), pct(r["share_matched_0.5"], 1),
             str(int(r["matched_0.75"]))] for _, r in d.iterrows()]
    save(pd.DataFrame(rows, columns=["Period", "Method", "Profiles", "Refit clusters", "Median J",
                                     "Mean J", "Matched at 0.5", "Their share (%)", "Matched at 0.75"]),
         "T_matching", ["l", "l", "r", "r", "r", "r", "r", "r", "r"])


def t_oracle():
    """Oracle-weight upper bound (E15): concentration attained with weights from the annotation."""
    c = pd.read_csv(R("E15") / "oracle_concentration.csv")
    ref = "reference (unsupervised)"
    rows = []
    for t in [x for x in TARGET if x in set(c.target)]:
        g = c[c.target == t].set_index("method")
        if ref not in g.index:
            continue
        base = g.loc[ref, "base_rate"]
        own_rp = f"oracle rank-power: {t}"
        own_mi = f"oracle MI-direct: {t}"
        best_other = g.drop(index=[i for i in (ref, own_rp, own_mi) if i in g.index]).ap_lift.max()
        rows.append([TARGET[t], pct(base, 2), f(g.loc[ref, "ap_lift"], 1), pct(g.loc[ref, "ap"], 1),
                     f(g.loc[own_rp, "ap_lift"], 1) if own_rp in g.index else "",
                     pct(g.loc[own_rp, "ap"], 1) if own_rp in g.index else "",
                     f(g.loc[own_mi, "ap_lift"], 1) if own_mi in g.index else "",
                     f(best_other, 1) if pd.notna(best_other) else ""])
    save(pd.DataFrame(rows, columns=["Annotation", "Base (%)", "ZSH lift", "ZSH attained (%)",
                                     "Oracle lift", "Oracle attained (%)", "Oracle MI-direct lift",
                                     "Best other oracle"]),
         "T_oracle", ["l", "r", "r", "r", "r", "r", "r", "r"])


def fmt_matched(g):
    return str(int(g["n_matched_ge_0.5"].iloc[0])) + "/" + str(int(g.profiles.iloc[0]))


BENCH_SHORT = {"ZSH": "ZSH", "K-means++ (K*)": "KM++", "MiniBatchKMeans": "MBK", "GMM (diag)": "GMM",
               "BIRCH": "BIRCH", "Ward (sample + NC)": "Ward", "VKV-partial (K*)": "VKV"}
BENCH_LONG = {"ZSH": "ZSH", "K-means++ (K*)": "K-means++", "MiniBatchKMeans": "Mini-batch K-means",
              "GMM (diag)": "Gaussian mixture (diag.)", "BIRCH": "BIRCH",
              "Ward (sample + NC)": "Ward (sample)", "VKV-partial (K*)": "Vlahavas et al., partial"}


def t_bench_battery():
    """E17: every clustering family through the same battery."""
    fa_ = pd.read_csv(R("E17") / "fit_and_assignment.csv").set_index("method")
    st = pd.read_csv(R("E17") / "stability_by_method.csv").set_index("method")
    tr = pd.read_csv(R("E17") / "transfer_by_method.csv")
    rows = []
    for m in BENCH_SHORT:
        if m not in fa_.index:
            continue
        t = tr[(tr.method == m) & (tr.period == "test")]
        p = tr[(tr.method == m) & (tr.period == "future")]
        rows.append([BENCH_LONG[m], pct(fa_.loc[m, "test_max_share"], 1), pct(fa_.loc[m, "future_max_share"], 1),
                     f(st.loc[m, "ari_mean"], 2) if m in st.index else "",
                     f(t.ari.iloc[0], 2) if len(t) else "",
                     f(p.ari.iloc[0], 2) if len(p) else "",
                     fmt_matched(t) if len(t) else "",
                     fmt_matched(p) if len(p) else ""])
    save(pd.DataFrame(rows, columns=["Method", "Largest, test (%)", "Largest, prosp. (%)", "Refit ARI",
                                     "ARI, test", "ARI, prosp.", "Matched, test", "Matched, prosp."]),
         "T_bench_battery", ["l", "r", "r", "r", "r", "r", "r", "r"])


def t_bench_attained():
    """E17: share of the attainable concentration reached by each family, test period."""
    c = pd.read_csv(R("E17") / "test_independent.csv")
    methods = [m for m in BENCH_SHORT if m in set(c.method)]
    rows = []
    for t in [x for x in TARGET if x in set(c.target)]:
        g = c[c.target == t].set_index("method")
        rows.append([TARGET[t], pct(g.base_rate.iloc[0], 2), f(1.0 / g.base_rate.iloc[0], 0)]
                    + [pct(g.loc[m, "ap"], 1) if m in g.index else "" for m in methods])
    med = c.groupby("method").ap.median()
    rows.append(["Median", "", ""] + [pct(med.get(m, np.nan), 1) for m in methods])
    save(pd.DataFrame(rows, columns=["Annotation", "Base (%)", "Ceiling"] + [BENCH_SHORT[m] for m in methods]),
         "T_bench_attained", ["l"] + ["r"] * (2 + len(methods)))


def t_ceiling():
    """E18: what the same twelve features give a supervised model, against the profiles."""
    c = pd.read_csv(R("E18") / "feature_ceiling.csv")
    piv = c.pivot(index="target", columns="method", values="ap")
    base = c.groupby("target").base_rate.first()
    rows = []
    for t in [x for x in TARGET if x in piv.index]:
        prof = piv.loc[t, "ZSH"]
        ip = piv.loc[t].get("supervised in-period", np.nan)
        tr = piv.loc[t].get("supervised transfer", np.nan)
        rows.append([TARGET[t], pct(base[t], 2), f(1.0 / base[t], 0), pct(prof, 1),
                     pct(ip, 1), pct(tr, 1), fa(100 * (ip - prof), 0)])
    save(pd.DataFrame(rows, columns=["Annotation", "Base (%)", "Ceiling", "Profiles (%)",
                                     "Supervised (%)", "A year earlier (%)", "Gap"]),
         "T_ceiling", ["l", "r", "r", "r", "r", "r", "r"])


BUILDERS = [elliptic_counts, t_data, t_rules, t_features, t_annotations, t_profiles, t_methods, t_factorial,
            t_contrasts, t_stability, t_transfer, t_concentration, t_heuristic, t_elliptic, t_atypicality,
            t_sensitivity, t_proxy_k, t_cjsource, t_oracle, t_matching, t_bench_battery,
            t_bench_attained, t_ceiling, t_loo, t_representatives, t_support]


def main():
    only = set(sys.argv[1:])
    for b in BUILDERS:
        if only and b.__name__ not in only:
            continue
        try:
            b()
        except (FileNotFoundError, KeyError, TypeError) as e:
            print(f"skip {b.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
