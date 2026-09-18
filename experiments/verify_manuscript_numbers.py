"""Check the numbers written in manuscript.md against the result files.

Every check recomputes a quantity from `results/` and asserts that the string the
article uses for it occurs in manuscript.md. Rounding variants can be given as a
tuple of acceptable strings. Run after `assemble.py`:

    python verify_numbers.py            # report
    python verify_numbers.py -v         # also list the checks that pass
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent  # repository root
RES = HERE / "results"
MS = Path(os.environ.get("ZSH_MANUSCRIPT", HERE.parent / "Fintech MDPI" / "Fintech MDPI Revised" / "v2_work" / "manuscript.md"))
TEXT = MS.read_text(encoding="utf-8") if MS.exists() else ""
VERBOSE = "-v" in sys.argv


def js(p):
    return json.load(open(RES / p, encoding="utf-8"))


def csv(p, **kw):
    return pd.read_csv(RES / p, **kw)


CHECKS = []


def check(label, *variants):
    """The article must contain at least one of `variants` (strings)."""
    CHECKS.append((label, [str(v) for v in variants]))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- data (E0)
sp = csv("E0/splits.csv").set_index("split")
check("DEV rows", f"{sp.loc[0, 'rows']:,}")
check("DEV blocks", f"{sp.loc[0, 'blocks']:,}")
check("TEST rows", f"{sp.loc[1, 'rows']:,}")
check("TEST blocks", f"{sp.loc[1, 'blocks']:,}")
check("excluded rows", f"{sp.loc[9, 'rows']:,}", str(sp.loc[9, "rows"]))
bb = js("E0/base_build.json")
for key, name in (("rows_raw", "raw rows"), ("blocks_raw", "raw blocks")):
    if key in bb:
        check(name, f"{bb[key]:,}")
fm = js("E0/future_manifest.json")
check("prospective rows", f"{fm['rows']:,}")
check("prospective blocks", f"{fm['blocks_with_page']:,}")
check("prospective months", str(len(fm["months"])))
eq = js("E0/api_equivalence.json")
check("equivalence sample", f"{eq['matched']:,}")
der = js("E0/rule_derivability.json")
check("rule derivability floor", pct(min(v["accuracy"] for v in der.values())))
ax = js("E0/audit_extra.json")
check("value_difference equals fee (all rows)", pct(ax["all_rows"]["value_difference_equals_fee_share"], 2))
check("value_difference equals fee (DEV)", pct(ax["development"]["value_difference_equals_fee_share"], 2))
check("rows with zero output value", str(ax["all_rows"]["rows_with_zero_output_value"]))
fs = js("E0/feature_selection.json")
check("features kept", str(len(fs["selected"])))
check("coinbase share", pct(1 - fs["mode_share"]["has_coinbase"], 3))

# ---------------------------------------------------------------- weights, primary fit (E1)
w = csv("E1/weights.csv").sort_values("rank")
for _, r in w.head(5).iterrows():
    check(f"weight of {r.feature}", f"{r.weight:.3f}")
check("smallest weight", f"{w.weight.min():.3f}")
e1 = js("E1/summary.json")
check("K", str(e1["k"]))
check("largest profile (DEV)", pct(e1["max_share"]))
check("largest cluster before refinement", pct(e1["splits"][0]["share"]))
check("fit seconds", f"{e1['timing_s']['total']:.0f} s", f"{round(e1['timing_s']['total'])} s")
check("weighting seconds", f"{e1['timing_s']['weighting']:.0f} s")
check("initialisation seconds", f"{e1['timing_s']['init']:.0f} s")

# ---------------------------------------------------------------- method comparison (E2)
g = csv("E2/methods_intrinsic_timing.csv").set_index("method")
for m, name in (("ZSH", "ZSH"), ("KMeans++", "K-means++"), ("MiniBatchKMeans", "mini-batch"),
                ("Ward (sample + NC)", "Ward"), ("GMM (diag)", "Gaussian mixture"), ("BIRCH", "BIRCH")):
    check(f"Silhouette of {name}", f"{g.loc[m, 'common_silhouette']:.3f}")
check("ZSH Davies-Bouldin", f"{g.loc['ZSH', 'common_dbi']:.2f}")
check("K-means++ Davies-Bouldin", f"{g.loc['KMeans++', 'common_dbi']:.2f}")
check("ZSH Silhouette (own space)", f"{g.loc['ZSH', 'own_silhouette']:.3f}")
check("VKV Silhouette (own space)", f"{g.loc['VKV-partial (k=5)', 'own_silhouette']:.3f}")
check("HDBSCAN noise share", pct(g.loc["HDBSCAN (own sample)", "noise_share"], 0))
check("ZSH fit seconds (E2)", f"{g.loc['ZSH', 'fit_seconds']:.0f} s")
check("K-means++ fit seconds (E2)", f"{g.loc['KMeans++', 'fit_seconds']:.0f} s")

# ---------------------------------------------------------------- factorial (E3)
h1 = csv("E3/H1_primary_A3_vs_A4_independent.csv")
h1 = h1[h1.method == "A3"].set_index("target")
for t, name in (("L2:coinbase", "coinbase"), ("L2:P2PKH", "P2PKH"), ("L3:omni", "Omni"),
                ("L2:P2SH", "P2SH"), ("L3:other_opreturn", "other OP_RETURN")):
    v = h1.loc[t, "d_ap_lift"]
    check(f"H1 difference, {name}", f"+{v:.1f}", f"{v:.1f}", f"+{v:.2f}")
h2 = csv("E3/H2_primary_A1_vs_A3_independent.csv")
h2 = h2[h2.method == "A1"]
check("largest H2 difference", f"{h2.d_ap_lift.abs().max():.2f}")
arms = csv("E3/arms_geometry.csv").set_index("arm")
check("A1 Silhouette", f"{arms.loc['A1', 'common_silhouette']:.3f}")
check("A3 Silhouette", f"{arms.loc['A3', 'common_silhouette']:.3f}")
allc = csv("E3/all_vs_A1_independent.csv")
med = allc.groupby("method").ap_lift.median()
check("A6 median AP lift", f"{med['A6']:.2f}")
check("A1 median AP lift", f"{med['A1']:.2f}")
check("A7 median AP lift", f"{med['A7']:.2f}")
a9 = allc[allc.method == "A9"].set_index("target")
a4 = allc[allc.method == "A4"].set_index("target")
check("A9 P2PKH", f"{a9.loc['L2:P2PKH', 'ap_lift']:.1f}")
check("A4 P2PKH", f"{a4.loc['L2:P2PKH', 'ap_lift']:.1f}")

# ---------------------------------------------------------------- stability (E4)
s4 = csv("E4/summary_by_method.csv", keep_default_na=False, na_values=[""]).set_index(["kind", "method"])
p4 = csv("E4/pairwise.csv", keep_default_na=False, na_values=[""]).set_index(["kind", "method"])
check("ZSH bootstrap ARI", f"{s4.loc[('bootstrap', 'ZSH'), 'ari_mean']:.2f}")
check("ZSH bootstrap ARI sd", f"{s4.loc[('bootstrap', 'ZSH'), 'ari_std']:.2f}")
check("ZSH seed ARI", f"{s4.loc[('seed', 'ZSH'), 'ari_mean']:.2f}")
check("K-means++ bootstrap ARI", f"{s4.loc[('bootstrap', 'KMeans++ K*'), 'ari_mean']:.2f}")
check("ZSH pairwise ARI", f"{p4.loc[('bootstrap', 'ZSH'), 'pairwise_ari_mean']:.2f}")
check("null pairwise ARI", f"{p4.loc[('null', 'ZSH'), 'pairwise_ari_mean']:.2f}")
check("MI rank tau (bootstrap)", f"{s4.loc[('bootstrap', 'ZSH'), 'mi_rank_tau_mean']:.2f}")
check("mean K over refits", f"{s4.loc[('bootstrap', 'ZSH'), 'k_mean']:.1f}")
cw = js("E4/summary.json")["clusterwise"]
check("stable profiles", str(cw["ZSH"]["n_ge_0.75"]))
check("unstable profiles", str(cw["ZSH"]["n_lt_0.5"]))
check("centroid shift ZSH", f"{s4.loc[('bootstrap', 'ZSH'), 'centroid_shift_mean']:.2f}")

# ---------------------------------------------------------------- transfer (E5)
e5 = js("E5/summary.json")
wd = js("E5/weight_drift.json")
for per, lab in (("TEST", "test"), ("FUTURE", "prospective")):
    a = e5[per]["ZSH"]
    check(f"{lab} transfer ARI", f"{a['ari']:.2f}")
    check(f"{lab} mean Jaccard", f"{a['mean_best_jaccard']:.2f}")
    check(f"{lab} refit K", str(a["refit_k"]))
    check(f"{lab} K-means++ refit ARI", f"{e5[per]['KMeans++ K*']['ari']:.2f}")
    check(f"{lab} Kendall tau", f"{wd[per]['kendall_tau']:.2f}")
f5 = csv("E5/drift_future.csv")
check("largest prospective profile", pct(f5.share_period.max()))
check("top three prospective profiles", pct(f5.share_period.nlargest(3).sum()))
check("prospective profiles below 1%", str(int((f5.share_period < 0.01).sum())))
check("prospective JS(L2) below 0.2", str(int((f5.js_L2 < 0.2).sum())))
d5 = csv("E5/drift_test.csv")
check("test JS(L2) below 0.2", str(int((d5.js_L2 < 0.2).sum())))
mon = csv("E5/monthly_profile_shares.csv").pivot(index="month", columns="profile", values="share").fillna(0)
check("P10 share in Aug 2026", pct(mon.loc["2026-08"].max()))

# ---------------------------------------------------------------- concentration (E6)
for per, lab in (("test", "test"), ("future", "prospective")):
    d = csv(f"E6/{per}_independent.csv")
    z = d[d.method == "ZSH"].set_index("target")
    k = d[d.method == "K-means++ (K*)"].set_index("target")
    for t in ("L2:P2PKH", "L4:exchange", "L2:mixed"):
        if t in z.index:
            check(f"{lab} AP lift {t}", f"{z.loc[t, 'ap_lift']:.1f}", f"{z.loc[t, 'ap_lift']:.2f}")
            check(f"{lab} K-means++ AP lift {t}", f"{k.loc[t, 'ap_lift']:.1f}", f"{k.loc[t, 'ap_lift']:.2f}")
    if per == "test":
        for t in ("L3:omni", "L2:coinbase", "L3:other_opreturn"):
            check(f"test AP lift {t}", f"{z.loc[t, 'ap_lift']:.1f}")
        check("test exchange precision at 25%", pct(z.loc["L4:exchange", "prec@0.25"]))
        check("number of ZSH-higher annotations",
              str(int(((k.d_ap_lift < 0) & (k.d_ap_lift_p_holm < 0.05)).sum())))
    else:
        check("prospective P2WSH AP lift", f"{z.loc['L2:P2WSH', 'ap_lift']:.1f}")
        check("prospective P2TR AP lift", f"{z.loc['L2:P2TR', 'ap_lift']:.1f}")
        check("prospective exchange p", f"{k.loc['L4:exchange', 'd_ap_lift_p_holm']:.3f}")
sk = js("E6/summary.json")["FUTURE"]["skipped"]
for t, n in sk.items():
    if n:
        check(f"skipped {t} positives", f"({n})", str(n))

# ---------------------------------------------------------------- CoinJoin rule (E7)
e7 = js("E7/summary.json")
d1 = e7["D1"]
check("flagged in D1", f"{d1['population_flagged']:,}")
check("D1 strata size", f"{d1['n_flagged']:,}")
est = d1["estimates"]
check("rule precision D1", pct(est["precision_rule"]["estimate"]))
check("rule recall D1", pct(est["recall_rule"]["estimate"], 0), f"{100 * est['recall_rule']['estimate']:.1f}%")
check("EO-CJ prevalence D1", pct(est["prevalence"]["estimate"], 2))
st = csv("E7/d1_strata.csv")
fl = st[st["sample"] == "e7_flagged"].set_index("split")
check("flagged EO-CJ count", str(int(fl.eocj.sum())))
check("flagged rate DEV", pct(fl.loc[0, "eocj_rate"]))
check("flagged rate TEST", pct(fl.loc[1, "eocj_rate"]))
fu = e7["FUTURE"]
check("prospective EO-CJ count", str(fu["confusion"]["rule&eocj"] + fu["confusion"]["~rule&eocj"]))
check("prospective rule count", f"{fu['confusion']['rule&eocj'] + fu['confusion']['rule&~eocj']:,}")
check("prospective rule precision", pct(fu["design_weighted"]["precision_rule"]))
check("prospective rule recall", pct(fu["design_weighted"]["recall_rule"]))
check("prospective EO-CJ prevalence", pct(fu["design_weighted"]["prevalence_eocj"], 3))
check("prospective rule prevalence", pct(fu["design_weighted"]["prevalence_rule"], 2))

# ---------------------------------------------------------------- Elliptic (E8)
e8 = js("E8/summary.json")
c8 = csv("E8/concentration.csv")
z8 = c8[(c8.setting == "AF-165") & (c8.method == "ZSH")].iloc[0]
check("Elliptic base rate", pct(z8.base_rate))
check("Elliptic AP lift", f"{z8.ap_lift:.2f}")
check("Elliptic AP lift CI", f"{z8.ap_lift_lo:.2f}–{z8.ap_lift_hi:.2f}")
check("Elliptic precision at 25%", pct(z8["prec@0.25"]))
check("Elliptic K", str(int(z8.k)))
check("Elliptic largest cluster", pct(e8["AF-165"]["max_share"], 0))
check("Elliptic largest weight", f"{e8['AF-165']['weights_top10'][0][1]:.2f}")
for m, name in (("K-means++ (K)", "K-means++"), ("rank-power K-means (K)", "rank-power"),
                ("uniform + refinement", "uniform + refinement")):
    r = c8[(c8.setting == "AF-165") & (c8.method == m)]
    if len(r):
        check(f"Elliptic {name} AP lift", f"{r.iloc[0].ap_lift:.2f}")
lf = c8[(c8.setting == "LF-93") & (c8.method == "ZSH")]
check("Elliptic LF-93 AP lift", f"{lf.iloc[0].ap_lift:.2f}")
rf = e8["random_forest_reference"]
check("random forest precision", f"{rf['precision']:.2f}", pct(rf["precision"], 0))
check("random forest recall", f"{rf['recall']:.2f}", pct(rf["recall"], 0))
check("random forest AP", f"{rf['ap']:.2f}")

# ---------------------------------------------------------------- atypicality (E9)
e9 = js("E9/summary.json")
for k, name in (("IF (rank-power space)", "IF ZSH space"), ("IF (unweighted space)", "IF unweighted"),
                ("LOF (rank-power space)", "LOF"), ("distance to ZSH centroid", "centroid distance")):
    v = e9["elliptic"][k]
    check(f"ROC-AUC {name}", f"{v['roc_auc']:.3f}")
    check(f"ROC-AUC CI {name}", f"{v['roc_auc_ci'][0]:.3f}–{v['roc_auc_ci'][1]:.3f}")
for per in e9["bitcoin"]:
    for t, v in e9["bitcoin"][per].items():
        check(f"ROC-AUC {per} {t}", f"{v['roc_auc']:.3f}")

# ---------------------------------------------------------------- sensitivity (E10)
g10 = csv("E10/variants_geometry.csv").set_index("variant")
c10 = csv("E10/variants_independent.csv").pivot(index="method", columns="target", values="ap_lift")
check("reference K (E10)", str(int(g10.loc["reference", "k"])))
check("reference P2PKH (E10)", f"{c10.loc['reference', 'L2:P2PKH']:.1f}")
check("reference P2SH (E10)", f"{c10.loc['reference', 'L2:P2SH']:.1f}")
check("largest cluster before refinement (E10)", pct(g10.loc["cap=None", "max_share_fit"]))
check("largest test cluster (E10 reference)", pct(g10.loc["reference", "max_share_test"]))
check("cap 5% K", str(int(g10.loc["cap=0.05", "k"])))
check("K0=60 Omni", f"{c10.loc['K0=60', 'L3:omni']:.1f}")
check("K0=60 coinbase", f"{c10.loc['K0=60', 'L2:coinbase']:.1f}")
check("s=0.5 P2PKH", f"{c10.loc['s=0.5', 'L2:P2PKH']:.1f}")
check("s=3 P2PKH", f"{c10.loc['s=3.0', 'L2:P2PKH']:.1f}")
check("s=0.5 P2SH", f"{c10.loc['s=0.5', 'L2:P2SH']:.1f}")
check("s=3 P2SH", f"{c10.loc['s=3.0', 'L2:P2SH']:.1f}")
check("s=0.5 exchange", f"{c10.loc['s=0.5', 'L4:exchange']:.1f}")
check("s=3 exchange", f"{c10.loc['s=3.0', 'L4:exchange']:.1f}")
check("s=0.5 Silhouette", f"{g10.loc['s=0.5', 'common_silhouette']:.3f}")
check("s=3 Silhouette", f"{g10.loc['s=3.0', 'common_silhouette']:.3f}")
check("upsampled K", str(int(g10.loc["upsampled (v1 style)", "k"])))
check("upsampled Silhouette", f"{g10.loc['upsampled (v1 style)', 'common_silhouette']:.2f}".replace("-", "−"))
check("sample-weighted Silhouette", f"{g10.loc['sample-weighted', 'common_silhouette']:.2f}".replace("-", "−"))
check("upsampled P2PKH", f"{c10.loc['upsampled (v1 style)', 'L2:P2PKH']:.1f}")
check("kmeans++ start P2PKH", f"{c10.loc['init=k-means++', 'L2:P2PKH']:.1f}")
check("kmeans++ start exchange", f"{c10.loc['init=k-means++', 'L4:exchange']:.1f}")
check("blend P2SH", f"{c10.loc['init=seed-Ward blend', 'L2:P2SH']:.1f}")
check("blend Omni", f"{c10.loc['init=seed-Ward blend', 'L3:omni']:.1f}")
# ---------------------------------------------------------------- proxy partition size (E11)
g11 = csv("E11/variants_geometry.csv").set_index("proxy_k")
c11 = csv("E11/variants_independent.csv")
for kp in (5, 20, 50):
    check(f"proxy K={kp} tau", f"{g11.loc[kp, 'kendall_tau_vs_reference']:.2f}")
v11 = c11.pivot(index="method", columns="target", values="ap_lift")
check("proxy K=5 P2PKH", f"{v11.loc['K_p=5', 'L2:P2PKH']:.1f}")
check("proxy K=50 P2PKH", f"{v11.loc['K_p=50', 'L2:P2PKH']:.1f}")
check("proxy K=50 Omni", f"{v11.loc['K_p=50', 'L3:omni']:.1f}")
check("proxy K=5 K", str(int(g11.loc[5, "k"])))

loo = csv("E10/loo_seeded.csv")
piv = loo.pivot_table(index="family", columns="method", values="ap_lift")
for fam in ("ManyInManyOut", "SingleInFanOut"):
    check(f"LOO {fam} withheld", f"{piv.loc[fam, 'seeded, family withheld']:.1f}")
    check(f"LOO {fam} all seeds", f"{piv.loc[fam, 'seeded, all families']:.1f}")
    check(f"LOO {fam} unseeded", f"{piv.loc[fam, 'unseeded ZSH']:.1f}")
check("LOO FanOut seeded", f"{piv.loc['FanOut', 'seeded, all families']:.1f}")
check("LOO FanOut unseeded", f"{piv.loc['FanOut', 'unseeded ZSH']:.1f}")

# ---------------------------------------------------------------- report
missing = []
for label, variants in CHECKS:
    hit = next((v for v in variants if v in TEXT), None)
    if hit is None:
        missing.append((label, variants))
    elif VERBOSE:
        print(f"ok    {label}: {hit}")
print(f"\n{len(CHECKS) - len(missing)}/{len(CHECKS)} checks found in manuscript.md")
for label, variants in missing:
    print(f"MISSING  {label}: expected one of {variants}")
sys.exit(1 if missing else 0)
