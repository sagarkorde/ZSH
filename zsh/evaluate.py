"""Evaluation targets and shared evaluation routines."""
import numpy as np
import pandas as pd

from .config import CFG
from .data import L1_NAMES, L2_NAMES, L3_NAMES, TAG_BITS, l1_rule_masks
from .metrics import ConcentrationData, concentration_many, crossfit, holm


def independent_targets(df, include_eocj=False):
    """Annotations that are not clustering inputs (L2-L5)."""
    t = {}
    for code, name in enumerate(L2_NAMES):
        if name == "other":
            continue
        t[f"L2:{name}"] = df["L2"].to_numpy() == code
    for code, name in enumerate(L3_NAMES):
        if name == "none":
            continue
        t[f"L3:{name}"] = df["L3"].to_numpy() == code
    tags = df["tags"].to_numpy()
    for cat in CFG["annotations"]["tag_categories"]:
        t[f"L4:{cat}"] = (tags & TAG_BITS[cat]) > 0
    if include_eocj and "eocj" in df:
        t["L5:eocj"] = df["eocj"].to_numpy() == 1
    return t


def structural_targets(df):
    """Count-rule annotations (derivable from the inputs; reported separately)."""
    m = l1_rule_masks(df["input_count"].to_numpy(), df["output_count"].to_numpy())
    return {f"L1:{k}": v for k, v in m.items()}


def eligible(target, block):
    """Pre-declared minimum: >= min positives in each block-parity fold."""
    mn = CFG["annotations"]["min_positives_per_fold"]
    b = np.asarray(block) % 2
    return int(target[b == 0].sum()) >= mn and int(target[b == 1].sum()) >= mn


def evaluate_targets(label_sets, df, targets, B, seed, reference, log=None,
                     row_weights=None, strata=None):
    """Concentration for every eligible target; paired differences vs reference; Holm on AP lift.

    `row_weights` (design weights) make every estimate a population estimate for the
    sampled frame; `strata` (e.g. month) switches the block bootstrap to a
    month-stratified one. Eligibility stays on *sampled* positives, so the weighted
    and unweighted runs evaluate exactly the same set of targets.
    """
    block = df["block_height"].to_numpy()
    rows, curves_all, skipped = [], {}, {}
    for ti, (tname, tgt) in enumerate(targets.items()):
        if not eligible(tgt, block):
            skipped[tname] = int(tgt.sum())
            continue
        res, curves = concentration_many(label_sets, tgt, block, B, seed + ti, reference=reference,
                                         row_weights=row_weights, strata=strata)
        curves_all[tname] = curves
        for m, r in res.items():
            row = {"target": tname, "method": m, "positives": int(tgt.sum()), "base_rate": r["base_rate"]}
            for k in ("ap", "ap_lift", "enrich@0.10", "enrich@0.25", "enrich@0.50",
                      "prec@0.25", "clusters@0.25", "share@0.25"):
                row[k] = r[k]
            for k, (lo, hi) in r["ci"].items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
            vs = r.get("vs_" + reference) if reference else None
            if vs:
                for k, v in vs.items():
                    row[f"d_{k}"] = v["diff"]
                    row[f"d_{k}_lo"], row[f"d_{k}_hi"] = v["ci"]
                    row[f"d_{k}_p"] = v["p"]
            rows.append(row)
        if log:
            log(f"    {tname}: " + ", ".join(f"{m}={res[m]['ap_lift']:.2f}" for m in res))
    tab = pd.DataFrame(rows)
    if reference and len(tab):
        for m in tab.method.unique():
            if m == reference:
                continue
            sel = tab.method == m
            tab.loc[sel, "d_ap_lift_p_holm"] = holm(tab.loc[sel, "d_ap_lift_p"].to_numpy())
    return tab, curves_all, skipped


def profile_table(df, labels, d2=None, k=None, weights=None):
    """Data-driven description of each profile."""
    k = int(labels.max()) + 1 if k is None else k
    w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=np.float64)
    rows = []
    tot = w.sum()
    for c in range(k):
        m = labels == c
        if not m.any():
            rows.append({"profile": c, "share": 0.0})
            continue
        g = df[m]
        gw = w[m]
        r = {"profile": c, "n": int(m.sum()), "share": float(gw.sum() / tot),
             "med_inputs": float(g.input_count.median()), "med_outputs": float(g.output_count.median()),
             "med_value_sat": float(g.total_input_value.median()),
             "med_fee_rate": float(g.fee_rate_sat_per_vbyte.median()),
             "med_vsize": float(g.vsize.median()),
             "opreturn_share": float(g.has_op_return.mean()), "rbf_share": float(g.rbf_enabled.mean())}
        l2 = g.L2.value_counts(normalize=True)
        r["top_script"] = L2_NAMES[int(l2.index[0])]
        r["top_script_share"] = float(l2.iloc[0])
        l1 = g.L1.value_counts(normalize=True)
        r["top_rule"] = L1_NAMES[int(l1.index[0])]
        r["top_rule_share"] = float(l1.iloc[0])
        l3 = g.L3.value_counts(normalize=True)
        r["runes_share"] = float(l3.get(1, 0.0))
        r["exchange_tag_share"] = float(((g.tags.to_numpy() & TAG_BITS["exchange"]) > 0).mean())
        if "eocj" in g:
            r["eocj_share"] = float(g.eocj.mean())
        if d2 is not None:
            r["representative_txid"] = str(g.txid.iloc[int(np.argmin(d2[m]))])
        r["descriptor"] = describe(r)
        rows.append(r)
    return pd.DataFrame(rows)


def _fmt_sat(v):
    if v >= 1e6:
        return f"{v / 1e8:.3f} BTC"
    return f"{v:,.0f} sat"


def describe(r):
    parts = [f"{r['med_inputs']:.0f}-in/{r['med_outputs']:.0f}-out",
             f"{r['top_script']} {r['top_script_share']:.0%}",
             f"value {_fmt_sat(r['med_value_sat'])}",
             f"{r['med_fee_rate']:.1f} sat/vB"]
    if r["opreturn_share"] >= 0.5:
        parts.append(f"OP_RETURN {r['opreturn_share']:.0%}")
    if r["top_rule"] not in ("Unlabelled",) and r["top_rule_share"] >= 0.5:
        parts.append(f"{r['top_rule']} {r['top_rule_share']:.0%}")
    return "; ".join(parts)
