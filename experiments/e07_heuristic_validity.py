"""E7: validity of the count rule 'in>3 & out>3' (v1 'coinjoin-like') against the
equal-output CoinJoin heuristic (EO-CJ) and GraphSense coinjoin tags."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, rng_for, write_json  # noqa: E402
from zsh.data import TAG_BITS, l1_rule_masks  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, load_base, load_future  # noqa: E402
from zsh.metrics import wilson_interval  # noqa: E402


def stratified_estimates(p_f, n_f, p_u, n_u, N_f, N_u, rng, B=10000):
    """Population prevalence of EO-CJ and recall/precision of the count rule, with
    parametric-bootstrap intervals from the two stratum binomials."""
    def est(pf, pu):
        cj_f, cj_u = N_f * pf, N_u * pu
        return {"precision_rule": pf, "recall_rule": cj_f / (cj_f + cj_u) if cj_f + cj_u > 0 else np.nan,
                "prevalence": (cj_f + cj_u) / (N_f + N_u)}
    point = est(p_f, p_u)
    sf = rng.binomial(n_f, p_f, B) / n_f
    su = rng.binomial(n_u, p_u, B) / n_u
    boots = [est(a, b) for a, b in zip(sf, su)]
    out = {}
    for k, v in point.items():
        vals = np.array([b[k] for b in boots], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[k] = {"estimate": v, "ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}
    return out


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e07_heuristic{sfx}.log")
    res = results_dir("E7" + sfx)
    rng = rng_for("E7", "bootstrap")
    summary = {}

    d5_rows = out_dir("d5") / "d5_sample_d1rows.parquet"
    d5_api = out_dir("d5") / "d5_api_records.parquet"
    if not SMOKE and d5_api.exists():
        base = load_base(["split", "input_count", "output_count"])
        many = l1_rule_masks(base.input_count.to_numpy(), base.output_count.to_numpy())["ManyInManyOut"]
        N_f, N_u = int(many.sum()), int((~many).sum())
        s = pd.read_parquet(d5_rows)[["txid", "sample", "split"]]
        a = pd.read_parquet(d5_api).drop_duplicates("txid")
        m = s.merge(a, on="txid")
        m = m[m["sample"].isin(["e7_flagged", "e7_unflagged"])]
        rows = []
        for (smp, sp), g in m.groupby(["sample", "split"]):
            k, n = int(g.eocj.sum()), len(g)
            lo, hi = wilson_interval(k, n)
            rows.append({"sample": smp, "split": int(sp), "n": n, "eocj": k, "eocj_rate": k / n,
                         "ci_lo": lo, "ci_hi": hi,
                         "coinjoin_tag": int(((g.tags & TAG_BITS["coinjoin"]) > 0).sum()),
                         "exchange_tag": int(((g.tags & TAG_BITS["exchange"]) > 0).sum())})
        pd.DataFrame(rows).to_csv(res / "d1_strata.csv", index=False)
        gf = m[m["sample"] == "e7_flagged"]
        gu = m[m["sample"] == "e7_unflagged"]
        summary["D1"] = {
            "population_flagged": N_f, "population_unflagged": N_u,
            "n_flagged": len(gf), "n_unflagged": len(gu),
            "estimates": stratified_estimates(gf.eocj.mean(), len(gf), gu.eocj.mean(), len(gu), N_f, N_u, rng),
            "eocj_top_values_sat": gf[gf.eocj == 1].eocj_value.value_counts().head(10).to_dict(),
            "eocj_k_distribution": gf[gf.eocj == 1].eocj_k.describe().to_dict(),
            "coinjoin_tag_matches": int(((m.tags & TAG_BITS["coinjoin"]) > 0).sum()),
        }
        log(f"D1: {summary['D1']['estimates']}")

    if SMOKE or FUTURE_PATH.exists():
        fut = load_future()
        many = l1_rule_masks(fut.input_count.to_numpy(), fut.output_count.to_numpy())["ManyInManyOut"]
        cj = fut.eocj.to_numpy() == 1
        w = fut["design_weight"].to_numpy()
        tags = fut.tags.to_numpy()
        def rates(wt):
            tp = np.sum(wt * (many & cj))
            return {"precision_rule": float(tp / np.sum(wt * many)) if many.any() else np.nan,
                    "recall_rule": float(tp / np.sum(wt * cj)) if cj.any() else np.nan,
                    "prevalence_eocj": float(np.sum(wt * cj) / np.sum(wt)),
                    "prevalence_rule": float(np.sum(wt * many) / np.sum(wt))}
        # percentile intervals: blocks resampled with replacement within each month (the sampling
        # design: months are strata, blocks the sampled units); weights are constant within a block
        blk = pd.DataFrame({"month": fut.month.to_numpy(), "block": fut.block_height.to_numpy(),
                            "tp": w * (many & cj), "rule": w * many, "cj": w * cj, "w": w})
        blk = blk.groupby(["month", "block"], sort=True)[["tp", "rule", "cj", "w"]].sum().reset_index()
        B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
        brng = rng_for("E7", "future_bootstrap")
        groups = [g[["tp", "rule", "cj", "w"]].to_numpy() for _, g in blk.groupby("month")]
        boots = []
        for _ in range(B):
            tp_, ru, c_, ww = sum(g[brng.integers(len(g), size=len(g))].sum(axis=0) for g in groups)
            boots.append({"precision_rule": tp_ / ru if ru > 0 else np.nan,
                          "recall_rule": tp_ / c_ if c_ > 0 else np.nan,
                          "prevalence_eocj": c_ / ww, "prevalence_rule": ru / ww})
        weighted_ci = {}
        for key in boots[0]:
            v = np.array([b[key] for b in boots], dtype=float)
            v = v[np.isfinite(v)]
            weighted_ci[key] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None
        summary["FUTURE"] = {
            "rows": len(fut), "unweighted": rates(np.ones(len(fut))), "design_weighted": rates(w),
            "design_weighted_ci": weighted_ci, "bootstrap_B": B,
            "confusion": {"rule&eocj": int((many & cj).sum()), "rule&~eocj": int((many & ~cj).sum()),
                          "~rule&eocj": int((~many & cj).sum()), "~rule&~eocj": int((~many & ~cj).sum())},
            "coinjoin_tag_matches": int(((tags & TAG_BITS["coinjoin"]) > 0).sum()),
            "coinjoin_tag_and_eocj": int((((tags & TAG_BITS["coinjoin"]) > 0) & cj).sum()),
            "eocj_top_values_sat": (fut.loc[cj, "eocj_value"].value_counts().head(10).to_dict()
                                    if "eocj_value" in fut else {}),
        }
        def month_rates(g):
            wt = g.w.to_numpy()
            r_, c2 = g.rule.to_numpy(), g.cj.to_numpy()
            return pd.Series({"rows": len(g), "rule": np.average(r_, weights=wt), "eocj": np.average(c2, weights=wt),
                              "eocj_rows": int(c2.sum()),
                              "precision_rule": np.average(c2[r_], weights=wt[r_]) if r_.any() else np.nan})
        by_month = (fut.assign(rule=many, cj=cj, w=w).groupby("month")
                    .apply(month_rates, include_groups=False))
        by_month.to_csv(res / "future_by_month.csv")
        log(f"FUTURE: {summary['FUTURE']['unweighted']}")
    write_json(summary, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
