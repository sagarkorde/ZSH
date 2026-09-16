"""E0c: fetch per-transaction API data for D1 samples (D5) and check feature equivalence.

Samples (seeded, DEV and TEST rows only):
  * equivalence: 2,000 random transactions
  * E7: 5,000 transactions matching the in>3 & out>3 count rule and 5,000 not matching
Only the equivalence report is computed here; E7 is analysed after the freeze.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.collect import Esplora  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, sha256_file, write_json  # noqa: E402
from zsh.data import base_from_esplora, l1_rule_masks, load_tagmap  # noqa: E402

COMPARE_EXACT = ["input_count", "output_count", "vsize", "size", "has_op_return", "rbf_enabled",
                 "L1", "L2", "L3", "tags"]
COMPARE_VALUE = ["total_input_value", "fee", "avg_output_value", "input_output_ratio",
                 "fee_rate_sat_per_vbyte", "fee_rate_sat_per_byte"]


def main():
    log = Log(out_dir("logs") / "e00c_fetch_d5.log")
    base = pd.read_parquet(out_dir("base") / "base.parquet")
    base = base[base.split.isin([0, 1])].reset_index(drop=True)
    rng = rng_for("E0", "api_sample")
    eq_idx = rng.choice(len(base), size=CFG["e0"]["api_rows"], replace=False)
    many = l1_rule_masks(base.input_count.to_numpy(), base.output_count.to_numpy())["ManyInManyOut"]
    rng7 = rng_for("E7", "sample")
    fl_idx = rng7.choice(np.flatnonzero(many), size=CFG["e7"]["flagged"], replace=False)
    un_idx = rng7.choice(np.flatnonzero(~many), size=CFG["e7"]["unflagged"], replace=False)
    sample = pd.concat([
        base.iloc[eq_idx].assign(sample="equivalence"),
        base.iloc[fl_idx].assign(sample="e7_flagged"),
        base.iloc[un_idx].assign(sample="e7_unflagged"),
    ], ignore_index=True)
    sample.to_parquet(out_dir("d5") / "d5_sample_d1rows.parquet", index=False)
    log(f"sample sizes: {sample['sample'].value_counts().to_dict()}")

    api = Esplora(out_dir("d5") / "d5_cache.sqlite", log)
    failed = api.get_many([f"/tx/{t}" for t in sample.txid], progress_every=500)
    tagmap = load_tagmap()
    recs = []
    for t in sample.txid:
        j = api.json(f"/tx/{t}")
        if j is None:
            continue
        r = base_from_esplora(j, tagmap)
        r["n_outputs_ge_min"] = sum(1 for o in j["vout"] if o.get("value", 0) >= CFG["annotations"]["eocj_min_value_sat"])
        recs.append(r)
    api_df = pd.DataFrame(recs)
    api_df.to_parquet(out_dir("d5") / "d5_api_records.parquet", index=False)
    log(f"parsed {len(api_df):,} API records, {len(failed)} failed")

    eq = sample[sample["sample"] == "equivalence"].merge(api_df, on="txid", suffixes=("_d1", "_api"))
    report = {"requested": int((sample["sample"] == "equivalence").sum()), "matched": len(eq), "features": {}}
    for f in COMPARE_EXACT:
        a, b = eq[f + "_d1"].to_numpy(np.float64), eq[f + "_api"].to_numpy(np.float64)
        report["features"][f] = {"agreement": float(np.mean(a == b)), "rule": "exact"}
    for f in COMPARE_VALUE:
        a, b = eq[f + "_d1"].to_numpy(np.float64), eq[f + "_api"].to_numpy(np.float64)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1.0)
        report["features"][f] = {"agreement": float(np.mean(rel <= 1e-6)), "rule": "rel_err<=1e-6",
                                 "max_rel_err": float(rel.max()), "p99_rel_err": float(np.quantile(rel, 0.99))}
    for f, v in report["features"].items():
        v["usable_on_prospective"] = v["agreement"] >= 0.99
    # show a few disagreements for diagnosis
    diag = {}
    for f in COMPARE_EXACT:
        bad = eq[eq[f + "_d1"].to_numpy(np.float64) != eq[f + "_api"].to_numpy(np.float64)]
        if len(bad):
            diag[f] = bad[["txid", f + "_d1", f + "_api"]].head(5).astype(str).to_dict("records")
    report["examples_of_disagreement"] = diag
    report["cache_sha256"] = sha256_file(out_dir("d5") / "d5_cache.sqlite")
    write_json(report, results_dir("E0") / "api_equivalence.json")
    log(json.dumps(report["features"], indent=1))


if __name__ == "__main__":
    main()
