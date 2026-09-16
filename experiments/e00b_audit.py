"""E0b: data audit and pre-declared feature selection on DEV (ANALYSIS_PLAN §3.1, E0)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, rng_for, write_json  # noqa: E402
from zsh.data import L1_NAMES, L2_NAMES, L3_NAMES, TAG_BITS, l1_rule_masks  # noqa: E402
from zsh.features import select_features  # noqa: E402


def main():
    log = Log(out_dir("logs") / "e00b_audit.log")
    res = results_dir("E0")
    base = pd.read_parquet(out_dir("base") / "base.parquet")
    dev = base[base.split == 0].reset_index(drop=True)
    log(f"base {base.shape}, DEV {len(dev):,}")

    # coverage by quarter and split
    t = pd.to_datetime(base.block_time, unit="s", utc=True)
    cov = (base.assign(quarter=t.dt.tz_localize(None).dt.to_period("Q").astype(str))
           .groupby(["quarter", "split"]).agg(rows=("row_id", "size"),
                                              blocks=("block_height", "nunique"),
                                              first_height=("block_height", "min"),
                                              last_height=("block_height", "max")))
    cov.to_csv(res / "coverage_by_quarter.csv")
    split_tab = base.groupby("split").agg(rows=("row_id", "size"), blocks=("block_height", "nunique"),
                                          first_time=("block_time", "min"), last_time=("block_time", "max"))
    split_tab["first_time"] = pd.to_datetime(split_tab.first_time, unit="s", utc=True).astype(str)
    split_tab["last_time"] = pd.to_datetime(split_tab.last_time, unit="s", utc=True).astype(str)
    split_tab.to_csv(res / "splits.csv")
    log("\n" + split_tab.to_string())

    # feature selection (pre-declared rules)
    selected, report = select_features(dev, rng_for("E0", "spearman"), log)
    write_json(report, res / "feature_selection.json")
    pd.DataFrame(report["spearman"]).to_csv(res / "spearman_dev.csv")

    # unit check of the stored fee-rate column
    pf = pq.ParquetFile(CFG["paths"]["d1_parquet"])
    raw = pf.read_row_group(0, columns=["fee", "vsize", "fee_rate_sat_per_vbyte", "has_p2pk",
                                        "has_p2pkh", "has_p2sh", "has_p2wsh", "input_script_types",
                                        "input_count", "has_coinbase", "input_address_count",
                                        "input_addresses"]).to_pandas()
    ratio = (raw.fee * 1e8 / raw.vsize) / raw.fee_rate_sat_per_vbyte.replace(0, np.nan)
    types = raw.input_script_types.map(lambda l: set(";".join(l).split(";")) if len(l) else set())
    script_check = {
        "stored_fee_rate_times_1e8_over_recomputed_median": float(ratio.median()),
        "rows_checked": int(len(raw)),
        "has_p2pk_true_share": float(raw.has_p2pk.mean()),
        "exact_p2pk_type_share": float(types.map(lambda s: "pubkey" in s).mean()),
        "has_p2pk_true_and_p2pkh_type_share": float((raw.has_p2pk & types.map(lambda s: "pubkeyhash" in s)).mean()),
        "has_p2sh_true_and_only_p2wsh_share": float((raw.has_p2sh & types.map(
            lambda s: "witness_v0_scripthash" in s and "scripthash" not in s)).mean()),
        "input_address_list_len_values": raw.input_addresses.map(len).value_counts().to_dict(),
        "coinbase_input_count_values": raw.loc[raw.has_coinbase, "input_count"].value_counts().to_dict(),
    }
    write_json(script_check, res / "unit_and_flag_checks.json")
    log(f"unit/flag checks: {script_check}")

    # derivability of the count rules from the selected features
    rng = rng_for("E0", "derivability")
    idx = rng.permutation(len(dev))
    tr, te = idx[:500_000], idx[500_000:1_000_000]
    masks = l1_rule_masks(dev.input_count.to_numpy(), dev.output_count.to_numpy())
    deriv = {}
    for name, m in masks.items():
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(dev.loc[tr, selected], m[tr])
        pred = clf.predict(dev.loc[te, selected])
        deriv[name] = {"prevalence_dev": float(m.mean()),
                       "accuracy": float((pred == m[te]).mean()),
                       "balanced_accuracy": float(balanced_accuracy_score(m[te], pred))}
    write_json(deriv, res / "rule_derivability.json")
    log(f"derivability: {deriv}")

    # annotation prevalence per split
    rows = []
    for sp, g in base.groupby("split"):
        n = len(g)
        for code, name in enumerate(L1_NAMES):
            rows.append({"split": sp, "annotation": "L1", "label": name, "count": int((g.L1 == code).sum()), "share": float((g.L1 == code).mean())})
        for code, name in enumerate(L2_NAMES):
            rows.append({"split": sp, "annotation": "L2", "label": name, "count": int((g.L2 == code).sum()), "share": float((g.L2 == code).mean())})
        for code, name in enumerate(L3_NAMES):
            rows.append({"split": sp, "annotation": "L3", "label": name, "count": int((g.L3 == code).sum()), "share": float((g.L3 == code).mean())})
        for name, bit in TAG_BITS.items():
            hit = (g.tags.to_numpy() & bit) > 0
            rows.append({"split": sp, "annotation": "L4", "label": name, "count": int(hit.sum()), "share": float(hit.mean())})
        for name, m in l1_rule_masks(g.input_count.to_numpy(), g.output_count.to_numpy()).items():
            rows.append({"split": sp, "annotation": "L1-rule", "label": name, "count": int(m.sum()), "share": float(m.mean())})
        rows.append({"split": sp, "annotation": "all", "label": "rows", "count": n, "share": 1.0})
    pd.DataFrame(rows).to_csv(res / "annotation_prevalence.csv", index=False)

    # exact duplicate feature vectors in DEV (selected features)
    dup = {"dev_rows": len(dev),
           "distinct_vectors_selected": int(len(dev[selected].drop_duplicates())),
           "distinct_vectors_all_candidates": int(len(dev[CFG["features"]["priority"]].drop_duplicates()))}
    dup["distinct_share_selected"] = dup["distinct_vectors_selected"] / dup["dev_rows"]
    write_json(dup, res / "duplicates.json")
    log(f"duplicates: {dup}")

    # opret prefix distribution per split (top 15)
    op = (base[base.has_op_return > 0].groupby(["split", "opret_prefix"]).size()
          .rename("count").reset_index().sort_values(["split", "count"], ascending=[True, False])
          .groupby("split").head(15))
    op.to_csv(res / "opreturn_prefixes.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
