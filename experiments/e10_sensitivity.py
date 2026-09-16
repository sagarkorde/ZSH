"""E10: sensitivity and ablations on a fixed 1,000,000-row DEV sample, evaluated on TEST.

Variants: s, K0, cap/depth, v1-style upsampled corpus, sample-weighted fit, initialisation
(k-means++, semantic seeds, seed-Ward blend) and leave-one-family-out for the seeded variant.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, seed_for, write_json  # noqa: E402
from zsh.data import L1_NAMES, l1_rule_masks  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets, structural_targets  # noqa: E402
from zsh.features import Preprocessor  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402
from zsh.metrics import intrinsic  # noqa: E402

FAMILY_FEATURE = {0: "has_coinbase", 6: "has_op_return", 7: "rbf_enabled"}


def strata(df):
    m = l1_rule_masks(df.input_count.to_numpy(), df.output_count.to_numpy())
    return (m["ManyInManyOut"].astype(int) * 4 + m["SingleInFanOut"].astype(int) * 2
            + (df.has_op_return.to_numpy() > 0).astype(int))


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e10_sensitivity{sfx}.log")
    res = results_dir("E10" + sfx)
    ec = CFG["e10"]
    feats = selected_features()
    dev, test = dev_test()
    rng = rng_for("E10", "rows")
    rows_n = min(ec["rows"], len(dev))
    sample = dev.iloc[np.sort(rng.choice(len(dev), size=rows_n, replace=False))].reset_index(drop=True)
    seed = seed_for("E10", "fit")
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]

    # v1-style upsampling of the working sample, and the equivalent sample weights
    st = strata(sample)
    sizes = np.bincount(st, minlength=8)
    target_size = int(ec["upsample_target"] * sizes.max())
    parts = []
    for s_ in np.unique(st):
        idx = np.flatnonzero(st == s_)
        if len(idx) < target_size:
            idx = rng.choice(idx, size=target_size, replace=True)
        parts.append(idx)
    up_idx = np.concatenate(parts)
    upsampled = sample.iloc[up_idx].reset_index(drop=True)
    sw = np.maximum(1.0, target_size / sizes[st])

    variants = [("reference", {}, None, None)]
    variants += [(f"s={s}", {"s": s}, None, None) for s in ec["s_values"]]
    variants += [(f"K0={k}", {"k0": k}, None, None) for k in ec["k0_values"]]
    variants += [(f"cap={c}", ({"cap": c} if c is not None else {"refine": False}), None, None)
                 for c in ec["cap_values"]]
    variants += [(f"depth={ec['deep_depth']}", {"depth": ec["deep_depth"]}, None, None)]
    variants += [("upsampled (v1 style)", {}, upsampled, None), ("sample-weighted", {}, None, sw)]
    variants += [("init=k-means++", {"init": "kmeans++"}, None, None),
                 ("init=semantic seeds", {"init": "seeds"}, None, None),
                 ("init=seed-Ward blend", {"init": "blend"}, None, None)]

    geo_idx = rng.choice(len(dev), size=min(CFG["e2"]["eval_rows"], len(dev)), replace=False)
    common = Preprocessor(feats).fit(dev)
    Xg = common.transform(dev.iloc[geo_idx])
    rows, labels = [], {}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for name, kw, data, weights in variants:
            t = time.time()
            m = ZSH(feats, name=name, **kw).fit(sample if data is None else data, seed, sample_weight=weights)
            labels[name] = m.predict(test)
            g = intrinsic(Xg, m.predict(dev.iloc[geo_idx]), seed_for("E10", "geo", name))
            rows.append({"variant": name, "k": m.k, "max_share_fit": float(m.shares_.max()),
                         "max_share_test": float(np.bincount(labels[name]).max() / len(test)),
                         "fit_seconds": time.time() - t,
                         **{f"common_{k}": v for k, v in g.items() if k in ("silhouette", "dbi", "chi")},
                         "weights": ";".join(f"{f}={w:.4f}" for f, w in zip(feats, m.w))})
            log(f"{name}: K={m.k} ({time.time() - t:.0f}s)")

        # leave-one-family-out for the seeded variant
        loo_rows = []
        for fam in range(8):
            feats_f = [f for f in feats if f != FAMILY_FEATURE.get(fam)]
            tgt = {f"L1:{L1_NAMES[fam]}": test.L1.to_numpy() == fam}
            arms = {}
            if feats_f == feats:
                arms["unseeded ZSH"] = labels["reference"]
                arms["seeded, all families"] = labels["init=semantic seeds"]
            else:
                arms["unseeded ZSH"] = ZSH(feats_f).fit(sample, seed).predict(test)
                arms["seeded, all families"] = ZSH(feats_f, init="seeds").fit(sample, seed).predict(test)
            arms["seeded, family withheld"] = ZSH(feats_f, init="seeds", exclude_family=fam).fit(sample, seed).predict(test)
            tab, _, skipped = evaluate_targets(arms, test, tgt, B, seed_for("E10", "loo", fam), "unseeded ZSH")
            tab["family"] = L1_NAMES[fam]
            tab["feature_removed"] = FAMILY_FEATURE.get(fam, "")
            loo_rows.append(tab)
            log(f"LOO {L1_NAMES[fam]} done (skipped: {skipped})")

    pd.DataFrame(rows).to_csv(res / "variants_geometry.csv", index=False)
    pd.concat(loo_rows, ignore_index=True).to_csv(res / "loo_seeded.csv", index=False)
    ind, _, skipped = evaluate_targets(labels, test, independent_targets(test), B, seed_for("E10", "ind"),
                                       "reference", log)
    ind.to_csv(res / "variants_independent.csv", index=False)
    stt, _, _ = evaluate_targets(labels, test, structural_targets(test), B, seed_for("E10", "st"), "reference", log)
    stt.to_csv(res / "variants_structural.csv", index=False)
    write_json({"rows": rows_n, "upsampled_rows": len(upsampled), "strata_sizes": sizes.tolist(),
                "upsample_target_rows": target_size, "skipped": skipped}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
