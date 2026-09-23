"""E6: concentration of annotations that are not clustering inputs (RQ3), TEST and prospective D4.

Methods: A1 ZSH (primary), A9 uniform K-means++ (K*), A3 rank-power K-means (K*, no
refinement), A2 uniform + refinement. Reference for paired differences: A1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets, profile_table, structural_targets  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, load_future  # noqa: E402

ARMS = {"A1": "ZSH", "A9": "K-means++ (K*)", "A3": "rank-power K-means (K*)", "A2": "uniform + refinement"}


def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e06_concentration{sfx}.log")
    res = results_dir("E6" + sfx)
    mdir = out_dir("models" + sfx)
    ldir = out_dir("labels" + sfx)
    _, test = dev_test()
    models = {a: (joblib.load(mdir / "zsh_primary.joblib") if a == "A1" else joblib.load(mdir / f"e3_{a}.joblib"))
              for a in ARMS}
    periods = {"TEST": (test, {ARMS[a]: np.load(ldir / f"test_e3_{a}.npy") for a in ARMS})}
    if SMOKE or FUTURE_PATH.exists():
        fut = load_future()
        periods["FUTURE"] = (fut, {ARMS[a]: m.predict(fut) for a, m in models.items()})
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    summary = {}
    for pname, (df, labs) in periods.items():
        log(f"{pname}: {len(df):,} rows")
        has_w = "design_weight" in df
        w = df["design_weight"].to_numpy() if has_w else None
        # Variants. The prospective sample was drawn with unequal inclusion
        # probabilities, so its primary estimates are design-weighted with a
        # month-stratified block bootstrap (R1.2); the unweighted run is kept as a
        # sample-specific sensitivity analysis and uses the original seed and the
        # unstratified bootstrap, so it reproduces the previously reported table.
        if has_w:
            months = df["month"].to_numpy()
            variants = [("", w, months, seed_for("E6", pname, "w")),
                        ("_unweighted", None, None, seed_for("E6", pname))]
        else:
            variants = [("", None, None, seed_for("E6", pname))]
        for tag, rw, st_, sd in variants:
            how = ("design-weighted, month-stratified bootstrap" if rw is not None
                   else "unweighted, block bootstrap")
            log(f"  {pname} [{how}]")
            tab, curves, skipped = evaluate_targets(labs, df, independent_targets(df, include_eocj=True), B,
                                                    sd, reference="ZSH", log=log,
                                                    row_weights=rw, strata=st_)
            # above-base-rate check for ZSH: AP lift lower CI bound > 1
            tab["ap_lift_gt1"] = tab["ap_lift_lo"] > 1
            tab.to_csv(res / f"{pname.lower()}_independent{tag}.csv", index=False)
            stab, _, _ = evaluate_targets(labs, df, structural_targets(df), B,
                                          sd + 10_000, reference="ZSH", log=log,
                                          row_weights=rw, strata=st_)
            stab.to_csv(res / f"{pname.lower()}_structural{tag}.csv", index=False)
            summary[pname + tag] = {"rows": len(df), "skipped": skipped, "curves": curves,
                                    "design_weighted": rw is not None,
                                    "bootstrap": "month-stratified block" if st_ is not None else "block"}
        _, d2 = models["A1"].predict(df, return_dist=True)
        profile_table(df, labs["ZSH"], d2, models["A1"].k, w).to_csv(res / f"profiles_{pname.lower()}.csv", index=False)
        if has_w:
            profile_table(df, labs["ZSH"], d2, models["A1"].k, None).to_csv(
                res / f"profiles_{pname.lower()}_unweighted.csv", index=False)
    write_json(summary, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
