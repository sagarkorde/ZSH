"""E17 (added after the freeze): the same evaluation applied to seven clustering families.

The confirmatory part of this study compares ZSH with K-means++ at the same K. The
findings that matter most -- that concentration is bounded by each base rate, that
better weights do not remove that bound, and that profiles do not survive a refit --
are then statements about ZSH. This script asks whether they are statements about the
task instead, by putting every clustering family of Section 6.1 through the whole
battery: concentration on the test period and on the prospective sample, stability
under block-bootstrap refits, temporal transfer, and profile matching across refits.

All methods use the same features, the same fitting data (the full development period,
except where a family cannot scale and is fitted on a subsample, as recorded in the
timing table) and the same number of clusters K* taken from the frozen primary model.

Stages (run separately; each writes its own artefacts):
  fit            fit every method on DEV, assign TEST / prospective / reference rows
  concentration  AP lift, ceiling and attained share per annotation, per method
  stability      block-bootstrap refits compared with the DEV partition
  transfer       refit on TEST and on the prospective sample; ARI and profile matching
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from e16_profile_matching import match  # noqa: E402
from zsh.baselines import BirchU, GMMU, MiniBatchU, VKVPartial, WardU  # noqa: E402
from zsh.cluster import ZSH, PlainKMeans  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets, structural_targets  # noqa: E402
from zsh.io import FUTURE_PATH, SMOKE, dev_test, load_future, selected_features  # noqa: E402
from zsh.metrics import agreement  # noqa: E402

# display name -> (slug, factory(features, K), model file already fitted on DEV or None)
METHODS = {
    "ZSH": ("zsh", lambda f, k: ZSH(f), "zsh_primary.joblib"),
    "K-means++ (K*)": ("kmu", lambda f, k: PlainKMeans(f, k, weighting="uniform", init="kmeans++"),
                       "e3_A9.joblib"),
    "MiniBatchKMeans": ("mbk", lambda f, k: MiniBatchU(f, k), None),
    "GMM (diag)": ("gmm", lambda f, k: GMMU(f, k), None),
    "BIRCH": ("birch", lambda f, k: BirchU(f, k), None),
    "Ward (sample + NC)": ("ward", lambda f, k: WardU(f, k), None),
    "VKV-partial (K*)": ("vkv", lambda f, k: VKVPartial(k, name="VKV-partial (K*)"), None),
}
# families that do not scale to the full development period (CF tree; trimmed k-means)
FIT_CAP = {"BIRCH": 1_000_000, "VKV-partial (K*)": 1_000_000}
# a fitted BIRCH tree cannot be pickled within the recursion limit; its labels are saved instead
NO_PICKLE = {"BIRCH"}
BENCH = "bench"


def bench_dir(*parts):
    return out_dir(BENCH, *parts)


def periods_available():
    p = ["test"]
    if SMOKE or FUTURE_PATH.exists():
        p.append("future")
    return p


def fit_one(name, feats, K, df, seed, log):
    """Fit one family on df, honouring its scaling limit; returns (model, seconds, rows used)."""
    cap = FIT_CAP.get(name)
    use = df
    if cap is not None and len(df) > cap:
        idx = np.sort(np.random.default_rng(seed_for("E17", "fitcap", name)).choice(len(df), cap, replace=False))
        use = df.iloc[idx].reset_index(drop=True)
    model = METHODS[name][1](feats, K)
    t = time.time()
    model.fit(use, seed)
    secs = time.time() - t
    log(f"  {name}: fitted on {len(use):,} rows in {secs:.0f}s")
    return model, secs, len(use)


def stage_fit(log, res):
    feats = selected_features()
    dev, test = dev_test()
    mdir = out_dir("models" + ("_smoke" if SMOKE else ""))
    K = joblib.load(mdir / "zsh_primary.joblib").k
    log(f"K* = {K}; DEV {len(dev):,}; TEST {len(test):,}")
    frames = {"test": test}
    if "future" in periods_available():
        frames["future"] = load_future()
        log(f"prospective {len(frames['future']):,}")
    ref_idx = np.sort(np.random.default_rng(seed_for("E17", "reference")).choice(
        len(dev), size=min(CFG["e4"]["reference_rows"], len(dev) // 2), replace=False))
    np.save(bench_dir("labels") / "reference_index.npy", ref_idx)
    ref = dev.iloc[ref_idx].reset_index(drop=True)

    rows = []
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for name, (slug, _, saved) in METHODS.items():
            if saved is not None:
                model = joblib.load(mdir / saved)
                secs, used = np.nan, len(dev)
                log(f"  {name}: loaded {saved}")
            else:
                model, secs, used = fit_one(name, feats, K, dev, seed_for("E17", "fit", name), log)
                if name not in NO_PICKLE:
                    joblib.dump(model, bench_dir("models") / f"dev_{slug}.joblib")
            row = {"method": name, "k": K, "fit_rows": used, "fit_seconds": secs}
            lab_ref = model.predict(ref)
            np.save(bench_dir("labels") / f"reference_{slug}.npy", lab_ref)
            row["ref_clusters_used"] = int(len(np.unique(lab_ref)))
            for pname, df in frames.items():
                t = time.time()
                lab = model.predict(df)
                np.save(bench_dir("labels") / f"{pname}_{slug}.npy", lab)
                w = df["design_weight"].to_numpy() if "design_weight" in df else np.ones(len(df))
                sh = np.bincount(lab, weights=w, minlength=K)
                row[f"{pname}_predict_seconds"] = time.time() - t
                row[f"{pname}_clusters_used"] = int(len(np.unique(lab)))
                row[f"{pname}_max_share"] = float(sh.max() / sh.sum())
                row[f"{pname}_top3_share"] = float(np.sort(sh)[-3:].sum() / sh.sum())
            rows.append(row)
            log("  " + name + ": " + ", ".join(
                p + " max share " + format(row[p + "_max_share"], ".3f") for p in frames))
            del model
    pd.DataFrame(rows).to_csv(res / "fit_and_assignment.csv", index=False)
    write_json({"K": K, "dev_rows": len(dev), "reference_rows": len(ref),
                "periods": {p: len(d) for p, d in frames.items()}, "fit_caps": FIT_CAP},
               res / "fit_summary.json")


def stage_concentration(log, res):
    _, test = dev_test()
    frames = {"test": test}
    if "future" in periods_available():
        frames["future"] = load_future()
    B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
    summary = {}
    for pname, df in frames.items():
        labs = {name: np.load(bench_dir("labels") / f"{pname}_{slug}.npy")
                for name, (slug, _, _) in METHODS.items()}
        log(f"{pname}: {len(df):,} rows, {len(labs)} methods")
        for kind, targets, tag in (("independent", independent_targets(df, include_eocj=True), "i"),
                                   ("structural", structural_targets(df), "s")):
            tab, curves, skipped = evaluate_targets(labs, df, targets, B, seed_for("E17", pname, tag),
                                                    reference="ZSH", log=log)
            if len(tab):
                tab["max_lift"] = 1.0 / tab["base_rate"]
                tab["attained"] = tab["ap"]
                tab["attained_pct"] = 100.0 * tab["ap"]
            tab.to_csv(res / f"{pname}_{kind}.csv", index=False)
            summary.setdefault(pname, {})[kind] = {
                "skipped": skipped, "targets": int(tab.target.nunique()) if len(tab) else 0}
    write_json(summary, res / "concentration_summary.json")


def stage_stability(log, res, n_boot):
    feats = selected_features()
    dev, _ = dev_test()
    mdir = out_dir("models" + ("_smoke" if SMOKE else ""))
    K = joblib.load(mdir / "zsh_primary.joblib").k
    ref_idx = np.load(bench_dir("labels") / "reference_index.npy")
    ref = dev.iloc[ref_idx].reset_index(drop=True)
    ref_labels = {name: np.load(bench_dir("labels") / f"reference_{slug}.npy")
                  for name, (slug, _, _) in METHODS.items()}
    rows_rep = min(CFG["e4"]["rows"], len(dev))

    blocks = dev.block_height.to_numpy()
    ub, binv = np.unique(blocks, return_inverse=True)
    order = np.argsort(binv, kind="stable")
    starts = np.searchsorted(binv[order], np.arange(len(ub)))
    ends = np.append(starts[1:], len(order))

    done = res / "stability_replicates.csv"
    rows = pd.read_csv(done).to_dict("records") if done.exists() else []
    seen = {(r["replicate"], r["method"]) for r in rows}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for i in range(n_boot):
            if all((i, name) in seen for name in METHODS):
                continue
            r = np.random.default_rng(seed_for("E17", "boot_rows", i))
            pick = r.integers(len(ub), size=len(ub))
            pool = np.concatenate([order[starts[b]:ends[b]] for b in pick])
            idx = np.sort(r.choice(pool, size=min(rows_rep, len(pool)), replace=False))
            fit_df = dev.iloc[idx].reset_index(drop=True)
            for name in METHODS:
                if (i, name) in seen:
                    continue
                model, secs, used = fit_one(name, feats, K, fit_df, seed_for("E17", "boot", name, i), log)
                lab = model.predict(ref)
                row = {"replicate": i, "method": name, "fit_seconds": secs, "fit_rows": used,
                       "k": int(getattr(model, "k", K))}
                row.update(agreement(ref_labels[name], lab))
                rows.append(row)
                log(f"bootstrap {i} {name}: ARI={row['ari']:.3f}")
                pd.DataFrame(rows).to_csv(done, index=False)
                del model
            del fit_df
    tab = pd.DataFrame(rows)
    agg = tab.groupby("method").agg(replicates=("ari", "size"), ari_mean=("ari", "mean"),
                                    ari_sd=("ari", "std"), ari_min=("ari", "min"),
                                    ami_mean=("ami", "mean"), vi_mean=("vi_bits", "mean"),
                                    fit_seconds_mean=("fit_seconds", "mean")).reset_index()
    agg.to_csv(res / "stability_by_method.csv", index=False)
    log(agg.to_string(index=False))


def stage_transfer(log, res):
    feats = selected_features()
    _, test = dev_test()
    mdir = out_dir("models" + ("_smoke" if SMOKE else ""))
    K = joblib.load(mdir / "zsh_primary.joblib").k
    frames = {"test": test}
    if "future" in periods_available():
        frames["future"] = load_future()
    rows, per_profile = [], []
    with threadpool_limits(CFG["evaluation"]["threads"]):
        for pname, df in frames.items():
            w = df["design_weight"].to_numpy() if "design_weight" in df else None
            for name, (slug, _, _) in METHODS.items():
                transferred = np.load(bench_dir("labels") / f"{pname}_{slug}.npy")
                model, secs, used = fit_one(name, feats, K, df, seed_for("E17", "refit", name, pname), log)
                refit = model.predict(df)
                if name not in NO_PICKLE:
                    joblib.dump(model, bench_dir("models") / f"refit_{pname}_{slug}.joblib")
                row = {"period": pname, "method": name, "refit_seconds": secs, "refit_rows": used,
                       "refit_k": int(getattr(model, "k", K))}
                row.update(agreement(transferred, refit))
                jac, ri, ci, size_o = match(transferred, refit, w)
                best = jac.max(1)
                matched = jac[ri, ci]
                tot = size_o.sum()
                row.update({"profiles": int(len(size_o)),
                            "median_matched_jaccard": float(np.median(matched)),
                            "max_matched_jaccard": float(matched.max()),
                            "n_matched_ge_0.5": int((matched >= 0.5).sum()),
                            "n_matched_ge_0.75": int((matched >= 0.75).sum()),
                            "share_matched_ge_0.5": float(size_o[ri][matched >= 0.5].sum() / tot),
                            "median_best_jaccard": float(np.median(best)),
                            "n_best_ge_0.5": int((best >= 0.5).sum())})
                rows.append(row)
                for a, b in zip(ri, ci):
                    per_profile.append({"period": pname, "method": name, "profile": int(a),
                                        "matched_to": int(b), "jaccard": float(jac[a, b]),
                                        "share": float(size_o[a] / tot)})
                log(f"{pname} {name}: ARI={row['ari']:.3f}, matched >= 0.5: "
                    f"{row['n_matched_ge_0.5']}/{row['profiles']}")
                del model
    pd.DataFrame(rows).to_csv(res / "transfer_by_method.csv", index=False)
    pd.DataFrame(per_profile).to_csv(res / "transfer_matching_long.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "concentration", "stability", "transfer", "all"])
    ap.add_argument("--bootstrap", type=int, default=10)
    a = ap.parse_args()
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e17_benchmark_{a.stage}{sfx}.log")
    res = results_dir("E17" + sfx)
    stages = ["fit", "concentration", "stability", "transfer"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"=== stage {s} ===")
        if s == "fit":
            stage_fit(log, res)
        elif s == "concentration":
            stage_concentration(log, res)
        elif s == "stability":
            stage_stability(log, res, 3 if SMOKE else a.bootstrap)
        else:
            stage_transfer(log, res)
    log("done")


if __name__ == "__main__":
    main()
