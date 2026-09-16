"""E1: fit the primary ZSH model on DEV, assign TEST, describe profiles, fit the atypicality score."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, seed_for, write_json  # noqa: E402
from zsh.evaluate import profile_table  # noqa: E402
from zsh.io import SMOKE, dev_test, selected_features  # noqa: E402


def main():
    tag = "smoke" if SMOKE else "full"
    log = Log(out_dir("logs") / f"e01_primary_{tag}.log")
    res = results_dir("E1" + ("_smoke" if SMOKE else ""))
    mdir = out_dir("models" + ("_smoke" if SMOKE else ""))
    ldir = out_dir("labels" + ("_smoke" if SMOKE else ""))
    feats = selected_features()
    dev, test = dev_test()
    log(f"DEV {len(dev):,}  TEST {len(test):,}  features {feats}")

    with threadpool_limits(CFG["evaluation"]["threads"]):
        t = time.time()
        model = ZSH(feats).fit(dev, seed_for("E1", "primary"), log=log)
        fit_s = time.time() - t
        log(f"fit: K={model.k}, max share {model.shares_.max():.3f}, {fit_s:.0f}s")
        t = time.time()
        lab_test, d2_test = model.predict(test, return_dist=True)
        pred_s = time.time() - t
        lab_dev, d2_dev = model.predict(dev, return_dist=True)
        assert np.array_equal(lab_dev, model.labels_)

        Xw = model.transform(dev)
        rng = rng_for("E1", "iforest_rows")
        idx = rng.choice(len(dev), size=min(CFG["atypicality"]["fit_rows"], len(dev)), replace=False)
        ac = CFG["atypicality"]
        iso = IsolationForest(n_estimators=ac["n_estimators"], max_samples=ac["max_samples"],
                              random_state=seed_for("E1", "iforest"), n_jobs=CFG["evaluation"]["threads"])
        iso.fit(Xw[idx])
        atyp_dev = -iso.score_samples(Xw)
        atyp_test = -iso.score_samples(model.transform(test))

    joblib.dump(model, mdir / "zsh_primary.joblib")
    joblib.dump(iso, mdir / "iforest_primary.joblib")
    np.save(ldir / "dev_zsh.npy", lab_dev)
    np.save(ldir / "test_zsh.npy", lab_test)
    np.save(ldir / "dev_zsh_d2.npy", d2_dev)
    np.save(ldir / "test_zsh_d2.npy", d2_test)
    np.save(ldir / "dev_atyp.npy", atyp_dev)
    np.save(ldir / "test_atyp.npy", atyp_test)

    summ = model.summary()
    summ.update({"fit_seconds": fit_s, "predict_test_seconds": pred_s,
                 "dev_rows": len(dev), "test_rows": len(test),
                 "test_shares": (np.bincount(lab_test, minlength=model.k) / len(test)).tolist(),
                 "dev_shares": model.shares_.tolist()})
    write_json(summ, res / "summary.json")
    pd.DataFrame({"feature": feats, "weight": model.w,
                  "rank": model.winfo.get("ranks"), "mi": model.winfo.get("scores")}
                 ).sort_values("rank").to_csv(res / "weights.csv", index=False)
    profile_table(dev, lab_dev, d2_dev, model.k).to_csv(res / "profiles_dev.csv", index=False)
    profile_table(test, lab_test, d2_test, model.k).to_csv(res / "profiles_test.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
