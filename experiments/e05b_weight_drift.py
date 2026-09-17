"""E5b (exploratory, added after the freeze): feature ranks and weights of the
refitted models in E5 compared with the development model. Reads saved models only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from zsh.config import out_dir, results_dir, write_json  # noqa: E402
from zsh.io import SMOKE  # noqa: E402


def main():
    sfx = "_smoke" if SMOKE else ""
    mdir = out_dir("models" + sfx)
    res = results_dir("E5" + sfx)
    dev = joblib.load(mdir / "zsh_primary.joblib")
    rows, summ = [], {}
    for period in ("test", "future"):
        p = mdir / f"e5_refit_{period}.joblib"
        if not p.exists():
            continue
        m = joblib.load(p)
        tau = kendalltau(dev.winfo["ranks"], m.winfo["ranks"])
        summ[period.upper()] = {"kendall_tau": float(tau.statistic), "p_value": float(tau.pvalue),
                                "refit_k": m.k, "refit_max_share": float(m.shares_.max()),
                                "refit_splits": len(m.splits)}
        for f, wd, wr, rd, rr in zip(dev.features, dev.w, m.w, dev.winfo["ranks"], m.winfo["ranks"]):
            rows.append({"period": period.upper(), "feature": f, "rank_dev": int(rd), "weight_dev": float(wd),
                         "rank_refit": int(rr), "weight_refit": float(wr)})
    pd.DataFrame(rows).to_csv(res / "weight_drift.csv", index=False)
    write_json(summ, res / "weight_drift.json")
    print(summ)


if __name__ == "__main__":
    main()
