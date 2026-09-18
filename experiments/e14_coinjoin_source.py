"""E14 (added after the freeze): the count rule and the equal-output rule against an
external list of CoinJoin transactions.

Sections 6.4 and 6.5 judge the published "coinjoin-like" count rule against our own
equal-output rule, which is itself a rule. This script adds a source-level check: a
public list of Wasabi 2.x CoinJoin transactions, each detected from the coordinator
that created it (https://github.com/crocs-muni/coinjoin-analysis, file
`data/wasabi2/txid_coord.json`, placed in `<out_root>/external/`). The list is used
only as a label; it is not redistributed with this repository.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.config import Log, out_dir, results_dir, sha256_file, write_json  # noqa: E402
from zsh.data import l1_rule_masks  # noqa: E402
from zsh.io import FUTURE_PATH, load_base, load_future  # noqa: E402
from zsh.metrics import wilson_interval  # noqa: E402

SRC = "wasabi2_txid_coord.json"


def load_list(ext):
    raw = json.load(open(ext / SRC, encoding="utf-8"))
    out = {}
    for part in raw.values():
        if isinstance(part, dict):
            out.update(part)
    return out


def main():
    log = Log(out_dir("logs") / "e14_coinjoin_source.log")
    res = results_dir("E14")
    ext = out_dir("external")
    coord = load_list(ext)
    ids = set(coord)
    log(f"external list: {len(ids):,} CoinJoin transactions, "
        f"{len(set(coord.values()))} coordinators")

    summary = {"source": {"file": SRC, "sha256": sha256_file(ext / SRC), "transactions": len(ids),
                          "coordinators": sorted(set(coord.values()))}}
    rows = []
    base = load_base(["txid", "split", "input_count", "output_count", "block_time"])
    for split, name in ((0, "Development"), (1, "Test")):
        d = base[base.split == split]
        m = d.txid.isin(ids).to_numpy()
        rule = l1_rule_masks(d.input_count.to_numpy(), d.output_count.to_numpy())["ManyInManyOut"]
        n = int(m.sum())
        lo, hi = wilson_interval(int(rule[m].sum()), n) if n else (np.nan, np.nan)
        rows.append({"period": name, "transactions": len(d), "known_coinjoins": n,
                     "count_rule_recall": float(rule[m].mean()) if n else np.nan,
                     "count_rule_recall_lo": lo, "count_rule_recall_hi": hi,
                     "equal_output_recall": np.nan, "known_among_rule_flagged": int((rule & m).sum()),
                     "rule_flagged": int(rule.sum())})
        log(f"{name}: {n} known CoinJoins, count rule recall "
            f"{100 * rule[m].mean():.1f}%" if n else f"{name}: none")

    if FUTURE_PATH.exists():
        f = load_future()
        m = f.txid.isin(ids).to_numpy()
        rule = l1_rule_masks(f.input_count.to_numpy(), f.output_count.to_numpy())["ManyInManyOut"]
        eo = f.eocj.to_numpy() == 1
        n = int(m.sum())
        lo, hi = wilson_interval(int(rule[m].sum()), n)
        elo, ehi = wilson_interval(int(eo[m].sum()), n)
        rows.append({"period": "Prospective", "transactions": len(f), "known_coinjoins": n,
                     "count_rule_recall": float(rule[m].mean()), "count_rule_recall_lo": lo,
                     "count_rule_recall_hi": hi, "equal_output_recall": float(eo[m].mean()),
                     "equal_output_recall_lo": elo, "equal_output_recall_hi": ehi,
                     "known_among_rule_flagged": int((rule & m).sum()), "rule_flagged": int(rule.sum()),
                     "known_among_equal_output": int((eo & m).sum()), "equal_output_flagged": int(eo.sum())})
        log(f"Prospective: {n} known CoinJoins; count rule recall {100 * rule[m].mean():.1f}%, "
            f"equal-output recall {100 * eo[m].mean():.1f}%; "
            f"{int((eo & m).sum())} of {int(eo.sum())} equal-output transactions are on the list")
        summary["prospective_coordinators"] = (pd.Series([coord[t] for t in f.txid[m]])
                                               .value_counts().to_dict())

    # where do the known CoinJoins fall among the profiles?
    ldir = out_dir("labels")
    prof = {}
    test = base[base.split == 1]
    lab = np.load(ldir / "test_zsh.npy")
    mt = test.txid.isin(ids).to_numpy()
    if mt.sum():
        cnt = pd.Series(lab[mt]).value_counts()
        prof["Test"] = {"known": int(mt.sum()), "profiles_used": int(cnt.size),
                        "largest_profile": int(cnt.index[0]), "largest_share": float(cnt.iloc[0] / mt.sum()),
                        "profiles_for_80pct": int((cnt.cumsum() / mt.sum() < 0.8).sum() + 1)}
        log(f"Test: known CoinJoins fall in {cnt.size} profiles; "
            f"{100 * cnt.iloc[0] / mt.sum():.0f}% in profile P{cnt.index[0]:02d}")
    if FUTURE_PATH.exists():
        fl = np.load(ldir / "future_transfer_zsh.npy")
        mf = f.txid.isin(ids).to_numpy()
        if mf.sum():
            cnt = pd.Series(fl[mf]).value_counts()
            prof["Prospective"] = {"known": int(mf.sum()), "profiles_used": int(cnt.size),
                                   "largest_profile": int(cnt.index[0]),
                                   "largest_share": float(cnt.iloc[0] / mf.sum()),
                                   "profiles_for_80pct": int((cnt.cumsum() / mf.sum() < 0.8).sum() + 1)}
    summary["profiles"] = prof
    pd.DataFrame(rows).to_csv(res / "coinjoin_source.csv", index=False)
    write_json(summary, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
