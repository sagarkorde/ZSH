"""E19 (added after the freeze): do address-level features add anything the twelve lack?

Section 6.8 finds that the twelve features carry much more about every annotation than
the profiles recover, so the clustering objective rather than the feature set is the
binding constraint. Exchange tags are the weakest case for the profiles, and exchanges
reuse addresses heavily, so address-level features are the natural test of whether a
richer description helps where the profiles do worst.

(The first version of this script compared supervised bounds cut into equal-size cells.
Equal-size cells cannot isolate a rare annotation, which understated every bound on the
rarer annotations; the bound now lets cell sizes vary, as in E18.)

The cached responses of the prospective collection hold the full transaction records,
so for those 456,292 transactions we can compute features the published sample does
not contain: how often each input and output address occurs elsewhere in the sample,
whether the transaction pays back to one of its own input addresses, the size and
number of witness items, sigops, locktime and sequence use, and the dispersion of the
input and output values. Address reuse is counted inside the sample only, which is a
page of up to 25 transactions from each of 800 blocks per month, so every count is a
lower bound on the reuse in the chain.

Four arms are scored on the prospective sample with the same cross-fitted protocol:
the frozen profiles, a refit on the twelve features, a refit on the twelve plus the
new ones, and the supervised bound of Section 6.8 computed both ways. Input script
types are deliberately not used as features, because they are the annotations of
Section 3.4.
"""
import gzip
import json
import sqlite3
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.cluster import ZSH  # noqa: E402
from zsh.config import CFG, Log, OUT, out_dir, results_dir, seed_for, write_json  # noqa: E402
from zsh.evaluate import evaluate_targets, independent_targets  # noqa: E402
from zsh.io import SMOKE, load_future, selected_features  # noqa: E402

CACHE = OUT / "future" / "d4_cache.sqlite"
NEW = ["in_reuse_max", "in_reuse_mean", "out_reuse_max", "out_reuse_mean", "self_transfer",
       "witness_items", "witness_bytes", "sigops", "locktime_set", "seq_nonfinal",
       "in_value_cv", "out_value_cv"]


def decompress(b):
    try:
        return gzip.decompress(b)
    except Exception:
        try:
            return zlib.decompress(b)
        except Exception:
            return b


def cv(values):
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or v.mean() <= 0:
        return 0.0
    return float(v.std() / v.mean())


def scan_cache(wanted, log):
    """Two passes over the cached pages: count address occurrences, then build features."""
    con = sqlite3.connect(str(CACHE))
    rows = con.execute("select body from resp where path like '%/txs/%'")
    seen, per_tx = Counter(), {}
    n_pages = 0
    for (body,) in rows:
        try:
            page = json.loads(decompress(body))
        except Exception:
            continue
        n_pages += 1
        for t in page:
            txid = t.get("txid")
            if txid is None or txid not in wanted or txid in per_tx:
                continue
            ins = [(vi.get("prevout") or {}).get("scriptpubkey_address") for vi in t.get("vin", [])]
            outs = [vo.get("scriptpubkey_address") for vo in t.get("vout", [])]
            ins = [a for a in ins if a]
            outs = [a for a in outs if a]
            seen.update(ins)
            seen.update(outs)
            per_tx[txid] = (ins, outs, t)
        if n_pages % 5000 == 0:
            log(f"  {n_pages:,} pages, {len(per_tx):,} transactions")
    con.close()
    log(f"  cache read: {n_pages:,} pages, {len(per_tx):,} of {len(wanted):,} sampled transactions")

    out = {}
    for txid, (ins, outs, t) in per_tx.items():
        ic = [seen[a] for a in ins] or [0]
        oc = [seen[a] for a in outs] or [0]
        wit_items = wit_bytes = 0
        seq_nonfinal = 0
        for vi in t.get("vin", []):
            w = vi.get("witness") or []
            wit_items += len(w)
            wit_bytes += sum(len(x) for x in w) // 2
            if vi.get("sequence", 0xFFFFFFFF) < 0xFFFFFFFE:
                seq_nonfinal += 1
        shared = len(set(ins) & set(outs))
        out[txid] = (
            float(max(ic)), float(np.mean(ic)), float(max(oc)), float(np.mean(oc)),
            float(shared / max(len(outs), 1)), float(wit_items), float(wit_bytes),
            float(t.get("sigops", 0)), float(t.get("locktime", 0) > 0),
            float(seq_nonfinal / max(len(t.get("vin", [])), 1)),
            cv([(vi.get("prevout") or {}).get("value", 0) for vi in t.get("vin", [])]),
            cv([vo.get("value", 0) for vo in t.get("vout", [])]),
        )
    return out


def free_cells(score, k, seed):
    """Cut a score into k cells of unconstrained size (K-means on the score).

    Equal-size cells cannot isolate a rare annotation, so the bound on what a partition
    of these features could reach must let the cell sizes vary (see E18).
    """
    lab = KMeans(k, n_init=5, random_state=seed).fit_predict(np.asarray(score).reshape(-1, 1))
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int32)


def supervised_bins(X, y, parity, k, seed):
    """Fit on one block-parity half, score the other, then cut the scores into k cells."""
    sc = np.empty(len(X), dtype=np.float64)
    for half in (0, 1):
        tr, ev = np.flatnonzero(parity == half), np.flatnonzero(parity == 1 - half)
        m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, early_stopping=True,
                                           validation_fraction=0.1, random_state=seed + half)
        m.fit(X[tr], y[tr])
        sc[ev] = m.predict_proba(X[ev])[:, 1]
    return free_cells(sc, k, seed)


def _design(df):
    """Design weights and month strata for the prospective sample; (None, None) elsewhere."""
    if "design_weight" not in df:
        return None, None
    return df["design_weight"].to_numpy(), df["month"].to_numpy()

def main():
    sfx = "_smoke" if SMOKE else ""
    log = Log(out_dir("logs") / f"e19_richer_features{sfx}.log")
    res = results_dir("E19" + sfx)
    mdir = out_dir("models" + sfx)
    feats = selected_features()
    fut = load_future()
    primary = joblib.load(mdir / "zsh_primary.joblib")
    K = primary.k
    log(f"prospective {len(fut):,} transactions, K* = {K}")

    t0 = time.time()
    table = scan_cache(set(fut.txid), log)
    cov = np.array([txid in table for txid in fut.txid])
    log(f"features built for {cov.sum():,} of {len(fut):,} rows ({time.time() - t0:.0f}s)")
    vals = np.zeros((len(fut), len(NEW)), dtype=np.float32)
    for i, txid in enumerate(fut.txid.to_numpy()):
        if txid in table:
            vals[i] = table[txid]
    for j, c in enumerate(NEW):
        fut[c] = vals[:, j]
    df = fut.loc[cov].reset_index(drop=True)
    log(f"evaluating on {len(df):,} rows with complete records")

    parity = (df["block_height"].to_numpy() % 2).astype(np.int8)
    X12 = df[feats].to_numpy(np.float32)
    Xall = df[feats + NEW].to_numpy(np.float32)
    targets = {k: v for k, v in independent_targets(df, include_eocj=True).items()
               if not k.startswith("L2:")}          # script classes would be read off the inputs
    log(f"targets: {list(targets)}")

    arms = {"ZSH (frozen, 12 features)": primary.predict(df)}
    with threadpool_limits(CFG["evaluation"]["threads"]):
        t = time.time()
        arms["ZSH refit (12 features)"] = ZSH(feats).fit(df, seed_for("E19", "refit12")).labels_
        log(f"  refit on 12 features ({time.time() - t:.0f}s)")
        t = time.time()
        arms["ZSH refit (12 + address features)"] = ZSH(feats + NEW).fit(df, seed_for("E19", "refitall")).labels_
        log(f"  refit on 12 + {len(NEW)} features ({time.time() - t:.0f}s)")

        parts = []
        B = 50 if SMOKE else CFG["evaluation"]["bootstrap_B"]
        for name, y in targets.items():
            if y.sum() < 400:
                log(f"  {name}: {int(y.sum())} positives, skipped")
                continue
            a = dict(arms)
            yy = y.astype(np.int8)
            a["supervised (12 features)"] = supervised_bins(X12, yy, parity, K, seed_for("E19", "s12", name))
            a["supervised (12 + address features)"] = supervised_bins(
                Xall, yy, parity, K, seed_for("E19", "sall", name))
            rw, st_ = _design(df)
            tab, _, _ = evaluate_targets(a, df, {name: y}, B, seed_for("E19", "boot", name),
                                         reference="ZSH (frozen, 12 features)", log=log,
                                         row_weights=rw, strata=st_)
            parts.append(tab)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out):
        out["max_lift"] = 1.0 / out["base_rate"]
        out["attained"] = out["ap"]
    out.to_csv(res / "richer_features.csv", index=False)
    write_json({"rows": int(len(df)), "coverage": float(cov.mean()), "new_features": NEW,
                "targets": list(targets), "K": K}, res / "summary.json")
    log("done")


if __name__ == "__main__":
    main()
