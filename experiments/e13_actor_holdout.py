"""E13 (added after the freeze): evaluation on transactions of held-out actors.

Reviewers 1 and 2 asked for the leave-one-family-out and the Elliptic analysis to be
evaluated on "source-level held-out transactions". The Elliptic data alone carry no
actor information, so we add the actor (wallet-address) annotations of Elliptic++
[Elmougy et al. 2023], which cover the same 203,769 transactions.

Two questions are answered here.

1. Can the transactions be split so that no address appears on both sides? The
   address-transaction graph is examined for connected components, and a random
   split of addresses is scored by how many transactions keep all their addresses
   on one side.
2. With the actors known, the Elliptic evaluation of E8 is repeated with the
   ranking restricted to training-period transactions that share no address with
   any transaction of the evaluation period: the clusters are then ranked on
   actors that never appear in the evaluation set.

Inputs (not redistributed with this repository): `AddrTx_edgelist.csv` and
`TxAddr_edgelist.csv` from https://github.com/git-disl/EllipticPlusPlus, placed in
`<out_root>/external/`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from zsh.config import CFG, Log, out_dir, results_dir, seed_for, sha256_file, write_json  # noqa: E402
from zsh.io import load_elliptic, elliptic_train_mask  # noqa: E402
from zsh.metrics import concentration_many  # noqa: E402

LAST = CFG["elliptic"]["train_last_timestep"]
MODELS = {"ZSH": "elliptic_AF-165_ZSH.joblib",
          "K-means++ (K)": "elliptic_AF-165_K-meanspp.joblib",
          "rank-power K-means (K)": "elliptic_AF-165_rank-power.joblib",
          "uniform + refinement": "elliptic_AF-165_uniform.joblib"}


def load_edges(ext):
    a = pd.read_csv(ext / "AddrTx_edgelist.csv")
    a.columns = ["address", "txId"]
    t = pd.read_csv(ext / "TxAddr_edgelist.csv")
    t.columns = ["txId", "address"]
    return pd.concat([a[["txId", "address"]], t[["txId", "address"]]], ignore_index=True).drop_duplicates()


def component_report(e, log):
    tx_ids, tx_idx = np.unique(e.txId.to_numpy(), return_inverse=True)
    ad_ids, ad_idx = np.unique(e.address.to_numpy(), return_inverse=True)
    n_t, n_a = len(tx_ids), len(ad_ids)
    M = sparse.coo_matrix((np.ones(len(e)), (tx_idx, ad_idx + n_t)), shape=(n_t + n_a, n_t + n_a))
    ncomp, comp = connected_components(M + M.T, directed=False)
    sizes = np.bincount(comp[:n_t])
    log(f"  components {ncomp}, largest holds {sizes.max():,} of {n_t:,} transactions")
    return {"transactions_with_addresses": int(n_t), "addresses": int(n_a),
            "components": int(ncomp), "largest_component_transactions": int(sizes.max()),
            "largest_component_share": float(sizes.max() / n_t)}


def random_split_report(e, keep_tx, seeds=(0, 1, 2)):
    """How many transactions keep all their addresses on one side of a random address split?"""
    el = e[e.txId.isin(keep_tx)]
    tx_ids, tx_idx = np.unique(el.txId.to_numpy(), return_inverse=True)
    ad_ids, ad_idx = np.unique(el.address.to_numpy(), return_inverse=True)
    shares = []
    for s in seeds:
        rng = np.random.default_rng(s)
        side = rng.integers(0, 2, len(ad_ids))[ad_idx]
        g = pd.DataFrame({"tx": tx_idx, "side": side}).groupby("tx").side.agg(["min", "max"])
        shares.append(float((g["min"] == g["max"]).mean()))
    return {"transactions": int(len(tx_ids)), "addresses": int(len(ad_ids)),
            "addresses_per_transaction_median": float(np.median(np.bincount(tx_idx))),
            "clean_share_mean": float(np.mean(shares)), "clean_share_by_seed": shares}


def main():
    log = Log(out_dir("logs") / "e13_actor_holdout.log")
    res = results_dir("E13")
    ext = out_dir("external")
    mdir = out_dir("models")
    B = CFG["evaluation"]["bootstrap_B"]

    e = load_edges(ext)
    df = load_elliptic()
    tr = elliptic_train_mask(df)
    lab = df.label.to_numpy() >= 0
    log(f"edges {len(e):,}; Elliptic transactions {len(df):,}, labelled {int(lab.sum()):,}")

    summary = {"inputs": {f: sha256_file(ext / f) for f in ("AddrTx_edgelist.csv", "TxAddr_edgelist.csv")},
               "components": component_report(e, log)}

    lab_ids = set(df.txId[lab].to_numpy())
    summary["random_address_split"] = random_split_report(e, lab_ids)
    log(f"  random address split keeps {summary['random_address_split']['clean_share_mean'] * 100:.1f}% "
        f"of labelled transactions intact")

    # ranking set: training-period labelled transactions sharing no address with the evaluation period
    eval_ids = set(df.txId[(~tr) & lab].to_numpy())
    train_ids = set(df.txId[tr & lab].to_numpy())
    eval_addr = set(e.address[e.txId.isin(eval_ids)].to_numpy())
    tr_e = e[e.txId.isin(train_ids)]
    shares_addr = tr_e.assign(bad=tr_e.address.isin(eval_addr)).groupby("txId").bad.max()
    disjoint_train = set(shares_addr[~shares_addr].index)
    with_addr = set(e.txId.to_numpy())
    eval_ids = eval_ids & with_addr
    summary["holdout"] = {
        "labelled_training": len(train_ids), "training_with_addresses": int(shares_addr.shape[0]),
        "training_actor_disjoint": len(disjoint_train),
        "training_actor_disjoint_share": len(disjoint_train) / max(shares_addr.shape[0], 1),
        "evaluation_transactions": len(eval_ids)}
    log(f"  ranking set {len(disjoint_train):,} of {shares_addr.shape[0]:,} training transactions; "
        f"evaluation set {len(eval_ids):,}")

    keep = df.txId.isin(disjoint_train | eval_ids).to_numpy() & lab
    sub = df[keep].reset_index(drop=True)
    target = sub.label.to_numpy() == 1
    block = sub.timestep_eval.to_numpy()

    def fold_fn(ub):
        return (ub > LAST).astype(np.int8)

    rows = []
    with threadpool_limits(CFG["evaluation"]["threads"]):
        labs = {}
        for name, fn in MODELS.items():
            p = mdir / fn
            if not p.exists():
                log(f"  missing model {fn}; skipped")
                continue
            labs[name] = np.asarray(joblib.load(p).predict(sub))
        res_c, curves = concentration_many(labs, target, block, B, seed_for("E13", "conc"), reference="ZSH",
                                           fold_fn=fold_fn, directions=((0, 1),),
                                           min_members=CFG["elliptic"]["min_labelled_rank"])
    for m, r in res_c.items():
        row = {"method": m, "base_rate": r["base_rate"], "ap": r["ap"], "ap_lift": r["ap_lift"]}
        for q in ("0.10", "0.25", "0.50"):
            row[f"enrich@{q}"] = r[f"enrich@{q}"]
            row[f"prec@{q}"] = r[f"prec@{q}"]
            row[f"clusters@{q}"] = r[f"clusters@{q}"]
        for k2, (lo, hi) in r["ci"].items():
            row[f"{k2}_lo"], row[f"{k2}_hi"] = lo, hi
        for k2, v in r.get("vs_ZSH", {}).items():
            row[f"d_{k2}"], (row[f"d_{k2}_lo"], row[f"d_{k2}_hi"]), row[f"d_{k2}_p"] = v["diff"], v["ci"], v["p"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(res / "actor_holdout_concentration.csv", index=False)
    summary["evaluated_rows"] = int(len(sub))
    summary["illicit_in_ranking_set"] = int(((sub.label == 1) & (sub.timestep_eval <= LAST)).sum())
    summary["illicit_in_evaluation_set"] = int(((sub.label == 1) & (sub.timestep_eval > LAST)).sum())
    write_json(summary, res / "summary.json")
    log("  " + ", ".join(f"{r['method']} AP lift {r['ap_lift']:.2f}" for r in rows))
    log("done")


if __name__ == "__main__":
    main()
