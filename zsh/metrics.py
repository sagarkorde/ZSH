"""Evaluation: intrinsic geometry, partition agreement, cross-fitted annotation concentration."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             calinski_harabasz_score, davies_bouldin_score, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix

from .config import CFG


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def wilson_lower(pos, n, z=1.959964):
    pos = np.asarray(pos, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, pos / n, 0.0)
        denom = 1 + z ** 2 / n
        centre = p + z ** 2 / (2 * n)
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        lb = (centre - half) / denom
    return np.where(n > 0, lb, 0.0)


def wilson_interval(pos, n, z=1.959964):
    p = pos / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def holm(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def ci(samples, level=0.95):
    a = (1 - level) / 2
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return [np.nan, np.nan]
    return [float(np.quantile(s, a)), float(np.quantile(s, 1 - a))]


def boot_p_two_sided(diffs):
    """Add-one bootstrap p-value for H0: difference = 0."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    B = len(d)
    lo = (1 + np.sum(d <= 0)) / (B + 1)
    hi = (1 + np.sum(d >= 0)) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


# ---------------------------------------------------------------------------
# intrinsic geometry
# ---------------------------------------------------------------------------
def intrinsic(X, labels, seed, noise_label=None):
    ev = CFG["evaluation"]
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    labels = np.asarray(labels)
    if noise_label is not None:
        keep = labels != noise_label
        X, labels = X[keep], labels[keep]
    out = {"k": int(len(np.unique(labels))), "n": int(len(labels))}
    if out["k"] < 2:
        return out
    sils = []
    for _ in range(ev["silhouette_draws"]):
        idx = rng.choice(len(X), size=min(ev["silhouette_rows"], len(X)), replace=False)
        if len(np.unique(labels[idx])) > 1:
            sils.append(silhouette_score(X[idx], labels[idx]))
    out["silhouette"] = float(np.mean(sils))
    out["silhouette_sd"] = float(np.std(sils))
    idx = rng.choice(len(X), size=min(ev["geometry_rows"], len(X)), replace=False)
    Xg, lg = X[idx].astype(np.float64), labels[idx]
    out["dbi"] = float(davies_bouldin_score(Xg, lg))
    out["chi"] = float(calinski_harabasz_score(Xg, lg))
    _, uidx = np.unique(Xg, axis=0, return_index=True)
    if len(np.unique(lg[uidx])) > 1:
        out["chi_dedup"] = float(calinski_harabasz_score(Xg[uidx], lg[uidx]))
        out["dedup_fraction"] = float(len(uidx) / len(Xg))
    return out


# ---------------------------------------------------------------------------
# partition agreement
# ---------------------------------------------------------------------------
def variation_of_information(a, b):
    M = contingency_matrix(a, b).astype(np.float64)
    n = M.sum()
    P = M / n
    pa = P.sum(1)
    pb = P.sum(0)
    nz = P > 0
    I = np.sum(P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz]))
    Ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    Hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    return float((Ha + Hb - 2 * I) / np.log(2))


def clusterwise_jaccard(ref, other):
    """For each reference cluster, the best Jaccard similarity with a cluster of `other`."""
    ref = np.asarray(ref)
    other = np.asarray(other)
    ru, ri = np.unique(ref, return_inverse=True)
    ou, oi = np.unique(other, return_inverse=True)
    M = np.zeros((len(ru), len(ou)))
    np.add.at(M, (ri, oi), 1)
    rs = M.sum(1)[:, None]
    os_ = M.sum(0)[None, :]
    J = M / (rs + os_ - M)
    return dict(zip(ru.tolist(), J.max(1).tolist()))


def agreement(a, b):
    return {"ari": float(adjusted_rand_score(a, b)),
            "ami": float(adjusted_mutual_info_score(a, b)),
            "vi_bits": variation_of_information(a, b)}


def matched_centroid_shift(X_ref, ref_labels, other_labels):
    """Centroids of both partitions computed on the same rows/space, Hungarian-matched.

    Returns mean matched centroid distance divided by the RMS within-cluster
    distance of the reference partition.
    """
    X = np.asarray(X_ref, dtype=np.float64)
    def cents(lab):
        u = np.unique(lab)
        return np.vstack([X[lab == c].mean(0) for c in u])
    Cr, Co = cents(ref_labels), cents(other_labels)
    D = np.sqrt(((Cr[:, None, :] - Co[None, :, :]) ** 2).sum(2))
    r, c = linear_sum_assignment(D)
    u = np.unique(ref_labels)
    within = np.sqrt(np.mean([((X[ref_labels == cl] - Cr[i]) ** 2).sum(1).mean() for i, cl in enumerate(u)]))
    return float(D[r, c].mean() / within)


# ---------------------------------------------------------------------------
# cross-fitted concentration with block bootstrap
# ---------------------------------------------------------------------------
class ConcentrationData:
    """Block x cluster count matrices for one partition and one binary target."""

    def __init__(self, labels, target, block, k=None, fold_fn=None):
        labels = np.asarray(labels, dtype=np.int64)
        target = np.asarray(target, dtype=bool)
        block = np.asarray(block, dtype=np.int64)
        self.k = int(labels.max()) + 1 if k is None else k
        ub, bi = np.unique(block, return_inverse=True)
        self.fold_of_block = (ub % 2 if fold_fn is None else fold_fn(ub)).astype(np.int8)
        nb = len(ub)
        tot = np.zeros(nb * self.k)
        pos = np.zeros(nb * self.k)
        key = bi * self.k + labels
        np.add.at(tot, key, 1)
        np.add.at(pos, key[target], 1)
        self.tot = tot.reshape(nb, self.k)
        self.pos = pos.reshape(nb, self.k)
        self.blocks_in_fold = [np.flatnonzero(self.fold_of_block == f) for f in (0, 1)]

    def fold_counts(self, f, weights=None):
        b = self.blocks_in_fold[f]
        if weights is None:
            return self.tot[b].sum(0), self.pos[b].sum(0)
        return self.tot[b].T @ weights, self.pos[b].T @ weights


def _curve(n_cal, p_cal, n_ev, p_ev, min_members):
    lb = wilson_lower(p_cal, n_cal)
    eligible = n_cal >= min_members
    order = np.lexsort((-n_cal, -lb, ~eligible))
    P = p_ev.sum()
    N = n_ev.sum()
    cum_p = np.cumsum(p_ev[order])
    cum_n = np.cumsum(n_ev[order])
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = cum_p / P
        prec = np.where(cum_n > 0, cum_p / cum_n, 0.0)
    return order, cov, prec, P / N, P, N


def _summ(cov, prec, base, coverages):
    dcov = np.diff(np.concatenate([[0.0], cov]))
    ap = float(np.sum(dcov * prec))
    out = {"ap": ap, "ap_lift": ap / base if base > 0 else np.nan, "base_rate": base}
    for q in coverages:
        i = int(np.searchsorted(cov, q - 1e-12))
        i = min(i, len(cov) - 1)
        out[f"enrich@{q:.2f}"] = float(prec[i] / base) if base > 0 else np.nan
        out[f"prec@{q:.2f}"] = float(prec[i])
        out[f"clusters@{q:.2f}"] = i + 1
        out[f"share@{q:.2f}"] = np.nan  # filled by caller when totals are known
    return out


def crossfit(cd, weights=(None, None), with_curve=False, directions=((0, 1), (1, 0)), min_members=None):
    ev = CFG["evaluation"]
    mm = ev["min_members_rank"] if min_members is None else min_members
    res = []
    curves = []
    for a, b in directions:
        n_a, p_a = cd.fold_counts(a, weights[a])
        n_b, p_b = cd.fold_counts(b, weights[b])
        order, cov, prec, base, P, N = _curve(n_a, p_a, n_b, p_b, mm)
        s = _summ(cov, prec, base, ev["coverages"])
        cum_n = np.cumsum(n_b[order])
        for q in ev["coverages"]:
            s[f"share@{q:.2f}"] = float(cum_n[s[f"clusters@{q:.2f}"] - 1] / N)
        s["positives_eval"] = float(P)
        res.append(s)
        if with_curve:
            curves.append({"coverage": cov.tolist(), "precision": prec.tolist(), "base_rate": base,
                           "order": order.tolist()})
    keys = res[0].keys()
    avg = {k: float(np.nanmean([r[k] for r in res])) for k in keys}
    avg["min_positives_fold"] = float(min(r["positives_eval"] for r in res))
    return (avg, curves) if with_curve else avg


def concentration_many(label_sets, target, block, B, seed, metrics=("ap_lift", "enrich@0.10",
                       "enrich@0.25", "enrich@0.50"), reference=None, fold_fn=None,
                       directions=((0, 1), (1, 0)), min_members=None):
    """Point estimates, block-bootstrap CIs and paired differences vs `reference`.

    label_sets: dict name -> labels (same rows)
    """
    target = np.asarray(target, dtype=bool)
    cds = {m: ConcentrationData(l, target, block, fold_fn=fold_fn) for m, l in label_sets.items()}
    any_cd = next(iter(cds.values()))
    nb = [len(any_cd.blocks_in_fold[0]), len(any_cd.blocks_in_fold[1])]
    out = {}
    curves = {}
    for m, cd in cds.items():
        out[m], curves[m] = crossfit(cd, with_curve=True, directions=directions, min_members=min_members)
    rng = np.random.default_rng(seed)
    boots = {m: {k: [] for k in metrics} for m in cds}
    for _ in range(B):
        w = [np.bincount(rng.integers(nb[f], size=nb[f]), minlength=nb[f]).astype(np.float64)
             for f in (0, 1)]
        for m, cd in cds.items():
            r = crossfit(cd, weights=w, directions=directions, min_members=min_members)
            for k in metrics:
                boots[m][k].append(r[k])
    for m in cds:
        out[m]["ci"] = {k: ci(boots[m][k]) for k in metrics}
    if reference is not None:
        for m in cds:
            if m == reference:
                continue
            out[m]["vs_" + reference] = {}
            for k in metrics:
                d = np.asarray(boots[m][k]) - np.asarray(boots[reference][k])
                out[m]["vs_" + reference][k] = {
                    "diff": out[m][k] - out[reference][k],
                    "ci": ci(d), "p": boot_p_two_sided(d)}
    return out, curves
