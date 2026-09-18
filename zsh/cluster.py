"""ZSH clustering: hierarchical initialisation, K-means, size-constrained refinement, inference."""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans

from .config import CFG
from .features import Preprocessor
from .weighting import fit_weights


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def assign(X, C, chunk=500_000):
    """Nearest centroid (squared Euclidean) with chunking; returns labels and squared distances."""
    C = np.asarray(C, dtype=np.float64)
    cn = (C ** 2).sum(1)
    labels = np.empty(len(X), dtype=np.int32)
    d2 = np.empty(len(X), dtype=np.float64)
    for a in range(0, len(X), chunk):
        x = np.asarray(X[a:a + chunk], dtype=np.float64)
        D = (x ** 2).sum(1)[:, None] - 2.0 * x @ C.T + cn[None, :]
        j = D.argmin(1)
        labels[a:a + chunk] = j
        d2[a:a + chunk] = np.maximum(D[np.arange(len(x)), j], 0.0)
    return labels, d2


def cluster_means(X, labels, k, sample_weight=None):
    d = X.shape[1]
    sums = np.zeros((k, d))
    cnt = np.zeros(k)
    w = np.ones(len(X)) if sample_weight is None else sample_weight
    for a in range(0, len(X), 1_000_000):
        lab = labels[a:a + 1_000_000]
        x = np.asarray(X[a:a + 1_000_000], dtype=np.float64)
        ww = w[a:a + 1_000_000]
        np.add.at(sums, lab, x * ww[:, None])
        np.add.at(cnt, lab, ww)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / cnt[:, None], cnt


def weighted_ward(centroids, sizes, k):
    """Ward agglomeration of weighted points (Lance-Williams); returns group index per point."""
    m = len(centroids)
    C = [np.asarray(c, dtype=np.float64) for c in centroids]
    n = [float(s) for s in sizes]
    members = [[i] for i in range(m)]
    active = set(range(m))

    def cost(a, b):
        return n[a] * n[b] / (n[a] + n[b]) * float(((C[a] - C[b]) ** 2).sum())

    while len(active) > k:
        best, pair = np.inf, None
        act = sorted(active)
        for ii, a in enumerate(act):
            for b in act[ii + 1:]:
                c = cost(a, b)
                if c < best:
                    best, pair = c, (a, b)
        a, b = pair
        C[a] = (n[a] * C[a] + n[b] * C[b]) / (n[a] + n[b])
        n[a] += n[b]
        members[a] += members[b]
        active.remove(b)
    groups = np.empty(m, dtype=np.int32)
    for g, a in enumerate(sorted(active)):
        groups[members[a]] = g
    return groups


def hierarchical_init(X, k, seed, sample_weight=None):
    ic = CFG["init"]
    micro = MiniBatchKMeans(n_clusters=ic["n_micro"], n_init=ic["micro_n_init"],
                            batch_size=CFG["fit"]["batch_size"], random_state=seed)
    micro.fit(X, sample_weight=sample_weight)
    lab = micro.predict(X)
    w = np.ones(len(X)) if sample_weight is None else sample_weight
    sizes = np.bincount(lab, weights=w, minlength=ic["n_micro"])
    keep = sizes > 0
    cents = micro.cluster_centers_[keep]
    sizes = sizes[keep]
    groups = weighted_ward(cents, sizes, k)
    init = np.vstack([np.average(cents[groups == g], axis=0, weights=sizes[groups == g])
                      for g in range(k)])
    return init


def seed_init(X, family, k, seed, exclude=None):
    """Semantic seeds (v1 allocation): K seeds split across rule families by support."""
    min_pts = CFG["e10"]["seed_min_points"]
    rng = np.random.default_rng(seed)
    fam = np.asarray(family)
    cats, counts = np.unique(fam, return_counts=True)
    support = {int(c): int(n) for c, n in zip(cats, counts)
               if int(c) != 8 and int(c) != exclude and n >= min_pts}
    tot = sum(support.values())
    alloc = {c: max(1, round(k * n / tot)) for c, n in support.items()}
    while sum(alloc.values()) > k:
        v = max(alloc, key=lambda c: (alloc[c], -support[c]))
        if alloc[v] <= 1:
            break
        alloc[v] -= 1
    while sum(alloc.values()) < k:
        alloc[max(alloc, key=lambda c: support[c])] += 1
    cents = []
    for c, kc in sorted(alloc.items()):
        idx = np.flatnonzero(fam == c)
        if len(idx) > 200_000:
            idx = rng.choice(idx, size=200_000, replace=False)
        Xc = X[idx]
        if kc == 1:
            cents.append(Xc.mean(0))
        else:
            km = MiniBatchKMeans(n_clusters=kc, n_init=5, batch_size=20_000, random_state=seed).fit(Xc)
            cents.extend(km.cluster_centers_)
    return np.asarray(cents, dtype=np.float64)[:k]


def blend_init(seed_c, ward_c, lam):
    cost = ((seed_c[:, None, :] - ward_c[None, :, :]) ** 2).sum(2)
    r, c = linear_sum_assignment(cost)
    aligned = np.empty_like(ward_c)
    aligned[r] = ward_c[c]
    return lam * seed_c + (1 - lam) * aligned


def refine(X, labels, cap, depth, seed, sample_weight=None, log=None):
    """Recursive size-constrained splitting (ANALYSIS_PLAN §3.6 step 2)."""
    fc = CFG["fit"]
    labels = labels.astype(np.int32).copy()
    w = np.ones(len(X)) if sample_weight is None else sample_weight
    total = float(w.sum())
    limit = cap * total
    rng = np.random.default_rng(seed)
    depth_of = {int(c): 0 for c in np.unique(labels)}
    queue = sorted(depth_of)
    nxt = int(labels.max()) + 1
    splits = []
    while queue:
        c = queue.pop(0)
        idx = np.flatnonzero(labels == c)
        size = float(w[idx].sum())
        if size <= limit or depth_of[c] >= depth:
            continue
        kk = max(2, int(np.ceil(size / limit)))
        fit_idx = idx if len(idx) <= fc["split_fit_rows"] else rng.choice(idx, fc["split_fit_rows"], replace=False)
        km = KMeans(n_clusters=kk, n_init=fc["split_n_init"], max_iter=fc["max_iter"],
                    tol=fc["tol"], random_state=int(rng.integers(2**31 - 1)))
        km.fit(X[fit_idx], sample_weight=None if sample_weight is None else w[fit_idx])
        sub, _ = assign(X[idx], km.cluster_centers_)
        children = []
        for j in range(kk):
            m = sub == j
            if not m.any():
                continue
            new = c if j == 0 else nxt
            if j != 0:
                nxt += 1
            labels[idx[m]] = new
            depth_of[new] = depth_of[c] + 1
            children.append(new)
        splits.append({"parent": c, "share": size / total, "k": kk, "children": children,
                       "depth": depth_of[c] + 1})
        if log:
            log(f"    split cluster {c} ({size / total:.3f} of rows) into {kk}")
        queue.extend(children)
    return labels, splits


def finalize(X, labels, sample_weight=None):
    """Centroids = cluster means; one nearest-centroid reassignment; drop empty; order by size."""
    k = int(labels.max()) + 1
    C, cnt = cluster_means(X, labels, k, sample_weight)
    C = C[cnt > 0]
    lab, _ = assign(X, C)
    w = np.ones(len(X)) if sample_weight is None else sample_weight
    sizes = np.bincount(lab, weights=w, minlength=len(C))
    keep = np.flatnonzero(sizes > 0)
    order = keep[np.argsort(-sizes[keep], kind="stable")]
    C = C[order]
    lab, _ = assign(X, C)
    return C, lab


# ---------------------------------------------------------------------------
# the ZSH estimator (and its ablation variants)
# ---------------------------------------------------------------------------
class ZSH:
    def __init__(self, features, weighting="rpw", s=None, init="ward", k0=None, cap=None,
                 depth=None, refine=True, lam=None, exclude_family=None, proxy_k=None, name="ZSH"):
        self.features = list(features)
        self.weighting = weighting
        self.s = CFG["weighting"]["s"] if s is None else s
        self.init = init
        self.k0 = CFG["init"]["k0"] if k0 is None else k0
        self.cap = CFG["fit"]["cap"] if cap is None else cap
        self.depth = CFG["fit"]["depth"] if depth is None else depth
        self.refine_on = refine and self.cap is not None
        self.lam = CFG["e10"]["blend_lambda"] if lam is None else lam
        self.exclude_family = exclude_family
        self.proxy_k = proxy_k          # None = the configured value (10)
        self.name = name
        self.timing = {}

    def params(self):
        return {"name": self.name, "features": self.features, "weighting": self.weighting,
                "s": self.s, "init": self.init, "k0": self.k0,
                "cap": self.cap if self.refine_on else None, "depth": self.depth,
                "lam": self.lam if self.init == "blend" else None,
                "exclude_family": self.exclude_family,
                "proxy_k": self.proxy_k or CFG["weighting"]["proxy_k"]}

    def fit(self, df, seed, sample_weight=None, log=None):
        t = time.time()
        self.prep = Preprocessor(self.features).fit(df, sample_weight)
        X = self.prep.transform(df)
        self.timing["preprocess"] = time.time() - t

        t = time.time()
        self.winfo = fit_weights(X, self.prep.is_binary, self.weighting, self.s, seed, sample_weight,
                                 self.proxy_k)
        self.w = np.asarray(self.winfo["w"], dtype=np.float64)
        self.sqrt_w = np.sqrt(self.w).astype(np.float32)
        Xw = X * self.sqrt_w
        del X
        self.timing["weighting"] = time.time() - t

        t = time.time()
        if self.init == "ward":
            C0 = hierarchical_init(Xw, self.k0, seed, sample_weight)
            n_init = 1
        elif self.init == "kmeans++":
            C0, n_init = "k-means++", 10
        elif self.init in ("seeds", "blend"):
            fam = df["L1"].to_numpy()
            C0 = seed_init(Xw, fam, self.k0, seed, self.exclude_family)
            if self.init == "blend":
                C0 = blend_init(C0, hierarchical_init(Xw, self.k0, seed, sample_weight), self.lam)
            n_init = 1
        else:
            raise ValueError(self.init)
        self.timing["init"] = time.time() - t

        t = time.time()
        km = KMeans(n_clusters=self.k0, init=C0, n_init=n_init, max_iter=CFG["fit"]["max_iter"],
                    tol=CFG["fit"]["tol"], random_state=seed, algorithm="lloyd")
        km.fit(Xw, sample_weight=sample_weight)
        labels, _ = assign(Xw, km.cluster_centers_)
        self.kmeans_iter = int(km.n_iter_)
        self.timing["kmeans"] = time.time() - t

        t = time.time()
        self.splits = []
        if self.refine_on:
            labels, self.splits = refine(Xw, labels, self.cap, self.depth, seed, sample_weight, log)
        self.centroids, self.labels_ = finalize(Xw, labels, sample_weight)
        self.timing["refine"] = time.time() - t
        self.k = len(self.centroids)
        w = np.ones(len(Xw)) if sample_weight is None else sample_weight
        self.shares_ = np.bincount(self.labels_, weights=w, minlength=self.k) / w.sum()
        self.timing["total"] = sum(v for kk, v in self.timing.items() if kk != "total")
        return self

    def transform(self, df):
        return self.prep.transform(df) * self.sqrt_w

    def predict(self, df, return_dist=False):
        lab, d2 = assign(self.transform(df), self.centroids)
        return (lab, d2) if return_dist else lab

    def summary(self):
        out = self.params()
        out.update({"k": self.k, "max_share": float(self.shares_.max()),
                    "weights": dict(zip(self.features, self.w.round(6).tolist())),
                    "ranks": (dict(zip(self.features, np.asarray(self.winfo["ranks"]).tolist()))
                              if "ranks" in self.winfo else None),
                    "scores": (dict(zip(self.features, np.asarray(self.winfo["scores"]).round(6).tolist()))
                               if "scores" in self.winfo else None),
                    "kmeans_iter": self.kmeans_iter, "splits": self.splits,
                    "timing_s": {k: round(v, 2) for k, v in self.timing.items()},
                    "preprocessor": self.prep.to_dict()})
        return out


class PlainKMeans:
    """K-means++ baseline on the same preprocessing, optional weighting, fixed K."""

    def __init__(self, features, k, weighting="uniform", s=None, init="kmeans++", name="KMeans"):
        self.features, self.kk, self.weighting = list(features), k, weighting
        self.s = CFG["weighting"]["s"] if s is None else s
        self.init, self.name = init, name
        self.timing = {}

    def fit(self, df, seed, sample_weight=None, log=None):
        t = time.time()
        self.prep = Preprocessor(self.features).fit(df)
        X = self.prep.transform(df)
        self.winfo = fit_weights(X, self.prep.is_binary, self.weighting, self.s, seed)
        self.sqrt_w = np.sqrt(np.asarray(self.winfo["w"])).astype(np.float32)
        Xw = X * self.sqrt_w
        if self.init == "ward":
            km = KMeans(n_clusters=self.kk, init=hierarchical_init(Xw, self.kk, seed), n_init=1,
                        max_iter=CFG["fit"]["max_iter"], tol=CFG["fit"]["tol"], random_state=seed)
        else:
            km = KMeans(n_clusters=self.kk, init="k-means++", n_init=10,
                        max_iter=CFG["fit"]["max_iter"], tol=CFG["fit"]["tol"], random_state=seed)
        km.fit(Xw)
        self.centroids, self.labels_ = finalize(Xw, assign(Xw, km.cluster_centers_)[0])
        self.k = len(self.centroids)
        self.timing["total"] = time.time() - t
        return self

    def transform(self, df):
        return self.prep.transform(df) * self.sqrt_w

    def predict(self, df, return_dist=False):
        lab, d2 = assign(self.transform(df), self.centroids)
        return (lab, d2) if return_dist else lab
