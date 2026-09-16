"""Feature relevance and rank-power (generalised-harmonic normalised) weights."""
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

from .config import CFG


def generalized_harmonic(d, s):
    k = np.arange(1, d + 1, dtype=np.float64)
    return float(np.sum(k ** (-s)))


def rank_power(ranks, s):
    """w_j = r_j^(-s) / H_d(s); s = 0 gives uniform weights."""
    ranks = np.asarray(ranks, dtype=np.float64)
    w = ranks ** (-s)
    return w / w.sum()


def ranks_from_scores(scores, higher_is_better=True):
    """Ranks 1..d; ties broken by feature order (priority list order)."""
    scores = np.asarray(scores, dtype=np.float64)
    key = -scores if higher_is_better else scores
    order = np.lexsort((np.arange(len(scores)), key))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def proxy_partition(X, seed, sample_weight=None):
    wc = CFG["weighting"]
    km = MiniBatchKMeans(n_clusters=wc["proxy_k"], n_init=wc["proxy_n_init"],
                         batch_size=CFG["fit"]["batch_size"], random_state=seed)
    km.fit(X, sample_weight=sample_weight)
    return km.predict(X)


def mi_scores(X, labels, is_binary, seed, sample_weight=None):
    """MI between each feature and the proxy labels on a class-balanced subsample.

    With sample weights, rows are drawn within each proxy cluster with probability
    proportional to their weight (with replacement).
    """
    wc = CFG["weighting"]
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        take = min(wc["mi_per_cluster"], len(members))
        if sample_weight is None:
            idx.append(rng.choice(members, size=take, replace=False))
        else:
            p = sample_weight[members] / sample_weight[members].sum()
            idx.append(rng.choice(members, size=take, replace=True, p=p))
    idx = np.sort(np.concatenate(idx))
    return mutual_info_classif(X[idx], labels[idx], discrete_features=np.asarray(is_binary),
                               n_neighbors=wc["mi_neighbors"], random_state=seed)


def laplacian_scores(X, seed):
    """Laplacian score (He, Cai and Niyogi, 2005); smaller means more relevant."""
    wc = CFG["weighting"]
    rng = np.random.default_rng(seed)
    n = min(wc["laplacian_rows"], len(X))
    Xs = X[rng.choice(len(X), size=n, replace=False)].astype(np.float64)
    k = wc["laplacian_knn"]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    dist, ind = nn.kneighbors(Xs)
    dist, ind = dist[:, 1:], ind[:, 1:]
    t = np.mean(dist ** 2) or 1.0
    rows = np.repeat(np.arange(n), k)
    S = sparse.csr_matrix((np.exp(-dist.ravel() ** 2 / t), (rows, ind.ravel())), shape=(n, n))
    S = S.maximum(S.T)
    deg = np.asarray(S.sum(axis=1)).ravel()
    scores = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        f = Xs[:, j]
        f_t = f - (f @ deg) / deg.sum()
        denom = f_t @ (deg * f_t)
        if denom <= 0:
            scores[j] = np.inf
            continue
        num = denom - f_t @ (S @ f_t)
        scores[j] = num / denom
    return scores


def fit_weights(X, is_binary, scheme, s, seed, sample_weight=None):
    """Return dict with weights and the relevance information behind them.

    scheme: 'uniform' | 'rpw' (MI rank-power) | 'mi_direct' | 'laplacian_rpw'
    """
    d = X.shape[1]
    info = {"scheme": scheme, "s": s}
    if scheme == "uniform":
        info["w"] = np.full(d, 1.0 / d)
        return info
    if scheme in ("rpw", "mi_direct"):
        proxy = proxy_partition(X, seed, sample_weight)
        scores = mi_scores(X, proxy, is_binary, seed, sample_weight)
        info["scores"] = scores
        info["proxy_sizes"] = np.bincount(proxy).tolist()
        if scheme == "mi_direct":
            info["w"] = scores / scores.sum() if scores.sum() > 0 else np.full(d, 1.0 / d)
        else:
            info["ranks"] = ranks_from_scores(scores, True)
            info["w"] = rank_power(info["ranks"], s)
        return info
    if scheme == "laplacian_rpw":
        scores = laplacian_scores(X, seed)
        info["scores"] = scores
        info["ranks"] = ranks_from_scores(scores, higher_is_better=False)
        info["w"] = rank_power(info["ranks"], s)
        return info
    raise ValueError(scheme)
