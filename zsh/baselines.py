"""Standard clustering baselines with a common fit/predict interface."""
import time

import numpy as np
from sklearn.cluster import (HDBSCAN, AgglomerativeClustering, Birch, KMeans, MiniBatchKMeans,
                             kmeans_plusplus)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .cluster import assign
from .config import CFG
from .features import Preprocessor


class _Base:
    name = "base"

    def __init__(self, features, k):
        self.features, self.kk = list(features), k
        self.timing = {}

    def space(self, df):
        return self.prep.transform(df)

    def _prep(self, df):
        self.prep = Preprocessor(self.features).fit(df)
        return self.prep.transform(df)


class KMeansU(_Base):
    name = "KMeans++"

    def fit(self, df, seed):
        t = time.time()
        X = self._prep(df)
        km = KMeans(self.kk, init="k-means++", n_init=10, max_iter=CFG["fit"]["max_iter"],
                    tol=CFG["fit"]["tol"], random_state=seed).fit(X)
        self.C = km.cluster_centers_
        self.timing["fit"] = time.time() - t
        return self

    def predict(self, df):
        return assign(self.space(df), self.C)[0]


class MiniBatchU(KMeansU):
    name = "MiniBatchKMeans"

    def fit(self, df, seed):
        t = time.time()
        X = self._prep(df)
        km = MiniBatchKMeans(self.kk, n_init=10, batch_size=CFG["fit"]["batch_size"],
                             random_state=seed).fit(X)
        self.C = km.cluster_centers_
        self.timing["fit"] = time.time() - t
        return self


class GMMU(_Base):
    name = "GMM (diag)"

    def fit(self, df, seed):
        t = time.time()
        X = self._prep(df).astype(np.float64)
        for reg in (1e-4, 1e-3, 1e-2):
            try:
                self.m = GaussianMixture(self.kk, covariance_type="diag", max_iter=200, n_init=1,
                                         init_params="k-means++", reg_covar=reg, random_state=seed).fit(X)
                self.reg_covar = reg
                break
            except ValueError:
                continue
        self.timing["fit"] = time.time() - t
        return self

    def predict(self, df):
        X = self.space(df).astype(np.float64)
        return np.concatenate([self.m.predict(X[a:a + 500_000]) for a in range(0, len(X), 500_000)])


class BirchU(_Base):
    name = "BIRCH"

    def fit(self, df, seed):
        t = time.time()
        X = self._prep(df)
        self.m = Birch(n_clusters=self.kk, threshold=0.5).fit(X)
        self.timing["fit"] = time.time() - t
        return self

    def predict(self, df):
        X = self.space(df)
        return np.concatenate([self.m.predict(X[a:a + 500_000]) for a in range(0, len(X), 500_000)])


class WardU(_Base):
    name = "Ward (sample + NC)"

    def fit(self, df, seed):
        t = time.time()
        X = self._prep(df)
        rng = np.random.default_rng(seed)
        n = min(CFG["e2"]["ward_rows"], len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        lab = AgglomerativeClustering(self.kk, linkage="ward").fit_predict(X[idx])
        self.C = np.vstack([X[idx][lab == c].mean(0) for c in range(self.kk)])
        self.timing["fit"] = time.time() - t
        return self

    def predict(self, df):
        return assign(self.space(df), self.C)[0]


class HDBSCANU(_Base):
    """Evaluated on its own fitting sample only (no out-of-sample assignment)."""
    name = "HDBSCAN"

    def fit(self, df, seed):
        t = time.time()
        X = self._prep(df)
        rng = np.random.default_rng(seed)
        n = min(CFG["e2"]["hdbscan_rows"], len(X))
        self.idx = rng.choice(len(X), size=n, replace=False)
        self.X = X[self.idx]
        self.labels_ = HDBSCAN(min_cluster_size=max(5, int(0.001 * n))).fit_predict(self.X)
        self.timing["fit"] = time.time() - t
        return self


def trimmed_kmeans(X, k, alpha, seed, n_init=10, max_iter=100):
    """Trimmed k-means (Cuesta-Albertos et al., 1997): best of n_init runs by trimmed SSE."""
    rng = np.random.default_rng(seed)
    n = len(X)
    keep_n = int(np.floor((1 - alpha) * n))
    best = None
    for _ in range(n_init):
        C, _ = kmeans_plusplus(X, k, random_state=int(rng.integers(2**31 - 1)))
        for _ in range(max_iter):
            lab, d2 = assign(X, C)
            keep = np.argsort(d2)[:keep_n]
            newC = np.vstack([X[keep][lab[keep] == c].mean(0) if np.any(lab[keep] == c) else C[c]
                              for c in range(k)])
            if np.allclose(newC, C, atol=1e-8):
                break
            C = newC
        lab, d2 = assign(X, C)
        sse = np.sort(d2)[:keep_n].sum()
        if best is None or sse < best[0]:
            best = (sse, C, np.sort(d2)[keep_n - 1] if keep_n else np.inf)
    return best[1], best[2]


class VKVPartial:
    """After Vlahavas, Karasavvas and Vakali (2024): z-score -> PCA(3) -> trimmed k-means.

    Only 5 of their 8 features exist in this corpus (inputs, outputs, total amount,
    OP_RETURN presence, fee per byte); coinbase distance, CoinJoin distance and
    blocks-unspent need full transaction-graph traversal.
    """
    FEATURES = ["input_count", "output_count", "total_input_value", "has_op_return", "fee_rate_sat_per_byte"]

    def __init__(self, k, name=None):
        self.kk = k
        self.name = name or f"VKV-partial (k={k})"
        self.timing = {}

    def space(self, df):
        return self.pca.transform(self.sc.transform(df[self.FEATURES].to_numpy(np.float64)))

    def fit(self, df, seed):
        t = time.time()
        X = df[self.FEATURES].to_numpy(np.float64)
        self.sc = StandardScaler().fit(X)
        ec = CFG["e12"]
        self.pca = PCA(n_components=ec["pca_components"], random_state=seed).fit(self.sc.transform(X))
        Z = self.space(df)
        self.C, self.trim_d2 = trimmed_kmeans(Z, self.kk, ec["trim_alpha"], seed)
        self.explained = self.pca.explained_variance_ratio_.tolist()
        self.timing["fit"] = time.time() - t
        return self

    def predict(self, df):
        return assign(self.space(df), self.C)[0]
