# -*- coding: utf-8 -*-
"""One (n, solver) probe of the landmark spectral embedding.

Answers whether the Table A1 collapse is a property of the graph or an artifact
of the eigensolver, by running both solvers over increasing landmark sizes and
recording connectivity, realised cluster count, and geometry.

Usage:  python solver_probe.py <n> <cpu|gpu>
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

n = int(sys.argv[1])
solver = sys.argv[2]
B = os.environ.get("ZSH_OUTPUT_DIR", "outputs")
K, SEED, NN = 30, 42, 15

Xg = np.load(f"{B}/X_graph_scaled.npy", mmap_mode="r")
land = np.sort(np.load(f"{B}/spectral_sample_idx.npy"))[:n]
X = np.asarray(Xg[land], dtype=np.float32)

W = kneighbors_graph(X, NN, mode="connectivity", include_self=False, n_jobs=-1)
W = W.maximum(W.T).tocsr()
ncomp, _ = sp.csgraph.connected_components(W, directed=False)
uniq = len(np.unique(X, axis=0))

t0 = time.time()
if solver == "gpu":
    sys.path.insert(0, ".")
    from spectral_gpu import spectral_embedding_gpu
    E = spectral_embedding_gpu(X, 10, NN, verbose=False)
else:
    from sklearn.manifold import SpectralEmbedding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E = SpectralEmbedding(n_components=10, affinity="nearest_neighbors",
                              n_neighbors=NN, random_state=SEED,
                              n_jobs=-1).fit_transform(X)
elapsed = time.time() - t0

Es = StandardScaler().fit_transform(E)
lab = KMeans(K, init="k-means++", n_init=10, max_iter=250,
             algorithm="elkan", random_state=SEED).fit_predict(Es)
realized = int(len(np.unique(lab)))

sub = np.random.default_rng(SEED).choice(len(Es), size=min(10_000, len(Es)),
                                         replace=False)
sil = float(silhouette_score(Es[sub], lab[sub])) if len(np.unique(lab[sub])) > 1 else float("nan")

print(json.dumps({"n": n, "solver": solver, "seconds": round(elapsed, 2),
                  "components": int(ncomp), "unique_rows": int(uniq),
                  "clusters_realized": realized, "silhouette": round(sil, 4)}))
