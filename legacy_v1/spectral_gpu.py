# -*- coding: utf-8 -*-
"""GPU normalised spectral embedding, mirroring sklearn's spectral_embedding.

sklearn's SpectralEmbedding is CPU-only and its ARPACK solve converges very
slowly on the disconnected landmark graph. This computes the same quantity on
the RTX 4060 via cupy's sparse eigensolver.

Mirrors sklearn: binary k-NN connectivity affinity, symmetrised; normalised
Laplacian; drop_first=False (the trivial eigenvector is kept, as
SpectralEmbedding does); diffusion-map scaling by 1/sqrt(degree).
"""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


def spectral_embedding_gpu(X, n_components=10, n_neighbors=15, verbose=True):
    n = X.shape[0]
    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False, n_jobs=-1)
    W = W.maximum(W.T).tocsr()                      # symmetrise, as sklearn does
    if verbose:
        ncomp, _ = sp.csgraph.connected_components(W, directed=False)
        print(f"    graph: {W.nnz:,} edges, {ncomp} connected components", flush=True)

    d = np.asarray(W.sum(axis=1)).ravel()
    d_safe = np.where(d > 0, d, 1.0)
    dis = 1.0 / np.sqrt(d_safe)
    M = sp.diags(dis) @ W @ sp.diags(dis)           # D^-1/2 W D^-1/2
    M = ((M + M.T) * 0.5).tocsr()                   # enforce exact symmetry

    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csl

    Mg = csp.csr_matrix(
        (cp.asarray(M.data, dtype=cp.float64),
         cp.asarray(M.indices), cp.asarray(M.indptr)), shape=M.shape)
    if verbose:
        print(f"    eigsh on GPU: {n:,} x {n:,}, k={n_components} ...", flush=True)

    # Largest algebraic eigenvalues of M == smallest of the normalised Laplacian.
    vals, vecs = csl.eigsh(Mg, k=n_components, which="LA", tol=1e-6, maxiter=5000)
    vals = cp.asnumpy(vals)
    vecs = cp.asnumpy(vecs)

    order = np.argsort(-vals)                       # descending, trivial first
    vecs = vecs[:, order]
    emb = vecs * dis[:, None]                       # diffusion-map scaling
    if verbose:
        print(f"    eigenvalues: {np.round(vals[order], 5)}", flush=True)
    return np.ascontiguousarray(emb, dtype=np.float64)


# Validated against sklearn's SpectralEmbedding at n = 6,000 (KMeans ARI 0.86)
# and at n = 5,000 / 10,000 / 20,000, where both solvers realise all 30 clusters
# with comparable geometry. Above ~40,000 landmarks sklearn's ARPACK solve does
# not converge in practical time on this graph, while this routine returns in
# seconds; see ARTIFACT_MAP.md.
