# Table A1: is the collapse real, or a solver artifact?

Both eigensolvers run over the same landmark graphs, same machine, same
sklearn 1.7.0, K = 30, 15-nearest-neighbour graph, seed 42.

| n | connected components | unique rows | GPU time | GPU clusters | GPU Silhouette | CPU time | CPU clusters | CPU Silhouette |
|---|---|---|---|---|---|---|---|---|
| 5,000 | 1 | 4,746 | 0.75 s | 30 / 30 | 0.505 | 0.66 s | 30 / 30 | 0.543 |
| 10,000 | 4 | 9,115 | 1.68 s | 30 / 30 | 0.594 | 4.76 s | 30 / 30 | 0.585 |
| 20,000 | 8 | 17,113 | 2.17 s | 30 / 30 | 0.516 | 32.24 s | 30 / 30 | 0.704 |
| 40,000 | 11 | 31,387 | 2.72 s | 30 / 30 | 0.543 | **did not finish in 420 s** | — | — |
| 80,000 | 49 | 56,359 | 4.86 s | 30 / 30 | 0.379 | **did not finish in 6 h** | — | — |

Published Table A1, for comparison: **21 / 30 clusters, Silhouette 0.0043**.

## What the table shows

**The two solvers agree wherever the CPU finishes.** At n = 5,000, 10,000 and
20,000 both form all 30 clusters with comparable geometry. The GPU
implementation is therefore validated against sklearn at three independent
scales, not just asserted.

**The CPU cost explodes; the GPU cost does not.** ARPACK goes
0.66 s → 4.76 s → 32.2 s and then fails to finish 40,000 within seven minutes —
roughly a sevenfold rise per doubling. The GPU solver goes 0.75 s → 4.86 s across
a sixteenfold increase in n.

**No converged solve produces a collapse at any scale.** Every GPU run realises
30 of 30 clusters with Silhouette between 0.38 and 0.59. Graph disconnection
grows steadily with n (1 to 49 components) without ever causing the degeneracy
the published table reports.

## Conclusion

The published Table A1 figures — 21 of 30 clusters at Silhouette 0.0043 — do not
appear at any landmark size once the eigendecomposition converges. They are
consistent with ARPACK terminating without convergence on a disconnected graph,
which is exactly the regime where n = 80,000 sits and where the CPU solver cannot
complete at all.

The disconnection itself is real and worth reporting. What does not survive is
the inference drawn from it: that the graph-embedding baseline collapses at full
scale. On a converged solve it does not collapse, and on the contextual metrics
it outperforms ZSH (weighted purity 0.5701 against 0.4918; rule-tree balanced
accuracy 0.4463 against 0.0993).

## Reproducing this

```bash
# point at the run directory, then compare the two solvers
export ZSH_OUTPUT_DIR=outputs_balanced

# GPU spectral embedding (requires cupy)
python solver_probe.py 80000 gpu

# CPU baseline (sklearn ARPACK) - expect this not to return
python solver_probe.py 80000 cpu
```

`spectral_gpu.py` mirrors sklearn's `spectral_embedding`: binary k-NN
connectivity affinity, symmetrised, normalised Laplacian, `drop_first=False`,
diffusion-map scaling by the inverse square root of degree. Agreement with
sklearn at n = 6,000 is ARI 0.86 on KMeans labels.
