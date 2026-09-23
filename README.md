# ZSH: transaction-level behavioural profiling of Bitcoin

ZSH clusters individual Bitcoin transactions using
**Z**eta-normalised rank weighting of features, **S**ize-constrained
refinement and **H**ierarchical (Ward) initialisation, and describes each
cluster with annotations that are not used as clustering inputs.

## Versions

| Branch / tag | Content |
|---|---|
| `v1-submitted` | Code cited by the first manuscript submission (Aug 2026). Kept unchanged in `legacy_v1/`. Its results are superseded by `v2` and are not reported in the published article. |
| `v2` (this branch) | Corrected method, temporal and external validation, prospective data. |
| `v2-final` | The state cited by the published article: every reported analysis (E0-E21), every manuscript and supplementary table and figure, and the checks that regenerate them. Use this tag to reproduce the article. |

The v2 study is specified in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) before any
confirmatory analysis. Changes made after the code freeze (tag `v2-frozen`) are
listed in [`DEVIATIONS.md`](DEVIATIONS.md), which also distinguishes the
pre-specified analyses (E0-E10) from those added after the freeze (E11-E21).

To reproduce everything the article reports, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); `python RUN_ALL.py` from a clean
checkout of `v2-final` runs every analysis and regenerates every table and
figure, then verifies each numerical claim in the manuscript against the saved
results.

This repository contains code only. The numbers reported in the article live in
`results/`; no findings are restated in these documents, so that the article and
the result files remain the single source of truth.

## Environment

Python 3.12 with pinned dependencies:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r env/requirements.lock
```

## Data

* Bitcoin transaction sample (Jul 2022 – Sep 2024), IEEE DataPort:
  <https://doi.org/10.21227/bxmt-mn56>
* Elliptic Bitcoin dataset: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
* GraphSense TagPacks: <https://github.com/graphsense/graphsense-tagpacks>
* Prospective sample (Oct 2024 – Aug 2026): collected with the scripts in this
  repository from the public Esplora API; file list and checksums are
  published with the results.

Input checksums are listed in `ANALYSIS_PLAN.md` §2.

## Licence

MIT (see `LICENSE`).
