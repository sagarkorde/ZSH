# Reproducing the v2 study

## 1. Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r env/requirements.lock
```

Reference machine: Windows 11, Intel Core i9-13900HX (32 logical cores),
64 GB RAM. All computation is on the CPU; experiments limit numerical
libraries to 16 threads.

## 2. Inputs

Edit `configs/zsh_v2.json` → `paths`, or set environment variables:

| Variable | Content |
|---|---|
| `ZSH_D1` | `Dataset.parquet` from IEEE DataPort (doi:10.21227/bxmt-mn56) |
| `ZSH_ELLIPTIC` | folder with `elliptic_txs_features.csv`, `elliptic_txs_classes.csv` |
| `ZSH_TAGMAP` | `btc_tagmap.json` (GraphSense TagPacks address → category) |
| `ZSH_OUT` | output folder for large files |

Scripts check the SHA-256 of every input against `configs/zsh_v2.json`.

## 3. Order

```bash
python RUN_ALL.py            # every reported analysis, then tables, figures, checks
python RUN_ALL.py e01 e10    # a range of steps
```

A clean-slate `python RUN_ALL.py` is the reproduction contract for the article:
it runs E0-E21, writes every table and figure of the manuscript and the
supplementary file, and then re-checks every numerical claim in both against the
saved result files, exiting non-zero if any disagrees.

### Pre-specified analyses (frozen at `v2-frozen`)

| Step | Script | Output (repository `results/`) |
|---|---|---|
| e00a | build canonical base table | `E0/base_build.json` |
| e00b | audit, feature selection | `E0/*.csv`, `E0/feature_selection.json` |
| e00c | API data for D1 samples, equivalence check | `E0/api_equivalence.json` |
| e00d | prospective sample (network, ~12 h) | `E0/future_manifest.json` |
| e01 | primary model | `E1/` |
| e02 | method comparison | `E2/` |
| e03 | weighting x refinement | `E3/` |
| e04 | stability | `E4/` |
| e05 | temporal transfer | `E5/` |
| e05b | weight drift | `E5/` |
| e06 | annotation concentration | `E6/` |
| e07 | count-rule validity | `E7/` |
| e08 | Elliptic | `E8/` |
| e09 | atypicality | `E9/` |
| e10 | sensitivity | `E10/` |

### Added after the freeze (see `DEVIATIONS.md`)

These are exploratory follow-ups. They are reported as such in the article and
are not part of the pre-specified confirmatory evidence.

| Step | Script | Output |
|---|---|---|
| e11 | proxy partition and choice of K | `E11/` |
| e12 | profile support over time | `E12/` |
| e13 | actor hold-out | `E13/` |
| e14 | external CoinJoin labels (Wasabi) | `E14/` |
| e15 | oracle weights | `E15/` |
| e16 | profile matching across refits | `E16/` |
| e17 | the whole evaluation for seven clustering families | `E17/` |
| e18 | supervised benchmark on the same twelve features | `E18/` |
| e19 | address-level features against the twelve | `E19/` |
| e20 | centroid-constrained (warm) refit | `E20/` |
| e21 | shared-partition benchmark | `E21/` |

### Manuscript outputs and checks

| Step | Script | Output |
|---|---|---|
| figures | `make_tables_figures.py` | `results/figures/`, `results/tables/` |
| tables | `make_manuscript_tables.py` | `results/manuscript/` (numbered tables) |
| verify | `verify_manuscript_numbers.py` | exits non-zero if any reported number disagrees |

`ZSH_SMOKE=1` runs the same code on DEV data only (results go to
`results/*_smoke`); it was used to test the code before the freeze.

## 3a. Weighting convention for the prospective sample

The prospective sample was drawn with unequal inclusion probabilities (blocks,
then pages within blocks), so `d4_future.parquet` carries a `design_weight`
column. The convention used throughout, and stated in the article:

| Quantity | Weighted? |
|---|---|
| Annotation prevalence and profile shares | design-weighted |
| Concentration: AP, AP lift, enrichment, precision at fixed coverage | design-weighted, with a **month-stratified** block bootstrap |
| Target eligibility (minimum positives per fold) | unweighted, on sampled rows |
| Partition-agreement measures (ARI, AMI, VI, Jaccard) | unweighted; they are properties of the sampled partition, not population totals |

Ranking on the calibration fold uses Kish effective counts
`(sum w)^2 / sum w^2`, so that a cluster carried by few sampled transactions
with large weights is not credited with spurious precision. The evaluation fold
uses the weighted totals directly, so reported precision and lift are estimates
for the sampled frame. With all weights equal, every quantity reduces exactly to
its unweighted form.

Unweighted, sample-specific versions of the prospective concentration results
are written alongside the weighted ones as `*_unweighted.csv` and are reported
in the supplementary file as a sensitivity analysis.

## 4. The prospective sample

`e00d` queries the public Esplora API. Raw responses are cached in
`future/d4_cache.sqlite`; re-running the step reuses the cache and makes no
requests. Block data on a public chain do not change after sufficient
confirmations, so a fresh collection with the same seeds selects the same
blocks and pages.

## 5. Run-to-run agreement

All seeds are fixed, but multithreaded K-means on millions of rows is not
guaranteed to be bit-identical across runs. E10 was run twice (the first run
was interrupted after writing its main tables). Every variant reproduced
exactly except the upsampled corpus (2.4 million rows), whose AP lifts
differed by at most 0.15 (exchange tag 18.47 vs 18.32); its number of clusters
and geometry were identical. No conclusion depends on differences of this size.
