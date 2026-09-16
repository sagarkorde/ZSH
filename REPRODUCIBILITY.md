# Reproducing the v2 study

## 1. Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r env/requirements.lock
```

Reference machine: Windows 11, Intel Core i9 (13th gen, 32 logical cores),
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
python RUN_ALL.py            # everything
python RUN_ALL.py e01 e10    # a range of steps
```

| Step | Script | Output (repository `results/`) |
|---|---|---|
| e00a | build canonical base table | `E0/base_build.json` |
| e00b | audit, feature selection | `E0/*.csv`, `E0/feature_selection.json` |
| e00c | API data for D1 samples, equivalence check | `E0/api_equivalence.json` |
| e00d | prospective sample (network, ~12 h) | `E0/future_manifest.json` |
| e01 | primary model | `E1/` |
| e02 | method comparison | `E2/` |
| e03 | weighting × refinement | `E3/` |
| e04 | stability | `E4/` |
| e05 | temporal transfer | `E5/` |
| e06 | annotation concentration | `E6/` |
| e07 | count-rule validity | `E7/` |
| e08 | Elliptic | `E8/` |
| e09 | atypicality | `E9/` |
| e10 | sensitivity | `E10/` |
| figures | tables and figures for the article | `tables/`, `figures/` |

`ZSH_SMOKE=1` runs the same code on DEV data only (results go to
`results/*_smoke`); it was used to test the code before the freeze.

## 4. The prospective sample

`e00d` queries the public Esplora API. Raw responses are cached in
`future/d4_cache.sqlite`; re-running the step reuses the cache and makes no
requests. Block data on a public chain do not change after sufficient
confirmations, so a fresh collection with the same seeds selects the same
blocks and pages.
