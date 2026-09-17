# Resume point — 17 Sep 2026, 16:35 IST

This file says where the work stands and how to continue.

## Git state

Branch `v2` on https://github.com/sagarkorde/ZSH.

| Tag / commit | Meaning |
|---|---|
| `v1-submitted` → `d7fe3f5` | code cited by the first submission |
| `v2-plan` → `71af80a` | analysis plan written before any analysis |
| `v2-data-audit` → `cdbbd86` | data audit and feature selection |
| `v2-frozen` → `c3daf1d` | frozen analysis code (start of confirmatory runs) |
| `v2-checkpoint-0917` → `0807803` | morning pause point |
| latest `v2` | see `git log`; every change after the freeze is listed in `DEVIATIONS.md` |

## What is finished (results committed in `results/`)

| Step | Status |
|---|---|
| E0a base table, E0b audit, E0c API equivalence (2,000 tx, all features agree) | done |
| E1 primary model (K = 31) | done |
| E2 method comparison | done |
| E3 weighting × refinement | done |
| E4 stability | done |
| E8 Elliptic (Gaussian mixture failed to fit; logged) | done |
| E10 sensitivity | rerun started 15:30 IST (~2.5 h); commit `results/E10` when `e10_sensitivity.log` ends with `done` |

## Prospective sample (E0d) — blocked by the data hosts

* From ~15:30 IST mempool.space refused every connection from this network, and
  blockstream.info answered every request with HTTP 429.
* Cache `D:\ZSH_v2_outputs\future\d4_cache.sqlite`: October 2024 – May 2025
  complete (one page missing in November 2024), June 2025 partial, later months
  empty.
* The collector runs on blockstream.info only, slowly, and resumes by itself when
  the limit lifts:

  ```bash
  ZSH_PAGE_HOSTS=blockstream ZSH_MIN_INTERVAL=2.5 .venv/Scripts/python.exe experiments/e00d_collect_future.py
  ```

  When mempool.space answers again, restart without `ZSH_PAGE_HOSTS` (both hosts),
  keeping `ZSH_MIN_INTERVAL=2.5`.
* **Agreed fallback (authors, 16:20 IST):** if both hosts still refuse requests on
  18 Sep 2026 at 09:00 IST, stop the collector and close the sample from the cache:

  ```bash
  ZSH_FUTURE_CLOSE=1 .venv/Scripts/python.exe experiments/e00d_collect_future.py
  ```

  Months with ≥ 95% of their pages are kept (October 2024 – May 2025); record the
  outcome in `DEVIATIONS.md`.

## After E0d

```bash
for s in e05_transfer e05b_weight_drift e06_concentration e07_heuristic_validity e09_atypicality; do
  .venv/Scripts/python.exe experiments/$s.py || break
done
.venv/Scripts/python.exe experiments/make_tables_figures.py
.venv/Scripts/python.exe experiments/make_manuscript_tables.py
```

Then commit `results/`, tag `v2-data-future` and `v2-results`, and finish the
manuscript.

## Manuscript (local, not in this repository)

Folder `C:\Users\sagar\Desktop\ZHS_23826\Fintech MDPI\Fintech MDPI Revised\v2_work`:

* section drafts `draft_*.md` (front, intro, related, methods, results, discussion,
  conclusion, backmatter, appendix);
* `assemble.py [--draft]` → `manuscript.md` → `build_manuscript.py` →
  `..\ZSH_FinTech_MDPI_manuscript_v2[_draft].docx` (PDF check with Word);
* remaining placeholders: `{FUTURE …}`, `{n_future}`, `{n_future_blocks}`,
  `{ZENODO_DOI}`; `author_notes.md` lists items for the authors.
