# Resume point — 17 Sep 2026, 07:10 IST

Work was paused on purpose (machine shut down). Nothing was lost; this file
says exactly where to continue.

## Git state

Branch `v2` on https://github.com/sagarkorde/ZSH.

| Tag / commit | Meaning |
|---|---|
| `v1-submitted` → `d7fe3f5` | code cited by the first submission |
| `v2-plan` → `71af80a` | analysis plan written before any analysis |
| `v2-data-audit` → `cdbbd86` | data audit and feature selection |
| `v2-frozen` → `c3daf1d` | frozen analysis code (start of confirmatory runs) |
| `v2-checkpoint-0917` | this resume point (see `git log`) |

Changes after the freeze are listed in `DEVIATIONS.md`.

## What is finished (results committed in `results/`)

| Step | Status |
|---|---|
| E0a base table, E0b audit, E0c API equivalence (12,000 tx, all features agree) | done |
| E1 primary model (K = 31) | done |
| E2 method comparison | done |
| E3 weighting × refinement | done |
| E4 stability | done |
| E8 Elliptic (GMM failed to fit; logged) | done |

## What is not finished

| Step | State | Action |
|---|---|---|
| E0d prospective sample | block headers done; 6,091 of 18,400 pages cached in `D:\ZSH_v2_outputs\future\d4_cache.sqlite` | rerun; it resumes from the cache (~3 h) |
| E10 sensitivity | was in its final evaluation when stopped; nothing written | rerun from the start (~90 min) |
| E5, E5b, E6, E7, E9 | preliminary runs on TEST/Elliptic only (files in `results/E5`, `E6`, `E7`, `E9` are **not final and not committed**) | run after E0d finishes |

## Commands to resume (PowerShell, in this folder)

```powershell
cd C:\Users\sagar\Desktop\ZHS_23826\ZSH_repo_clean
# 1. prospective data (network, resumes from cache)
.venv\Scripts\python.exe experiments\e00d_collect_future.py
# 2. in a second window, at the same time
.venv\Scripts\python.exe experiments\e10_sensitivity.py
# 3. after step 1 has finished
.venv\Scripts\python.exe RUN_ALL.py e05 e09
.venv\Scripts\python.exe experiments\make_tables_figures.py
```

Then commit `results/` (all E*), tag `v2-results`, and continue with the
manuscript.

## Manuscript work (local, not in this repository)

Folder: `C:\Users\sagar\Desktop\ZHS_23826\Fintech MDPI\Fintech MDPI Revised\v2_work`

| File | Content |
|---|---|
| `draft_intro.md`, `draft_related.md`, `draft_methods.md`, `draft_results.md`, `draft_discussion.md`, `draft_backmatter.md` | section drafts; placeholders in `{…}` |
| `references.json` | 60 references checked against CrossRef / publisher pages |
| `references_verified.md` | how each reference was checked |
| `build_manuscript.py` | Markdown → MDPI .docx (uses the submitted manuscript as template); tested |

Still to write once all results exist: RQ3 results (E6, E7), prospective-period
parts of RQ2 (E5), sensitivity (E10), abstract, conclusions, final assembly.

## Findings so far (for the manuscript)

* Rank-power weights emphasise transaction size (serialised size 0.49, virtual size 0.17).
* E2: at matched K, ZSH concentrates 8 of 11 independent annotations more than
  K-means++ (script types, OP_RETURN protocols); K-means++ is better for
  exchange tags, coinbase and mixed-script inputs; ZSH geometry is weaker.
* E3: weighting changes concentration in both directions (H1 supported);
  refinement changes it negligibly at matched K (its role is the size cap);
  hierarchical initialisation matters.
* E4: refit ARI ≈ 0.83 (K-means++ 0.76); permuted-data reference 0.61;
  13/31 profiles stable (Jaccard ≥ 0.75), 2 unstable.
* E8: Elliptic temporal split — weak concentration (AP lift 1.57, precision
  10.5% at 25% coverage, base 6.5%); better than K-means++ (1.12);
  random forest AP 0.78. H5 supported but small.
* E9 (preliminary): atypicality scores are anti-associated with illicit status
  on Elliptic (IF ROC-AUC 0.175); on Bitcoin, IF fitted before Runes ranks 2024
  Runes as atypical (ROC-AUC 0.93).
* E5 (preliminary, TEST): transferred vs refitted partition ARI 0.21; Runes
  absorbed into P05/P07/P10/P22; refit weights re-rank features (τ = 0.48).
