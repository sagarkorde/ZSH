# State — 19 Sep 2026, 13:20 IST

All confirmatory experiments are finished and their results are committed.

## Git state

Branch `v2` on https://github.com/sagarkorde/ZSH.

| Tag | Commit | Meaning |
|---|---|---|
| `v1-submitted` | `d7fe3f5` | code cited by the first submission |
| `v2-plan` | `71af80a` | analysis plan, written before any analysis |
| `v2-data-audit` | `cdbbd86` | data audit and feature selection |
| `v2-frozen` | `c3daf1d` | frozen analysis code (start of the confirmatory runs) |
| `v2-data-future` | `b52674b` | prospective sample collected (23 months) |
| `v2-results` | `ae27f7e` | all results E0–E10 |

Changes made after `v2-frozen` are listed with reasons in `DEVIATIONS.md`.

## Experiments

| Step | Content | State |
|---|---|---|
| E0a–E0c | base table, audit, API equivalence (2,000 tx) | done |
| E0d | prospective sample: 456,292 transactions, 18,400 blocks, Oct 2024 – Aug 2026 | done |
| E1 | primary model, K = 31 | done |
| E2 | method comparison at matched K | done |
| E3 | weighting × refinement | done |
| E4 | stability (seeds, block bootstrap, permuted null) | done |
| E5, E5b | transfer to test and prospective periods, weight drift | done |
| E6 | concentration of non-input annotations, both periods | done |
| E7 | count rule vs equal-output CoinJoin rule | done |
| E8 | Elliptic, temporal split (Gaussian mixture failed to fit; logged) | done |
| E9 | atypicality scores | done |
| E10 | sensitivity and ablations | done |
| E11–E16 | proxy size, profile support, actor hold-out, external CoinJoin labels, oracle weights, profile matching (all added after the freeze) | done |
| E17 | the whole evaluation for seven clustering families (added after the freeze) | done |
| E18 | supervised bound on the same twelve features (added after the freeze) | done |
| E19 | address-level features from the prospective cache (added after the freeze) | done |

Rebuild tables and figures: `make_tables_figures.py`, then `make_manuscript_tables.py`.

## Main findings

* Weights change *which* properties the profiles concentrate, in both directions;
  refinement caps cluster size but changes concentration by ≤ 0.21 AP lift.
* All eleven non-input annotations are concentrated above their base rates in the
  test period; K-means++ is stronger for exchange tags and mixed-script inputs.
* Read against the ceiling 1/π, the profiles capture 78–84% of what is attainable for
  common annotations and under 11% for rare ones; oracle weights do not lift this.
* The binding limit is the objective, not the features: a supervised model on the same
  twelve features attains 90.9% of the ceiling for P2SH inputs against 23.6% for the
  profiles. Exchange tags are the exception (7.4% against 7.0%), and adding address
  reuse and witness features does not lift them (4.8% with 24 features).
* Seven clustering families share the ceiling on rare annotations, the collapse of the
  partition by 2026 and the failure of profile matching; ZSH has the highest median
  attained share (22.6%) and much the strongest P2PKH concentration (80.9%).
* Refits reproduce the partition (ARI 0.83; permuted-data reference 0.61);
  13 of 31 profiles are stable in Hennig's sense.
* Transfer is weak: a 2024 refit agrees with the transferred partition at
  ARI 0.21, a prospective refit at 0.11, and by 2026 three profiles hold three
  quarters of the transactions. Concentration survives, balance does not.
* The published "coinjoin-like" count rule has a precision of 9.6% (2022–2024)
  and 7.8% (2024–2026) against the equal-output rule.
* Elliptic: AP lift 1.57, precision 10.5% at 25% coverage (base 6.5%); a random
  forest reaches 91% precision. Atypicality is anti-associated with illicit
  status, but flags Runes and equal-output CoinJoins as unusual.

## Manuscript

Folder `C:\Users\sagar\Desktop\ZHS_23826\Fintech MDPI\Fintech MDPI Revised`:

* `ZSH_FinTech_MDPI_manuscript_v2.docx` / `.pdf` — 37 pages, 14 tables, 12 figures;
* `ZSH_response_to_reviewers_v2.docx` / `.pdf` — point-by-point response, 9 pages;
* sources in `v2_work` (`draft_*.md`, `response_to_reviewers.md`, `references.json`),
  built by `python assemble.py` (add `--draft` to allow unfilled placeholders);
* backups of `v2_work` in `D:\ZSH_v2_outputs\manuscript_backup`.

Open for the authors (`v2_work/author_notes.md`): author contributions, the
AI-use statement, and the archive record for the prospective sample
(`{ZENODO_DOI}` in the Data Availability Statement and in the response letter).
