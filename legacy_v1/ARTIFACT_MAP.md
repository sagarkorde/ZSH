# Artifact Map

Maps each manuscript table and figure to the pipeline step and output artifact
that produced it. Numbering follows the manuscript (Tables 1–14 plus Appendix
Tables A1 and A2, Figures 1–9).

## Output directories

The pipeline writes to three directories, one per corpus:

| Directory | Corpus | Result |
|---|---|---|
| `outputs_balanced/` | class-balanced, 11,303,526 transactions | 43 profiles |
| `outputs_raw/` | raw, unbalanced, 5,884,387 transactions | 41 profiles |
| `outputs_elliptic/` | Elliptic Bitcoin Dataset, 46,564 labelled | 30 clusters |

Unless stated otherwise, paths below are relative to `outputs_balanced/`.

## Tables

| Table | Content | Step | Source artifact |
|---|---|---|---|
| 1 | Core experimental settings | — | manuscript values; run parameters echoed in `pipeline_runner_log.txt` |
| 2 | Wall-clock fitting time per method | 7 | `step7_log.txt` (per-method elapsed timestamps at each "Fitting …" line) |
| 3 | Same-space comparison, 17 methods | 7 | `step7_sota_comparison.csv` |
| 4 | Ablation study | 7 | `step7_ablation_study.csv` |
| 5 | Sensitivity to `s` and `lambda` | 3, 5 | `step7_s_grid_search.csv` (decay exponent), `zsh_lambda_grid_search.csv` (blend weight) |
| 6 | External validation, GraphSense tags | — | `outputs_raw/step6_graphsense_external_validation.json` and `.txt` — **regenerated against a newer tag snapshot; does not reproduce the published values, see "Items that did not reproduce"** |
| 7 | Representative profiles and anomaly rates | 6 | `profile_statistics.csv`, `RESULTS_SUMMARY.txt` |
| 8 | Contextual profiling comparison | 8 | `step8_contextual_comparison_summary.csv` (per-repeat values in `step8_contextual_comparison_raw.csv`) |
| 9 | Leave-one-family-out recovery of withheld rule families | — | `step_loo_rule_holdout_summary.csv`, `step_loo_rule_holdout_raw.csv`, `step_loo_rule_holdout.json` (`leave_rules_out.py`) |
| 11 | Anomaly layer against Elliptic illicit labels | — | `outputs_elliptic/step9b_anomaly_validation.json`, `step9b_anomaly_topk.csv` (`anomaly_validation.py`) |
| 12 | Feature-weighting scheme comparison | — | `step_weighting_comparison_summary.csv`, `step_weighting_comparison_raw.csv` (`weighting_comparison.py`) |
| 12 | Minority-stratum upsampling sensitivity | 2 | `balance_fraction_sensitivity.csv` |
| 13 | Balanced versus raw corpus | 5, 6 | `outputs_balanced/RESULTS_SUMMARY.txt` against `outputs_raw/RESULTS_SUMMARY.txt` |
| 14 | Independent replication on Elliptic | 9 | `outputs_elliptic/step9_elliptic_report.txt`, `outputs_elliptic/step9_cluster_illicit_breakdown.csv` |
| A1 | Graph-embedding baseline under two eigensolvers | — | ARPACK column: original run, no saved artifact. Converged column: `stepA1_graph_baseline_fullscale.csv`, `stepA1_graph_baseline_fullscale_raw.csv`, `stepA1_graph_baseline_report.txt` (`spectral_gpu.py`) |
| A2 | Eigensolver behaviour across landmark scales | — | `solver_probe.py`, results tabulated in `solver_convergence_evidence.md` |

## Figures

| Figure | Content | Source artifact |
|---|---|---|
| 1 | ZSH system architecture pipeline | author-prepared diagram, not a script output |
| 2 | Top 10 features by zeta weight | `zeta_weights.png` (ranks in `feature_ranks.csv`) |
| 3 | ZSH pipeline flowchart | author-prepared diagram, not a script output |
| 4 | Bootstrap confidence intervals | `fig9_bootstrap_ci.png` |
| 5 | Same-space benchmark comparison | `fig10_sota_comparison.png` |
| 6 | Ablation contribution | `fig11_ablation_study.png` |
| 7 | Silhouette against `s` and `lambda` | `zeta_sensitivity.png` |
| 8 | Profile feature heatmap | `fig3_profile_heatmap.png` |
| 9 | Contextual profiling comparison | `fig12_contextual_comparison.png` |

Output file names predate the manuscript's figure numbering, so for example
`fig9_bootstrap_ci.png` is Figure 4 in the paper. The names are left unchanged so
they continue to match what the pipeline actually writes.

## Supplementary visuals

Referenced in Section 5.6 but not reproduced in full in the manuscript:

- UMAP profile map — `fig1_cluster_umap.png`
- anomaly overlay — `fig2_anomaly_overlay.png`
- cluster size distribution — `fig4_cluster_sizes.png`
- temporal profile distribution — `fig5_temporal_distribution.png`
- outlier rate by profile — `fig6_outlier_by_profile.png`
- radar fingerprints — `fig7_radar_fingerprints.png`
- intrinsic metrics comparison — `fig8_metrics_comparison.png`

## Statistical validation

- Bootstrap, permutation and ablation caches: `step7_bootstrap_cache_xw_norm_v7.pkl`,
  `step7_permutation_cache_xw_norm_v7.pkl`, `step7_ablation_cache_xw_norm_v7.pkl`
- Consolidated report: `step7_stats_report.txt`
- Balance-refinement trace: `zsh_balance_refinement_log.csv`
- Isolation Forest threshold: `isolation_forest_threshold_report.csv`

## Items that did not reproduce

Two manuscript items had no saved output artifact. Both were re-run; neither
reproduced its published values, for different reasons. Table A1 is resolved —
the manuscript was revised to report what the re-run shows. Table 6 is not, and
the decision taken there is recorded below.

### Table 6 — external validation against GraphSense tags

The published values (200,000-row subsample, 2,559 matches at 1.28% coverage,
ZSH NMI 0.026 / AMI 0.023 against KMeans++ Elkan NMI 0.019 / AMI 0.017) were
computed against a snapshot of 418,575 tagged addresses that is not archived
with this project.

Re-runs against the public `graphsense-tagpacks` repository as of 2026-08-30
(483,296 tagged BTC addresses) are saved as
`step6_graphsense_external_validation.json` in both `outputs_raw/` and
`outputs_balanced/`. These are **different experiments, not reproductions**: the
tag snapshot has changed. Balanced-corpus rows carry no source index, so they
were recovered to their raw source rows by feature hash (81.35% resolved).

| | published | raw corpus | balanced corpus |
|---|---|---|---|
| matched rows | 2,559 | 1,172 | 3,075 |
| coverage | 1.28% | 0.59% | 1.54% |
| ZSH NMI / AMI | 0.026 / 0.023 | 0.071 / 0.061 | 0.234 / 0.230 |
| Elkan NMI / AMI | 0.019 / 0.017 | 0.105 / 0.096 | 0.325 / 0.321 |
| higher alignment | ZSH | Elkan | Elkan |

The balanced corpus reproduces the published coverage closely (1.54% against
1.28%), which indicates the published Table 6 was computed on the balanced
corpus. The direction of the ZSH-versus-Elkan comparison, however, reverses on
both corpora: on this tag snapshot KMeans++ Elkan aligns better with external
entity tags than ZSH does. The published Table 6 cannot be reproduced without
the original tag snapshot, and the reversal is not explained by corpus choice.

### Table A1 — resolved; Appendix A now reports both solvers

The originally published figures (Silhouette 0.0043, 21 of 30 clusters, CHI 1.0)
had no saved artifact and did not reproduce. Re-running the Appendix A protocol
with a converged eigensolver gives Silhouette 0.3703 with all 30 clusters formed.

Appendix A has been rewritten around this: Table A1 now reports **both** solves
side by side, and Table A2 gives the scaling evidence. The manuscript no longer
claims the graph-embedding baseline collapses at scale; it reports that the
baseline's full-scale behaviour is governed by eigensolver convergence on a
disconnected landmark graph, and that on a converged solve the baseline is
competitive with ZSH.

The converged column is reproducible from
`outputs_balanced/stepA1_graph_baseline_fullscale.csv` via `spectral_gpu.py`.
The ARPACK column is retained as published; it cannot be regenerated, because
sklearn's ARPACK solve does not complete on this graph (no result in six hours at
n = 80,000, none in seven minutes at n = 40,000).

Note that the subsample-scale graph-embedding rows in `step7_sota_comparison.csv`
(Silhouette 0.1755 and 0.1843) are a different evaluation and are not the source
of Table A1 — `Step_7_Statistical_Rigor.py` says so in its own comments. Reusing
the stored, 1-NN-propagated `X_spectral.npy` does not reproduce it either (that
shortcut gives Silhouette 0.0952 with 28 of 30 clusters).

The projection step was implemented from the Appendix A description, because the
original full-scale code is not in this repository.
