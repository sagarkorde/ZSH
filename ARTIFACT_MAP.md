# Artifact Map

This file maps manuscript evidence to the scripts and generated artifacts that
produced it. Numbering follows the current manuscript (Tables 1–11 plus Appendix
Table A1, Figures 1–9).

Entries marked **[to confirm]** still need their output file recorded; the
producing script is given where it is known.

## Tables

- `Table 1` core experimental settings
  - source: manuscript values
- `Table 2` wall-clock fitting time per method
  - script: `Step_7_Statistical_Rigor.py` — **[to confirm]** output file
- `Table 3` same-space comparison across 17 methods and weighting variants
  - source: `outputs/step7_sota_comparison.csv`
- `Table 4` ablation study
  - source: `outputs/step7_ablation_study.csv`
- `Table 5` sensitivity of same-space metrics to `s` and `lambda`
  - script: `Step_3_Zeta_Weighting.py` — **[to confirm]** output file
- `Table 6` external validation against GraphSense-tagged addresses
  - **[to confirm]** script and output file
- `Table 7` representative high-support profiles and anomaly rates
  - source: `outputs/RESULTS_SUMMARY.txt`
  - source: `outputs/profile_statistics.csv`
- `Table 8` contextual profiling comparison
  - source: `outputs/step8_contextual_comparison_summary.csv`
- `Table 9` sensitivity of the minority-stratum upsampling target
  - script: `Step_2_Preprocess.py` — **[to confirm]** output file
- `Table 10` balanced-corpus versus raw-corpus headline results
  - **[to confirm]** script and output file
- `Table 11` independent replication on the Elliptic dataset
  - script: `Step_9_Elliptic_Replication.py` — **[to confirm]** output file
- `Table A1` full-scale graph-embedding baseline (Appendix A)
  - **[to confirm]** script and output file

## Figures

- `Figure 1` ZSH system architecture pipeline
  - author-prepared diagram, not a script output
- `Figure 2` top 10 features by zeta weight
  - script: `Step_3_Zeta_Weighting.py` — **[to confirm]** output file
- `Figure 3` ZSH pipeline flowchart
  - author-prepared diagram, not a script output
- `Figure 4` bootstrap confidence intervals
  - source: `outputs/fig9_bootstrap_ci.png`
- `Figure 5` same-space benchmark comparison
  - source: `outputs/fig10_sota_comparison.png`
- `Figure 6` ablation summary
  - source: `outputs/fig11_ablation_study.png`
- `Figure 7` Silhouette against decay exponent `s` and blend weight `lambda`
  - **[to confirm]** script and output file
- `Figure 8` profile feature heatmap
  - source: `outputs/fig3_profile_heatmap.png`
- `Figure 9` contextual profiling comparison
  - source: `outputs/fig12_contextual_comparison.png`

## Supplementary Visuals

Referenced in Section 5.6 of the manuscript but not reproduced there in full.

- UMAP profile map
  - source: `outputs/fig1_cluster_umap.png`
- anomaly overlay
  - source: `outputs/fig2_anomaly_overlay.png`
- temporal profile distribution
  - source: `outputs/fig5_temporal_distribution.png`
- cluster size distribution
  - source: `outputs/fig4_cluster_sizes.png`
- outlier rate by profile
  - source: `outputs/fig6_outlier_by_profile.png`
- radar fingerprints
  - source: `outputs/fig7_radar_fingerprints.png`

## Note on output file names

The `outputs/` file names above predate the current manuscript numbering, so a
file called `fig9_bootstrap_ci.png` corresponds to `Figure 4` in the manuscript.
The names are left unchanged so they continue to match what the pipeline
actually writes.
