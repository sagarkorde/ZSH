# ZSH IEEE Access Figure and Table Placement Notes

This note is intended for final assembly in Word or LaTeX. The main manuscript source is:

- `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/ZSH_IEEE_Access_Submission.md`

Use the following figures and tables in the main paper. The ordering below is chosen to support a reviewer-safe storyline: validate the cluster structure first, report the geometry benchmark honestly, then show why ZSH is stronger for semantic profiling.

## Main-Manuscript Tables

### Table I. Core experimental settings

- Placement: Section IV-E, immediately after the paragraph introducing the experimental settings.
- Source: already embedded in the manuscript.
- Purpose: gives the full dataset and protocol snapshot in one place.
- Suggested caption: `Core experimental settings used in the final ZSH study.`

### Table II. Same-space benchmark comparison

- Placement: Section V-B, immediately after the paragraph introducing the corrected same-space comparison.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/step7_sota_comparison.csv`
- Purpose: makes the KMeans++ Elkan geometry advantage explicit.
- Suggested caption: `Same-space comparison in standardized zeta-weighted feature space X_w_norm. All methods are trained and evaluated in the same space.`

### Table III. Ablation study

- Placement: Section V-C, immediately after the paragraph introducing the ablation study.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/step7_ablation_study.csv`
- Purpose: isolates the effects of weighting, geometry refinement, and anomaly separation.
- Suggested caption: `Ablation study of ZSH components. Condition C isolates the geometry-refinement branch and should not be interpreted as the final named ZSH profiling result.`

### Table IV. Contextual profiling comparison

- Placement: Section V-E, immediately after the paragraph introducing the contextual comparison.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/step8_contextual_comparison_summary.csv`
- Purpose: supports the task-aligned claim that ZSH is better for semantic profiling than KMeans.
- Suggested caption: `Contextual profiling comparison between ZSH and KMeans++ Elkan over five repeated 120,000-row samples. Metrics are interpreted as task-alignment evidence rather than external ground truth.`

### Table V. Representative ZSH profiles

- Placement: Section V-D, after the first paragraph of profile-level findings.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/RESULTS_SUMMARY.txt`
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/profile_statistics.csv`
- Purpose: shows the most prevalent semantic profiles and their anomaly rates.
- Suggested caption: `Representative high-support ZSH semantic profiles and their anomaly rates.`

## Main-Manuscript Figures

### Fig. 1. Global profile map

- Placement: Section V-D, after the sentence discussing the global arrangement of profiles.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig1_cluster_umap.png`
- Purpose: visual overview of profile separation.
- Reviewer note: make clear in the caption that UMAP is for visualization only.
- Suggested caption: `UMAP visualization of the 30 ZSH transaction profiles. The projection is used for visualization only; all clustering metrics are computed in the standardized zeta-weighted feature space rather than in UMAP space.`

### Fig. 2. Anomaly overlay

- Placement: Section V-D, immediately after Fig. 1 or on the following page if space is tight.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig2_anomaly_overlay.png`
- Purpose: shows how anomaly flags sit on top of the semantic profile layout.
- Suggested caption: `UMAP view of the learned transaction profiles with anomaly flags overlaid. The figure illustrates the profile-preserving anomaly design of ZSH.`

### Fig. 3. Semantic heatmap

- Placement: Section V-D, after the paragraph discussing interpretable profile families.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig3_profile_heatmap.png`
- Purpose: helps readers see how profile groups differ behaviorally.
- Suggested caption: `Heatmap of representative profile-level feature patterns across the learned ZSH transaction families.`

### Fig. 5. Temporal distribution

- Placement: Section V-D, after the sentence discussing dynamic profiling across time.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig5_temporal_distribution.png`
- Purpose: supports the dynamic-profiling framing.
- Suggested caption: `Temporal distribution of the major ZSH profiles across the covered blockchain period.`

### Fig. 9. Bootstrap confidence intervals

- Placement: Section V-A, immediately after the paragraph reporting bootstrap and permutation validity.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig9_bootstrap_ci.png`
- Purpose: visual evidence that the learned structure is stable under resampling.
- Suggested caption: `Bootstrap confidence intervals for the intrinsic clustering metrics of ZSH in the corrected same-space evaluation protocol.`

### Fig. 10. Geometry benchmark comparison

- Placement: Section V-B, directly after Table II.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig10_sota_comparison.png`
- Purpose: highlights that KMeans++ Elkan is the strongest geometry-first baseline.
- Suggested caption: `Same-space comparison between ZSH and classical baselines in X_w_norm. KMeans++ Elkan attains the strongest intrinsic geometry, while ZSH remains competitive and provides semantic profile labels plus anomaly outputs.`

### Fig. 11. Ablation summary

- Placement: Section V-C, directly after Table III.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig11_ablation_study.png`
- Purpose: quickly communicates what the ablation contributes.
- Suggested caption: `Ablation study showing the contribution of zeta weighting, geometry refinement, and anomaly separation.`

### Fig. 12. Contextual profiling comparison

- Placement: Section V-E, directly after Table IV.
- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig12_contextual_comparison.png`
- Purpose: shows the tradeoff between geometry-first and profiling-first evaluation.
- Suggested caption: `Contextual profiling comparison showing that ZSH is stronger on semantic purity, entropy, and minority-profile recoverability, whereas KMeans++ Elkan remains stronger on some global-information alignment measures.`

## Recommended Supplementary or Appendix Figures

Use these only if page budget permits, or move them to supplementary material.

### Fig. S1. Cluster size distribution

- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig4_cluster_sizes.png`
- Suggested use: appendix figure to show balance across the 30 profiles.

### Fig. S2. Outlier rate by profile

- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig6_outlier_by_profile.png`
- Suggested use: appendix figure to support the anomaly micro-profile discussion.

### Fig. S3. Radar fingerprints

- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig7_radar_fingerprints.png`
- Suggested use: appendix figure if you want a more visual behavioral interpretation section.

### Fig. S4. Legacy metric comparison

- Source: `C:/Users/sagar/Desktop/Q2 Paper 22326/outputs/fig8_metrics_comparison.png`
- Suggested use: supplementary only. The main paper should prioritize the corrected same-space comparisons in Fig. 10 and Fig. 12.

## Word Assembly Tips

- Use the manuscript file as the text source and insert each figure or table at the placement point above.
- Keep Table II, Fig. 10, Table IV, and Fig. 12 close together in the Results section. These four items carry the core reviewer-facing argument.
- If the paper exceeds the target page length, move Fig. 2 or Fig. 3 to supplementary material before removing Fig. 10 or Fig. 12.
- Keep the sentence that UMAP is visualization-only wherever Fig. 1 or Fig. 2 appears.

## LaTeX Assembly Tips

- Use `figure*` for wide comparison charts if the IEEE Access template compresses them too aggressively.
- Keep benchmark tables in standard `table` environments unless column width becomes too tight.
- Preserve the same numbering used in the manuscript so that the results narrative and captions stay aligned across Word and LaTeX versions.
