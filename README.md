# ZSH Blockchain Transaction Profiling

This repository accompanies the paper:

`ZSH: A Zeta-Weighted Hybrid Semantic Clustering Framework for Blockchain Transaction Profiling and Anomaly-Aware Forensic Analysis`

It provides the end-to-end experiment pipeline, reproducibility instructions, and the artifact mapping used to generate the main tables and figures reported in the manuscript.

The latest packaged LaTeX manuscript project is available at `downloads/ZSH_LaTeX_Project_Latest.zip`.

## What This Repository Contains

- `RUN_PIPELINE.py`
- `Step_1_Load_and_Explore.py`
- `Step_2_Preprocess.py`
- `Step_3_Zeta_Weighting.py`
- `Step_4_UMAP_Embedding.py`
- `Step_5_ZSH_Clustering.py`
- `Step_6_Profile_and_Visualize.py`
- `Step_7_Statistical_Rigor.py`
- `Step_8_Contextual_Profiling_Comparison.py`
- `outputs/` generated artifacts used by the paper

## Environment

Recommended environment:

- Python 3.10 or newer
- 64 GB RAM recommended
- Windows workstation used in the paper: Intel Core i9, 64 GB RAM, NVIDIA RTX 4060 8 GB

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place the transaction dataset at the repository root as:

```text
Dataset.parquet
```

If the dataset cannot be shared publicly, provide:

- a schema description
- extraction instructions
- a synthetic or reduced sample for sanity checks
- checksums for the internal experiment dataset

## Reproducing the Paper

Run the main pipeline:

```bash
python RUN_PIPELINE.py
```

Run the statistical validation:

```bash
python Step_7_Statistical_Rigor.py
```

Run the contextual profiling comparison:

```bash
python Step_8_Contextual_Profiling_Comparison.py
```

## Main Paper Artifacts

The following files correspond directly to the manuscript:

- `outputs/step7_stats_report.txt`
- `outputs/step7_sota_comparison.csv`
- `outputs/step7_ablation_study.csv`
- `outputs/step8_contextual_comparison_summary.csv`
- `outputs/fig9_bootstrap_ci.png`
- `outputs/fig10_sota_comparison.png`
- `outputs/fig11_ablation_study.png`
- `outputs/fig12_contextual_comparison.png`
- `outputs/profile_statistics.csv`
- `outputs/RESULTS_SUMMARY.txt`

See `ARTIFACT_MAP.md` for the exact mapping from manuscript items to output files.

## Important Claim Boundary

This repository supports a conditional claim:

- `KMeans++ Elkan` is stronger on intrinsic Euclidean clustering geometry in the corrected same-space benchmark.
- `ZSH` is stronger for semantic blockchain profiling, minority-profile recoverability, and anomaly-aware forensic interpretation.

The paper should not claim universal geometry-only superiority over KMeans.

## Citation

Use `CITATION.cff` for repository citation metadata after the repository is made public.
