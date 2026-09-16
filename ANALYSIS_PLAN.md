# ZSH v2 — Analysis Plan

Status: **written before any confirmatory analysis** (tag `v2-plan`).
Code and configuration are frozen later at tag `v2-frozen`; no test-period,
prospective or Elliptic-test result is computed before that tag exists.
Any change made after `v2-frozen` is logged in `DEVIATIONS.md` with date,
commit and reason.

ZSH = **Z**eta-normalised rank weighting, **S**ize-constrained refinement,
**H**ierarchical (Ward) initialisation. Version 1 of this repository (tag
`v1-submitted`) is kept unchanged under `legacy_v1/` for traceability; none of
its numbers are reused.

---

## 1. Research questions and hypotheses

* **RQ1 — design.** What do rank-power feature weighting and size-constrained
  refinement change, compared with standard clustering on the same data and
  with the same number of clusters?
  * **H1.** With features, initialisation, K and refinement held fixed,
    rank-power weighting (s = 1.5) changes the concentration of annotations
    that are not clustering inputs (L2–L5, §3.3) relative to uniform weighting.
    Two-sided.
  * **H2.** At matched final K, size-constrained refinement changes
    annotation concentration and intrinsic geometry relative to plain K-means.
    Two-sided.
* **RQ2 — reliability.**
  * **H3.** Partitions produced by the full pipeline are reproducible across
    block-bootstrap refits (ARI, AMI, VI, clusterwise Jaccard), reported
    against uniform K-means at the same K and against the same pipeline on
    column-permuted data (no-structure reference).
  * **H4.** Profiles fitted on 2022–2023 transfer to 2024 and to
    Oct 2024–Aug 2026: agreement between transferred and refitted partitions,
    and drift in annotation composition per profile. Runes activity (first
    possible at block 840,000, April 2024) is absent from the development
    period and is analysed as an emergent behaviour (exploratory).
* **RQ3 — meaning.** Profiles concentrate annotations not used for clustering
  above their base rates (primary vs. uniform K-means, Holm-adjusted). The
  count-based "coinjoin-like" rule is checked against an equal-output CoinJoin
  heuristic and against GraphSense coinjoin tags.
* **RQ4 — forensic relevance (Elliptic).**
  * **H5.** On held-out timesteps 35–49, clusters ranked by their illicit rate
    in timesteps 1–34 give test precision above the base rate at fixed
    coverage; ZSH is compared with uniform K-means at the same K.
  * **H6.** Atypicality scores (Isolation Forest, LOF, distance to assigned
    centroid) are not positively associated with illicit status on held-out
    timesteps. Decision rule per score: supported if the upper 95% bound of
    ROC-AUC ≤ 0.50; contradicted if the lower bound > 0.50; otherwise
    inconclusive.

Everything else (sensitivity, balancing, seeding, leave-one-family-out,
partial reproduction of Vlahavas et al. 2024) is secondary.

## 2. Data

| ID | Content | Location | SHA-256 |
|---|---|---|---|
| D1 | Bitcoin transaction sample, 5,884,387 rows (Korde et al., IEEE DataPort, doi:10.21227/bxmt-mn56) | `Dataset.parquet` | `d1c02258ada8afb769b3ee3078eb13333844fb63693d4f74a22e819867ef81ea` |
| D2 | Elliptic Bitcoin dataset (features, classes, edges) | `elliptic_txs_*.csv` | features `fd7f8357…0ff0`, classes `93e2e7b2…9493`, edges `a35053ba…156e` |
| D3 | GraphSense TagPacks address→category map, public repository cloned 30 Aug 2026, 483,296 BTC addresses | `btc_tagmap.json` | `2200d3601d8d7a372c596dad710a6db684281322075ab84dc15877c8ef3bbdd7` |
| D4 | Prospective sample, Oct 2024–Aug 2026 (collected after `v2-frozen`, §2.2) | `fut_*` | recorded at `v2-data-future` |
| D5 | Per-output data for 12,000 D1 transactions (E0, E7) | `api_tx_*` | recorded at `v2-data-audit` / `v2-data-future` |

### 2.1 Splits of D1 (by block time, UTC)

* **DEV**: 13 Jul 2022 ≤ t < 1 Jan 2024 (3,299,616 rows). Used for all fitting.
* **TEST**: 1 Jan 2024 ≤ t < 1 Oct 2024 (2,584,530 rows). Evaluation only.
* 241 rows with t ≥ 1 Oct 2024 are excluded and reported.

### 2.2 Prospective sample D4

Source: mempool.space Esplora REST API (blockstream.info as fallback for
Esplora-compatible endpoints), ≤ 1 request/s, exponential back-off on HTTP
429/5xx, raw JSON cached with checksums.

For each calendar month from Oct 2024 to Aug 2026 (23 months): 800 block
heights drawn uniformly without replacement from the blocks mined in that
month (month boundaries from `/api/v1/mining/blocks/timestamp/:ts`); for each
block one page of up to 25 transactions (`/api/block/:hash/txs/:start`) with
the start index drawn uniformly from the valid multiples of 25. Design weight
per transaction = (pages in block) × (blocks in month / blocks sampled in
month). Clustering analyses are unweighted; composition estimates use the
design weights. Seeds: §6.

### 2.3 Feature equivalence (E0)

2,000 random DEV/TEST txids are fetched from the API and their features
recomputed with the prospective feature code. A feature is used on D4 only if
agreement with D1 is ≥ 99% (exact for counts and booleans; relative error
≤ 1e-6 for values). Failures are reported and the feature is flagged.

## 3. Method (primary pipeline)

### 3.1 Features

Start from the 27 v1 clustering features and remove, in order:

1. the five count-rule flags (`is_consolidation`, `is_distribution`,
   `is_peer_to_peer`, `is_batch_payment`, `is_coinjoin_like`) — they are exact
   functions of `input_count` and `output_count` and are used only as
   annotations (L1);
2. any feature whose most frequent value covers ≥ 99.9% of DEV rows
   (expected: `value_concentration_ratio`, which is 1 by construction);
3. redundancy: iterate the priority list below and keep a feature only if its
   |Spearman ρ| < 0.98 with every feature already kept (ρ on 1,000,000 random
   DEV rows).

Priority: `input_count, output_count, vsize, total_input_value,
total_output_value, fee, fee_rate_sat_per_vbyte, avg_input_value,
avg_output_value, input_output_ratio, weight, size, input_address_count,
output_address_count, input_script_count, total_addresses, value_difference,
fee_rate_sat_per_byte, has_coinbase, has_op_return, rbf_enabled`.

Unit correction: values stored in BTC are converted to satoshis; the stored
`fee_rate_*` columns are in BTC per (v)byte and are recomputed as
fee (sat) / vsize and fee (sat) / size.

### 3.2 Preprocessing

Continuous features: x ↦ sign(x)·log(1+|x|), then robust scaling
(median, IQR) fitted on DEV only; if a feature's IQR is 0 the scale is
1.349 × SD. Binary features stay 0/1. No resampling in the primary analysis.

### 3.3 Annotations (never clustering inputs)

* **L1 count-rule families** (v1 definitions, first-match order: coinbase,
  in>3∧out>3, in=1∧out>5, in>out∧in>2, out>in∧out>2, in=1∧out=1, OP_RETURN,
  RBF). Reported as structural consistency only, with a derivability ceiling
  (depth-3 decision tree from the input features, DEV).
* **L2 input script class** from `input_script_types`: P2PKH, P2SH, P2WPKH,
  P2WSH, P2TR if all inputs share the class, otherwise `mixed`; coinbase
  separate.
* **L3 OP_RETURN protocol** from the OP_RETURN script: Runes (`6a5d…`,
  OP_RETURN OP_13), Omni (`omni` marker as first push), other.
* **L4 entity tags** (D3): a transaction carries category c if any input or
  output address maps to c; evaluated categories with ≥ 200 positives per
  fold: exchange, miner, coinjoin.
* **L5 equal-output CoinJoin (EO-CJ)** (needs per-output values; D4, D5):
  the most frequent output value v\* ≥ 50,000 sat occurs k ≥ 3 times, and
  n_inputs ≥ k.
* **L6 Elliptic illicit label.**

### 3.4 Rank-power weighting

1. Proxy partition: MiniBatchKMeans (K_p = 10, n_init = 10) on scaled DEV.
2. Mutual information `mutual_info_classif` (continuous/binary mask,
   n_neighbors = 3) between each feature and the proxy labels on a
   class-balanced subsample (≤ 20,000 rows per proxy cluster).
3. Ranks r_j by decreasing MI (ties: priority order).
4. w_j = r_j^(−s) / H_d(s), H_d(s) = Σ_{k=1..d} k^(−s), s = 1.5 (fixed a
   priori, not tuned).
5. Metric: d²(x, y) = Σ_j w_j (x_j − y_j)², implemented as X_w = X · diag(√w).

### 3.5 Hierarchical initialisation

MiniBatchKMeans with 160 micro-clusters on X_w (DEV), then size-weighted Ward
agglomeration of the 160 centroids (merge cost n_a n_b/(n_a+n_b)·‖c_a−c_b‖²)
to K₀ = 30 groups; initial centroids are the size-weighted group means.

### 3.6 Fit, size-constrained refinement, inference

1. K-means (Lloyd, n_init = 1 from §3.5, max_iter = 300, tol = 1e-4) on all
   DEV rows of X_w.
2. While some cluster holds > c·N rows and its depth < D: split it with
   K-means++ (n_init = 5) into k = ⌈size/(c·N)⌉ children (fit on ≤ 500,000
   members, assign all members). c = 0.10, D = 3.
3. Final centroids = cluster means; every row is reassigned once to its
   nearest final centroid (empty clusters dropped). The realised maximum
   share is reported.
4. Inference for any new transaction: DEV scaler → DEV weights → nearest
   final centroid.

### 3.7 Profile description

Per profile: share; medians of input/output counts, value (sat), fee rate
(sat/vB), vsize; L2 composition; OP_RETURN share and L3 composition; L1
shares; a descriptor string built from these; a representative transaction
(member nearest the centroid, txid reported).

### 3.8 Atypicality score (separate component)

Isolation Forest (200 trees, 256 samples/tree) fitted on 500,000 DEV rows of
X_w; score = −score_samples. No threshold is part of the method.

## 4. Experiments

| ID | Content | Data | Output |
|---|---|---|---|
| E0 | Data audit (coverage, constants, redundancy, units, flag definitions, script-flag check, duplicates) + API feature equivalence | D1, D5 | T2, T3, A-tables |
| E1 | Primary fit and profile description | DEV | T5, F3, F4 |
| E2 | Method comparison, matched K, identical 200,000-row DEV fit sample, disjoint 200,000-row DEV evaluation sample, TEST by transfer; end-to-end timing with 16 threads | DEV, TEST | T6 |
| E3 | Weighting × refinement (2×2) on full DEV; weighting curves: uniform, MI-direct, rank-power s = 1.5, Laplacian-score rank-power | DEV, TEST | T7 |
| E4 | Stability: 10 seeds; 30 block-bootstrap replicates (DEV blocks with replacement, then 1,000,000 rows); fixed 200,000-row DEV reference set; ZSH, uniform K-means (K\*), weighted K-means without refinement (K\*); 10 replicates on column-permuted DEV; MI-rank Kendall τ | DEV | T8, F6 |
| E5 | Temporal transfer DEV→TEST and DEV→D4; refits on TEST and D4; monthly profile shares; per-profile annotation drift (Jensen–Shannon); Runes emergence (exploratory) | DEV, TEST, D4 | T9, F7 |
| E6 | Annotation concentration (L2–L5), ZSH vs uniform K-means (K\*) vs weighted K-means (K\*) vs uniform+refinement | TEST, D4 | T10, F5 |
| E7 | Count rule vs EO-CJ vs GraphSense coinjoin tag (5,000 flagged + 5,000 unflagged D1 transactions; all of D4) | D5, D4 | T11 |
| E8 | Elliptic, temporal split (fit on all transactions of timesteps 1–34, 165 features; LF-93 as sensitivity); baselines at matched K; supervised random forest as a reference ceiling | D2 | T12, F8 |
| E9 | Atypicality vs illicit status (Elliptic test); atypicality vs L3/L5 on TEST and D4 (exploratory) | D2, TEST, D4 | T13, F9 |
| E10 | Sensitivity on 1,000,000-row DEV samples, evaluated on TEST: s ∈ {0.5, 1, 2, 3}; K₀ ∈ {10, 20, 40, 60}; c ∈ {0.05, 0.15, 0.20, none}; D = 6 at c = 0.10; v1-style upsampled corpus; sample-weighted variant; initialisation: k-means++, semantic seeds, seed–Ward blend (λ = 0.6); leave-one-family-out for the seeded variant (family removed from seeding; for coinbase/OP_RETURN/RBF also from the features, with the whole pipeline refitted) | DEV, TEST | T14, F10 |
| E12 | Partial reproduction of Vlahavas et al. (2024): standardise → PCA (3) → trimmed k-means (α = 0.01) on the 5 of their 8 features available here (inputs, outputs, total amount, OP_RETURN count, fee per byte), at k = 5 and at K\* | DEV, TEST | row in T6 |

Not repeated from v1: the graph-embedding appendix, the label-permutation
test and the Table-6 GraphSense ranking. They are out of scope for the
questions above; the reasons are stated in the article.

### 4.1 Baselines in E2 (all unweighted scaled features, K = K\*)

K-means++ (Lloyd, n_init = 10); MiniBatchKMeans; Gaussian mixture
(diagonal); BIRCH; Ward (30,000-row sample, nearest-centroid extension);
HDBSCAN (50,000-row sample, min_cluster_size = 0.1% of the sample; evaluated
on its own sample only, noise reported as its own group); E12.

## 5. Metrics and statistics

* **Intrinsic geometry** in the common unweighted scaled space (primary) and
  in each method's own space: Silhouette (mean of 5 random 20,000-row draws),
  Davies–Bouldin, Calinski–Harabasz; CH also on de-duplicated rows.
* **Concentration (cross-fitted).** Blocks are split into two folds by block
  height parity. Clusters are ranked on one fold by the Wilson 95% lower bound
  of their annotation rate (clusters with < 100 members in that fold are
  ranked last); the ranking is applied to the other fold, and the two
  directions are averaged. Reported: precision–coverage curve, average
  precision of the ranking, and enrichment (precision / base rate) at 10%,
  25% and 50% coverage.
* **Uncertainty.** Percentile bootstrap over blocks (Bitcoin, B = 1,000) or
  timesteps (Elliptic, B = 1,000). Method differences are paired within
  replicates.
* **Multiplicity.** Holm correction within each hypothesis family (per RQ).
  Monte Carlo p-values use the add-one estimator.
* **Stability.** ARI, AMI, variation of information; clusterwise Jaccard
  (Hennig 2007) with the primary partition; centroid displacement after
  Hungarian matching.
* **Timing.** Wall-clock fit + assignment, same data, 16 threads, same
  machine; hardware and library versions reported.

## 6. Reproducibility

* Environment: Python 3.12, `env/requirements.lock` (uv).
* One configuration file, `configs/zsh_v2.json`; master seed 42, child seeds
  via `numpy.random.SeedSequence(42).spawn`.
* Every table and figure is produced by a script from files in
  `results/`; large arrays live outside the repository and are listed with
  checksums.
* Checkpoints: `v2-plan`, `v2-data-audit`, `v2-frozen`, `v2-data-future`,
  `v2-results`, `v2-figures`, `v2-manuscript`.

---

## 7. Clarifications recorded before the freeze

Made after `v2-plan` and before `v2-frozen`. They come from the data audit and
code tests on DEV data only; no TEST, prospective or Elliptic-test result had
been computed.

1. **§2.1** The last D1 transaction before 1 Oct 2024 is dated 7 Sep 2024, so
   TEST covers 1 Jan – 7 Sep 2024 (36,259 blocks). DEV covers 63,075 blocks.
2. **§2.2** Requests are spread over both hosts at the same time, each host at
   ≤ 1 request/s; a host's interval doubles after HTTP 429 (maximum 8 s), and
   failed paths get a second pass. The timestamp endpoint returns the last
   block mined before a given time, so month *m* covers heights
   (h_m, h_{m+1}].
3. **§3.1** The selection rules keep 12 features: `input_count, output_count,
   vsize, total_input_value, fee, fee_rate_sat_per_vbyte, avg_output_value,
   input_output_ratio, size, fee_rate_sat_per_byte, has_op_return,
   rbf_enabled`. `has_coinbase` is removed by the 99.9% mode rule (0.037% of
   DEV rows); `input_address_count`, `output_address_count`,
   `input_script_count` and `total_addresses` are removed by the same rule
   (they are lengths of lists that hold one joined string);
   `total_output_value`, `avg_input_value`, `weight`, `value_difference` by the
   redundancy rule. The stored fee-rate columns are in BTC per (v)byte and are
   recomputed in sat/(v)B.
4. **§3.3** The D1 columns `has_p2pk … has_taproot`, `is_self_transfer` and
   `address_reuse` are false/zero in every row and are not used. GraphSense
   coinjoin tags match no D1 transaction, so L4 on D1 is effectively the
   exchange (and, where the minimum is met, miner) category.
   Evaluated targets: L2 coinbase, P2PKH, P2SH, P2WPKH, P2WSH, P2TR, mixed;
   L3 runes, omni, other OP_RETURN; L4 exchange, miner, coinjoin; L5 EO-CJ
   (prospective sample only). Primary metric for hypothesis tests: AP lift
   (average precision / base rate), Holm-adjusted across targets; enrichment
   at fixed coverage is descriptive.
5. **E2** BIRCH threshold 0.5; HDBSCAN min_cluster_size = 0.1% of its sample;
   GMM reg_covar tried in the order 1e-4, 1e-3, 1e-2 (first that fits is
   used and reported); trimmed k-means with 10 k-means++ starts and at most
   100 iterations.
6. **E3** Arms: A1 primary; A2 uniform + refinement (K_u); A3 rank-power,
   no refinement, K\*; A4 uniform, no refinement, K\*; A5 uniform, no
   refinement, K_u; A6 MI-direct + refinement; A7 Laplacian rank-power +
   refinement; A9 uniform K-means++ (n_init = 10), K\* (the standard baseline
   used in E4–E6). H1 primary contrast A3 vs A4, secondary A1 vs A2. H2
   primary contrast A1 vs A3, secondary A2 vs A5. All arms use the primary
   seed.
7. **E4** Reference partitions for the baselines are their full-DEV fits (A9,
   A3). Centroid displacement is measured in the unweighted scaled space.
8. **E8** Elliptic uses the §3.2 preprocessing with all 165 features treated as
   continuous. Clusters need ≥ 20 labelled training members to be ranked
   ahead of others. The per-timestep analysis uses the smallest prefix of
   training-ranked clusters that covers 25% of training illicit transactions.
9. **E9** Bitcoin atypicality analyses are exploratory, block bootstrap
   B = 200.
10. **E10** v1-style upsampling strata: (in>3 ∧ out>3, in=1 ∧ out>5,
    OP_RETURN); strata below 30% of the largest are upsampled with
    replacement. The sample-weighted variant uses weights
    max(1, target/stratum size) in scaling, proxy partition, MI sampling,
    initialisation, K-means and refinement.
