# Draft — Data, Method, Experimental Design (method parts)

<!-- Working draft. Numbers marked {…} are filled from results files. -->

## 3. Data

### 3.1. Bitcoin transaction sample

We use a published sample of Bitcoin transactions [@Korde2026] drawn from blocks
744,837 to 903,456. Each record describes one transaction: input and output
counts, input and output values, fee, serialised size, virtual size and weight,
flags for coinbase inputs, OP_RETURN outputs and replace-by-fee signalling
[@BIP125],
the input and output addresses, the script type of each input and output, and
the OP_RETURN script. The sample contains 5,884,387 transactions from 99,341
blocks, on average 59 per block.

Although the block range reaches mid-2025, almost all records fall between
13 July 2022 and 7 September 2024; only 241 transactions are dated later. We
therefore split the sample by block time into a development period
(13 July 2022 – 31 December 2023; 3,299,616 transactions in 63,075 blocks) and
a test period (1 January – 7 September 2024; 2,584,530 transactions in 36,259
blocks), and set the 241 later records aside (Table {T_data}, Figure {F_design}). Every model in this study is
fitted on development data only. Test-period data are used only through
models that were fitted earlier.

### 3.2. Prospective sample

To test whether profiles fitted on 2022–2023 data still describe later
activity, we collected a new sample covering October 2024 to August 2026 after
the analysis code had been frozen. For each of the 23 calendar months we drew
800 block heights at random without replacement and, from each block, one page
of up to 25 consecutive transactions starting at a random position. Data were
retrieved from the public Esplora interfaces of mempool.space and
blockstream.info, at a rate each host tolerated (at most one request per
second per host, and less when a host asked for fewer); for part of the
collection only one of the two interfaces was reachable. Every response was
stored, so the sample can be rebuilt without contacting the interfaces again. Each transaction carries a design weight equal to the
number of pages in its block multiplied by the ratio of blocks mined to blocks
sampled in its month; the weights are used only when we estimate population
shares. The sample contains 456,292 transactions from 18,400
blocks.

The prospective records are converted to the same features with the same code
path. To check that this conversion reproduces the published sample, we
retrieved 2,000 randomly chosen transactions of the development and test
periods from the same interfaces and compared the two versions feature by
feature. All 2,000 transactions agreed on every compared quantity: counts,
sizes, flags, script class, OP_RETURN protocol and entity tags exactly, and
values, fees, ratios and fee rates within a relative error of 6 × 10⁻⁸, which
is the precision of the published single-precision values.

### 3.3. Data audit and feature set

The published sample has 27 derived features. An audit of the development
period showed several problems that change their meaning:

* Five behavioural flags — consolidation, distribution, peer-to-peer, batch
  payment and "coinjoin-like" — are fixed rules on the input and output counts
  (Table {T_rules}). For example, a transaction is "coinjoin-like" whenever it
  has more than three inputs and more than three outputs; no test of equal
  output values is involved. Because the counts are themselves features, a
  depth-3 decision tree recovers each rule from the other features with at
  least 99.6% accuracy.
* The fee-rate columns are stored in bitcoin per byte rather than satoshi per
  byte, which leaves them almost constant after a logarithmic transform. We
  recompute both rates from the fee in satoshis.
* The value-concentration ratio equals one for every transaction in the sample
  except the 940 whose output value is zero, and `value_difference` equals the
  fee in 99.97% of all rows (99.96% of development rows); both are therefore
  uninformative. The address and script counts are lengths of lists that hold a
  single joined string, so they are 1 for almost every transaction.
* The stored script-type flags, the self-transfer flag and the address-reuse
  count are zero in every row.

We fixed the feature set with three rules written down before any evaluation.
First, the five count rules are removed from the inputs and kept only as
annotations. Second, any feature whose most frequent value covers at least
99.9% of development rows is removed. Third, candidates are visited in a fixed
priority order and a feature is kept only if its absolute Spearman correlation
with every kept feature is below 0.98. The rules keep twelve features: input
and output counts, virtual size, serialised size, total input value, fee, fee
rate per virtual byte and per byte, average output value, the ratio of input
to output value, and the OP_RETURN and replace-by-fee flags (Table {T_features}).
The coinbase flag is removed by the second rule because only 0.037% of
development transactions are coinbase transactions.

### 3.4. Annotations that are not clustering inputs

To ask what the clusters mean, we need descriptions of transactions that the
clustering never sees. We use five kinds (Table {T_annotations}); we call L2–L5
the non-input annotations. They are not statistically independent of the
inputs (script types, for example, affect transaction size), but none of them
is an input or is defined from the inputs.

* **Count rules (L1).** The five rules above, together with coinbase,
  OP_RETURN and replace-by-fee, combined into eight mutually exclusive families
  and an unlabelled rest by a fixed first-match order (coinbase first,
  replace-by-fee last), as in the
  first version of this work. They are exact functions of the inputs, so we
  report them only as a check of structural consistency.
* **Input script class (L2).** From the script type of every input:
  pay-to-public-key-hash (P2PKH), pay-to-script-hash (P2SH), pay-to-witness-
  public-key-hash (P2WPKH), pay-to-witness-script-hash (P2WSH), pay-to-taproot
  (P2TR), `mixed` when inputs differ, and coinbase.
* **OP_RETURN protocol (L3).** Runes, when the OP_RETURN script starts with
  OP_RETURN OP_13 [@OrdRunes]; Omni, when the first data push starts with the
  ASCII marker "omni" [@OmniSpec]; and other OP_RETURN use. Only the presence of
  an OP_RETURN output is a clustering input; its protocol is not.
* **Entity tags (L4).** A transaction receives an entity category when any of
  its addresses appears in the GraphSense TagPacks [@Haslhofer2021] snapshot we
  archived on 30 August 2026 (483,296 Bitcoin addresses). Only the exchange
  category has enough matches in the published sample to be evaluated.
* **Equal-output CoinJoin (L5).** CoinJoin transactions combine inputs from
  several users and create several outputs of the same value [@CoinJoinPriv2024]. We mark a transaction when its most frequent output
  value is at least 50,000 sat, occurs at least three times, and the
  transaction has at least as many inputs as equal outputs. The rule needs
  individual output values, which are available only for the prospective
  sample and for API-retrieved records.

### 3.5. Elliptic data

For external labels we use the Elliptic data set [@Weber2019]: 203,769
transactions in 49 time steps with 165 anonymised features, of which 4,545 are
labelled illicit and 42,019 licit. We follow the temporal split of the original
study: time steps 1–34 for fitting and 35–49 for testing. All transactions of
the training steps, labelled or not, are used to fit the clustering; labels
are used only to rank clusters and to evaluate them. Recent work reports that
the construction of the Elliptic features was not disclosed and that
information leaks between commonly used splits [@Safar2026]; we return to this
in Section 7.8.

### 3.6. External labels used only for validation

Two public data sets are used as labels, never as inputs, and were added after the
freeze in response to the reviews.

* **Actors of the Elliptic transactions.** Elliptic++ [@Elmougy2023] annotates the
  same 203,769 transactions with the wallet addresses that appear in them
  (822,942 addresses, 1,268,260 transaction–address pairs). We use them to ask
  whether the Elliptic evaluation can be made disjoint in its actors as well as in
  time (Section 5.3).
* **CoinJoin transactions with a known coordinator.** A public list of Wasabi 2.x
  CoinJoin transactions, each attributed to the coordinator that created it, is
  published with the CoinJoin measurement work of Svenda et al. [@Svenda2026]; the
  copy we used holds 42,241 transactions from eleven coordinators. It gives a
  source-level label for the family that the count rule tries to capture
  (Section 6.4). It covers coordinators that were active after mid-2024, so it has
  no overlap with our development period.

## 4. Method

### 4.1. Overview

ZSH assigns each transaction to a behavioural profile in four steps
(Figure {F_pipeline}): (i) transform and scale the features; (ii) weight the
features by the rank of their relevance; (iii) initialise K-means from a
hierarchical merge of micro-clusters and fit it; (iv) split clusters that hold
more than a set share of the data. New transactions are assigned to the
nearest final centroid. The name refers to these components: zeta-normalised
rank weighting, size-constrained refinement and hierarchical initialisation.

### 4.2. Preprocessing

Monetary values are expressed in satoshis. Each continuous feature x is
mapped to $\operatorname{sign}(x)\log(1+|x|)$ and then centred on its
development-period median and divided by its interquartile range; if the
interquartile range is zero, 1.349 times the standard deviation is used
instead. Binary features are left as 0/1. The resulting matrix is
$X \in \mathbb{R}^{n \times d}$ with $d = 12$.

### 4.3. Rank-power feature weights

Feature relevance is judged against a proxy partition: mini-batch K-means with
$K_p = 10$ clusters on $X$. For each feature $j$ we estimate the mutual information
between $x_j$ and the proxy labels with the nearest-neighbour estimator for a
discrete target [@Ross2014], on a sample with at most 20,000
transactions per proxy cluster. Features are ranked by decreasing mutual
information ($r_j = 1$ for the most informative) and weighted by

$$w_j = \frac{r_j^{-s}}{H_d(s)}, \qquad H_d(s) = \sum_{k=1}^{d} k^{-s},$$ {#eq:weights}

with $s = 1.5$. $H_d(s)$ is the generalised harmonic number, the $d$-term partial
sum of the Riemann zeta function $\zeta(s)$; it only makes the weights sum to
one. The weights define the distance

$$d_w(\mathbf{x}, \mathbf{y})^2 = \sum_{j=1}^{d} w_j \, (x_j - y_j)^2,$$ {#eq:distance}

which we implement by multiplying column $j$ by $\sqrt{w_j}$ and running
Euclidean K-means on the result, $X_w$. With $s = 1.5$ and $d = 12$ the weights
fall from 0.489 for the top-ranked feature to 0.012 for the last.

The proxy partition is itself a K-means solution on the same features, so the
ranking favours the directions that already dominate an unweighted partition.
Equation \eqref{weights} therefore sharpens existing structure rather than discovering new
structure; we test what this does in Section 6.2.

### 4.4. Hierarchical initialisation

Mini-batch K-means with 160 clusters is fitted to $X_w$. The 160 centroids are
merged by Ward's criterion [@Ward1963] with their sizes as weights: the cost of
merging groups $a$ and $b$ is

$$\Delta(a, b) = \frac{n_a n_b}{n_a + n_b} \, \lVert \mathbf{c}_a - \mathbf{c}_b \rVert^2,$$ {#eq:ward}

where $n_a$ is the number of transactions in group $a$ and $\mathbf{c}_a$ its
mean. Merging stops at
$K_0 = 30$ groups, and the size-weighted group means are the initial centroids.

### 4.5. Fitting and size-constrained refinement

Lloyd's algorithm [@Lloyd1982] is run from these centroids on all development
transactions (at most 300 iterations, tolerance 10⁻⁴). A single cluster can
still absorb a large share of transactions, which makes it useless for
triage. We therefore split any cluster that holds more than $c = 10\%$ of the
transactions into $\lceil \mathrm{size}/(cN) \rceil$ parts with K-means++ [@Arthur2007], and repeat for
the parts up to depth $D = 3$. Refinement is a heuristic: when depth $D$ is
reached a cluster may still exceed $c$, and we report the largest share that
remains. Finally the centroids are recomputed as cluster means and every
transaction is reassigned once to its nearest centroid, so that training and
new transactions are assigned by the same rule. Algorithm 1 gives the steps.

Size constraints are usually built into the clustering objective [@BalancedClust2026]; recursive splitting is cruder but scales to millions of rows
and leaves every cluster a Voronoi cell of the final centroids.

| **Algorithm 1.** ZSH: fitting and inference |
|:----------------------------------------------------------------------------------------------|
| **Input:** $N$ development transactions; decay $s$, initial clusters $K_0$, size cap $c$, depth $D$. |
| 1: Transform and scale the features (Section 4.2). |
| 2: Fit the proxy partition, estimate mutual information, compute $w$ by \eqref{weights} and form $X_w$. |
| 3: Fit 160 micro-clusters on $X_w$, merge them by size-weighted Ward to $K_0$ groups, and take the group means as $C_0$. |
| 4: Run Lloyd's algorithm from $C_0$ on $X_w$. |
| 5: **while** a cluster holds more than $cN$ rows and its depth is below $D$ **do** split it into $\lceil \mathrm{size}/(cN) \rceil$ parts with K-means++. |
| 6: Set the final centroids $C$ to the cluster means and reassign every row to its nearest centroid. |
| **Output:** scaler, weights $w$, centroids $C$. |
| **Inference:** scale a new transaction with the development scaler, weight it with $w$, and assign it to the nearest centroid in $C$. |

The cost is dominated by the K-means passes, $O(nKd)$ per iteration; mutual
information is estimated on a sample of at most 200,000 rows.

### 4.6. Profile description

Each profile is described from its own members: its share, the medians of
input and output counts, value, fee rate and virtual size, its input script
mix, its OP_RETURN and Runes shares, its exchange-tag share, the share of each
count rule, and the transaction closest to the centroid. We do not give
profiles behavioural names such as "mixer"; the descriptors state what the
members have in common.

### 4.7. Atypicality score

As a separate component we fit an Isolation Forest [@Liu2008] (200 trees, 256
samples per tree) to 500,000 development transactions in $X_w$ and use the
negative path-length score as an atypicality measure. The score has no
built-in threshold. Section 6.5 tests whether it carries information about
illicit activity.

## 5. Experimental design

### 5.1. Research questions

Six hypotheses make the research questions of Section 1 testable:

* **H1.** With the features, initialisation, K and refinement held fixed,
  rank-power weighting changes the concentration of annotations that are not
  inputs, compared with uniform weights (two-sided).
* **H2.** At the same final K, size-constrained refinement changes annotation
  concentration and geometry, compared with the same model without refinement
  (two-sided).
* **H3.** Refits of the whole pipeline on resampled development data
  reproduce the partition, judged against K-means++ at the same K and against
  the pipeline on data without joint structure.
* **H4.** Profiles fitted on 2022–2023 still describe the transactions of the
  test period and of the prospective sample.
* **H5.** On Elliptic time steps 35–49, clusters ranked by their illicit rate
  in steps 1–34 reach a precision above the base rate at fixed coverage. Tested
  as a frozen reanalysis after prior exposure to the Elliptic outcomes, not as a
  blind confirmatory test.
* **H6.** Atypicality scores are not positively associated with illicit
  status on Elliptic time steps 35–49. For each score, H6 is supported when the
  upper 95% bound of the ROC-AUC is at most 0.5, contradicted when the lower
  bound exceeds 0.5, and inconclusive otherwise. Like H5, this is a frozen
  reanalysis of data we had already examined, and the negative association it
  records was known to us in outline before the plan was written.

For RQ3 the claim tested is that every non-input annotation has an AP lift
whose lower 95% bound exceeds one. Everything else, including the sensitivity
analysis and the Runes and CoinJoin analyses, is secondary or exploratory.

The hypotheses, splits, metrics and settings were written down and the code was
frozen before any test-period, prospective or Elliptic-test result was
computed [repository tags `v2-plan`, `v2-frozen`]. Clarifications made before
the freeze and all later deviations are listed in the repository.

Ten analyses were specified in that plan. Fourteen more were added afterwards, each
for a reason recorded with its commit in the deviations log, and each labelled where
it appears; Table {T_prespec} lists all of them and says which is which. Two of the
later analyses overturned conclusions drawn from the frozen ones, and we report the
earlier reading as the one the new evidence replaced rather than quietly removing it.
The freeze protects the evaluation from being tuned to its own result; it does not
make the design blind, and Section 7.8 says what that leaves open.

One qualification is important enough to state here, because it changes how parts
of this evaluation should be read. The first submitted version of this work had
already analysed the complete labelled Elliptic data set and the same Bitcoin
corpus before the revised plan was written and the code frozen. The freeze
therefore came *after* exposure to those outcomes. Accordingly we distinguish
three tiers of evidence, and use the corresponding terms throughout:

* **Frozen reanalysis after prior exposure.** H5 and H6, and every Elliptic
  result in Section 6.5, together with all analyses of the 2022–2024 Bitcoin
  corpus. The hypotheses and the code were fixed in advance of the reported
  runs, which rules out tuning the analysis to its result, but the outcomes were
  not unknown to us when the plan was written. These are *not* blind
  confirmatory tests and we do not present them as such.
* **Prospective evidence.** The 2024–2026 sample, which did not exist when the
  plan was frozen and whose blocks had not been mined. Only these analyses are
  described as prospective.
* **Exploratory.** The analyses added after the freeze, labelled where they
  appear and listed in Table {T_prespec}.

### 5.2. Baselines and matched comparisons

On a 200,000-row development sample, ZSH and the following methods are fitted
with the same preprocessing and the same number of clusters K as ZSH obtains
on that sample: K-means++ [@Arthur2007], mini-batch K-means, a diagonal Gaussian
mixture, BIRCH [@Zhang1996], and Ward clustering [@Ward1963] on 30,000 rows with
nearest-centroid assignment. HDBSCAN [@Campello2013] chooses its own number of
clusters and cannot assign new points, so it is evaluated only on its 50,000-row
fitting sample, with the noise share reported. We also run a partial
reproduction of Vlahavas et al. [@Vlahavas2024] — standardisation, three
principal components and trimmed k-means with 1% trimming [@CuestaAlbertos1997]
— on the five of their eight features available in our data, at their k = 5
and at K. Timing covers fitting and assignment on the same data with 16
threads.

To separate the two ZSH components we fit, on all development data, a 2×2
design: weights (rank-power or uniform) crossed with refinement (on or off).
Arms without refinement use K equal to the number of clusters of the refined
arm they are compared with. We also compare other weight curves: weights
proportional to mutual information, and rank-power weights from the
unsupervised Laplacian score [@He2005].

### 5.3. Measures

This section gives the measures that carry the main results. The secondary
analyses are described in the same detail in Section S1 of the Supplementary
Materials.

**Geometry.** Silhouette [@Rousseeuw1987] (mean of five random 20,000-row
draws), Davies–Bouldin [@DaviesBouldin1979] and Calinski–Harabasz [@Calinski1974]
indices on 200,000 development rows, computed in one common unweighted space
for all methods and, separately, in each method's own space.

**Concentration.** For an annotation with base rate π, a useful profiling
method places most positive transactions in few clusters with a high positive
rate. We rank clusters by the lower 95% Wilson bound [@Wilson1927] of their
positive rate on one half of the data and read precision and coverage on the
other half. The halves are defined by block-height parity and the two
directions are averaged, so the clusters are never ranked and scored on the
same transactions. From the resulting precision–coverage curve we report the
average precision divided by π (AP lift) and the precision divided by π at 10%,
25% and 50% coverage (enrichment). The AP lift has a ceiling: a partition that
separated the annotation perfectly would reach an average precision of one and
therefore a lift of $1/\pi$. We also report how much of that ceiling is
attained, which is the average precision itself, because it makes annotations
with different base rates comparable: a lift of 23 out of a possible 29 and a
lift of 21 out of a possible 299 describe very different partitions. Clusters with fewer than 100 members in the
ranking half are ranked last. Annotations need at least 200 positives in each
half; in the test period this leaves eleven non-input annotations (the seven
input classes, the three OP_RETURN protocols and the exchange tag), and the
equal-output CoinJoin rule is added in the prospective sample. For Elliptic, clusters are ranked on the labelled transactions of time
steps 1–34 and scored on those of time steps 35–49.

**Stability.** Agreement between partitions is measured by the adjusted Rand
index (ARI) [@Hubert1985], adjusted mutual information (AMI) [@Vinh2010] and
variation of information [@Meila2007] on a fixed set of 200,000 development
transactions, and per cluster by the best Jaccard similarity with a cluster of
the other partition [@Hennig2007]. Replicates are (i) ten refits with different
seeds on one million transactions, (ii) thirty refits on block-bootstrap
resamples [@Kunsch1989] of the development period, each subsampled to one million
transactions, and (iii) ten refits on data whose columns were permuted
independently, which removes all joint structure. Every replicate reruns the
whole pipeline, including the feature weights [@StabilitySel2025].

**Transfer.** The primary model and K-means++ at the same K assign the
test-period and prospective transactions without refitting, and are also
refitted on each later period; transferred and refitted partitions are compared
with the agreement measures above. For every profile we compare its development
and later members through the Jensen–Shannon distance [@Lin1991] between their
input-script mixes and between their count-rule mixes, and through the medians
of the main features. Monthly profile shares show how the mix of activity moves;
shares in the prospective sample use the design weights.

**Weighting convention for the prospective sample.** Because that sample was
drawn with unequal inclusion probabilities, every quantity reported for it is one
of two kinds, and we say which throughout. *Design-weighted*, and therefore an
estimate for the sampled population: annotation prevalence, profile shares, and
the concentration measures (average precision, AP lift, enrichment and precision
at fixed coverage), whose intervals come from a month-stratified block bootstrap.
*Unweighted*, and therefore a property of the sampled transactions themselves:
target eligibility, which counts sampled positives so that the weighted and
unweighted analyses cover the same annotations, and the partition-agreement
measures (ARI, AMI, VI and Jaccard similarities), which describe the agreement of
two labellings of the sampled rows rather than any population total. Ranking a
cluster on the calibration half uses Kish effective counts
$(\sum w)^2 / \sum w^2$, so that a cluster carried by few sampled transactions
with large weights is not credited with spurious precision. With equal weights
every quantity reduces exactly to its unweighted form. Unweighted,
sample-specific versions of the prospective concentration results are reported in
the supplementary file as a sensitivity analysis. The feature ranking of
each refit is compared with the development ranking by Kendall's τ, an
exploratory comparison added after the freeze.

**Sensitivity.** On a fixed sample of 1,000,000 development transactions we
refit the pipeline with one setting changed at a time: $s \in \{0.5, 1, 2, 3\}$;
$K_0 \in \{10, 20, 40, 60\}$; $c \in \{5\%, 15\%, 20\%\}$ or no refinement; depth 6;
the upsampled corpus of the first version of this work and the equivalent sample
weights; the size of the proxy partition, $K_p \in \{5, 20, 50\}$ instead of ten;
and three initialisations: K-means++ with ten starts, semantic seeds, and a blend
of the seeds with the hierarchical centroids. The semantic seeds share the $K_0$
initial centroids among the count-rule families (L1) in proportion to their size,
with at least one centroid for every family of at least 500 development
transactions, and place them by mini-batch K-means within each family in $X_w$;
this is the seeding of the first version of this work. The blend pairs each seed
$\mathbf{c}^{\mathrm{seed}}_k$ with one hierarchical centroid
$\mathbf{c}^{\mathrm{hier}}_{\pi(k)}$ by a minimum-cost assignment of squared
distances [@Kuhn1955] and starts K-means from
$\lambda\,\mathbf{c}^{\mathrm{seed}}_k + (1-\lambda)\,\mathbf{c}^{\mathrm{hier}}_{\pi(k)}$
with $\lambda = 0.6$. All variants are evaluated on the test period against the
reference variant.

**The remaining analyses.** Six further analyses are defined in Section S1. The
"coinjoin-like" count rule is checked against the equal-output rule on a
stratified sample of 10,000 transactions whose individual output values were
retrieved from the Esplora interfaces, and against an external list of Wasabi 2.x
CoinJoin transactions. On Elliptic, five methods at matched K are compared with a
random forest [@Breiman2001] trained on the same labels, and the cluster ranking
is repeated on training transactions that share no address with the evaluation
period. Atypicality is measured by the Isolation Forest of Section 4.7, the local
outlier factor [@Breunig2000] and the distance to the assigned centroid, each
scored by ROC-AUC against the illicit label. Five analyses were added after the
freeze: an upper bound for the weighting, in which the feature ranking is
computed from an annotation itself; the matching of each profile to one cluster
of a refit by minimum-cost assignment, scored by Jaccard similarity; a refit
constrained to the previous centroids, which keeps the scaler and the weights of
the frozen model and re-estimates only the centroids, started at the development
centroids; one partition built from all eleven annotations at once, which prices
the constraint of sharing a partition; and the whole evaluation applied to the
six comparison methods of Section 5.2 at K = 31, with the same fitting data
wherever a method scales to it.

**Uncertainty and tests.** Confidence intervals are percentile intervals from
1,000 bootstrap resamples of blocks (Bitcoin) or time steps (Elliptic).
Differences between methods are computed within each resample. p-values use the
add-one estimator, and p-values within a hypothesis family are adjusted by
Holm's method [@Holm1979]. The primary comparison statistic is the AP lift.

### 5.4. Environment

All experiments ran on one computer (Intel Core i9-13900HX, 32 logical cores,
64 GB RAM) with Python 3.12 and scikit-learn 1.9.1; the environment is pinned in the repository. The master seed
is 42 and every other seed is derived from it. Multithreaded K-means is not
guaranteed to give bit-identical results; a repeated run of the sensitivity
analysis reproduced every variant exactly except the upsampled corpus, whose
AP lifts differed by at most 0.15.
