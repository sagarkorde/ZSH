::: {custom-style="MDPI_1.1_article_type"}
Article
:::

::: {custom-style="MDPI_1.2_title"}
ZSH: Rank-Weighted Profiling of Bitcoin Transactions and a Pre-Specified Evaluation of What the Profiles Capture
:::

::: {custom-style="MDPI_1.3_authornames"}
Sagar D. Korde ^1,2,\*^ and Narendra M. Shekokar ^1^
:::

::: {custom-style="MDPI_1.6_affiliation"}
^1^ Department of Computer Engineering, Dwarkadas J. Sanghvi College of Engineering, Mumbai, Maharashtra 400056, India; narendra.shekokar@djsce.ac.in; ORCID: https://orcid.org/0000-0002-2507-4140

^2^ K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai, Maharashtra 400077, India

^\*^ Correspondence: sagarkorde04@gmail.com; ORCID: https://orcid.org/0000-0002-6289-6741
:::

::: {custom-style="MDPI_1.7_abstract"}
**Abstract:** Bitcoin's ledger records what each transaction moves but not what it
does; unsupervised profiles describe that activity without labels, yet are rarely tested
against information the clustering never saw. We evaluate ZSH, which weights twelve
features by the rank of their mutual information with a proxy partition, initialises
K-means hierarchically and caps cluster size; the plan and code were frozen before
evaluation. It was fitted on 3.3 million transactions from 2022–2023 and tested on 2.6
million from 2024 and 456,292 collected to August 2026. Its 31 profiles reached 78–84%
of the ceiling for the four annotations they capture best and under 11% for the five
least frequent. Neither oracle weights nor six other clustering families changed this,
yet for P2SH inputs supervision on the same features reached 95% against
the profiles' 24%. Refits agreed at an adjusted Rand index of 0.83, yet by 2026 three
profiles held three quarters of the sample and none survived a free refit; constraining
it to the previous centroids followed 24 of 31 in 2024 but 14 in 2026. For most annotations the limit is the clustering objective rather than the
features; for exchange tags richer features raise what supervision extracts, not what
the profiles recover. These profiles are a readable map of activity, not a detector.

:::

::: {custom-style="MDPI_1.8_keywords"}
**Keywords:** Bitcoin; cluster stability; clustering; CoinJoin; distribution shift; Elliptic data set; feature weighting; transaction profiling
:::

# 1. Introduction

Bitcoin records every transaction on a public ledger. Each record
lists inputs, outputs, amounts, fees and scripts, but it does not say what the
transaction was for. An analyst who looks at a single transaction cannot tell
directly whether it is an exchange paying many customers, a wallet merging
small coins, a token mint, or a payment between two people. Investigators,
compliance teams and researchers still need that context to decide which
activity deserves attention, and anti-money-laundering work on cryptocurrency
rests on it [@Jensen2023].

Most machine-learning work on this problem is supervised. Classifiers trained
on labelled transactions or accounts reach high scores on benchmark data
[@Weber2019; @Elmougy2023], but labels are scarce, they age quickly, and the
output is a risk score rather than a description of activity. Unsupervised
clustering avoids the need for labels and can, in principle, group
transactions that look alike into profiles that a person can read. Address
clustering already serves entity attribution [@AddrClust2024; @Haslhofer2021];
clustering individual transactions is less studied [@Vlahavas2024].

Three practical difficulties stand in the way. First, transaction features are
heavy-tailed and partly redundant, so an unweighted distance can be dominated
by a few monetary fields. Second, a clustering of millions of transactions
should be reproducible: a profile that appears in one fit and vanishes in the
next is of little use. Third, and most important, there is no ground truth for
"behaviour". Profile names are often derived from rules on the same features
that were clustered, so a clustering that agrees with those names has not shown
anything new. Bitcoin usage also changes quickly; the Runes token protocol, for
example, went from nothing to 40% of the transactions in our
2024 test data within months.

This article studies these questions with ZSH, a transaction-profiling pipeline
that combines three standard ingredients: feature weights that decay as a
power of the rank of each feature's mutual information with a proxy partition,
a hierarchical initialisation of K-means, and recursive splitting of clusters
that exceed a size cap. We do not claim a new clustering algorithm. Instead we
ask what these choices change and whether the resulting profiles hold up under
checks that do not depend on the clustering itself:

* **RQ1.** What do rank-power weighting and size-constrained refinement change
  compared with standard clustering on the same data and with the same number
  of clusters?
* **RQ2.** Are the profiles reproducible when the whole pipeline is refitted,
  and do they still describe transactions from later periods?
* **RQ3.** Do the profiles concentrate transaction properties that were not
  used to build them?
* **RQ4.** Do the profiles, or an atypicality score, carry information about
  illicit activity?

The study was specified and its code frozen before any evaluation data were
analysed, and a new sample of transactions from October 2024 to August 2026 was
collected after the freeze. The main contributions are:

1. A corrected and fully documented profiling pipeline. An audit of the
   published transaction sample removed count-derived rule flags, constant and
   duplicate columns, and a unit error from the feature set, and every decision
   was fixed by rules written before evaluation (Sections 3 and 4).
2. An evaluation protocol for transaction profiling without behavioural ground
   truth: matched-K comparisons with standard methods, refit and bootstrap
   stability, transfer to a later test period and to a prospective sample,
   cross-fitted concentration of annotations that were never inputs, and
   external illicit labels (Section 5).
3. Evidence on what the design choices do. Rank-power weighting changes which
   properties the profiles concentrate, raising some and lowering others;
   refinement controls cluster size but hardly changes concentration; and the
   weight decay moves the trade-off gradually rather than having a best value
   (Sections 6.2 and 6.6).
4. Evidence on what the profiles do and do not show. All eleven non-input
   annotations are concentrated above their base rates in the test period,
   although K-means++ does better for exchange tags and mixed-script inputs.
   Refits reproduce the partition fairly well, but a refit on 2024 data does
   not reproduce the transferred profiles once a new protocol dominates the
   data, and by 2026 three profiles hold three quarters of the transactions. A widely used count rule for CoinJoin transactions is right
   only about one time in ten. On Elliptic, the profiles carry little
   information about illicit status, and atypicality scores point the wrong
   way (Sections 6.3–6.5).
5. A public, versioned repository with the frozen analysis plan, the code, the
   prospective-data collector and all result files.

Section 2 reviews related work, Section 3 describes the data and the audit,
Section 4 the method, Section 5 the experimental design, Section 6 the results
and Section 7 their meaning and limits. Section 8 concludes.

# 2. Related Work

## 2.1. Entities and addresses

Most unsupervised analysis of Bitcoin works at the level of addresses and
entities. Common-input-ownership and change-address heuristics group addresses
that are probably controlled by the same party [@AddrClust2024], and platforms such as GraphSense combine such clustering with
attribution tags [@Haslhofer2021]. These methods answer *who* controls funds.
They do not say what a single transaction does, and privacy techniques such as
CoinJoin are designed to break their assumptions [@CoinJoinPriv2024]. The
profiles studied here are complementary: they describe transactions, and they
can be read next to entity information when it exists.

## 2.2. Transaction-level unsupervised analysis

Closest to this work, Vlahavas et al. cluster about 105 million transactions
from blocks 610,000–660,000 [@Vlahavas2024]. They derive eight features, three of
which require traversing the transaction graph (distance to the nearest
coinbase and CoinJoin transaction, and the average time outputs stay unspent),
reduce them to three principal components, and apply trimmed k-means with five
clusters chosen by the elbow method. They interpret the clusters with domain
knowledge and compare them with address tags. Earlier studies used clustering
and outlier detection mainly to flag suspicious activity, most recently with
histogram-based and isolation-based outlier scores [@Witayanont2024]. On the
Elliptic data, a recent comparison of unsupervised detectors — Isolation
Forest, the local outlier factor, k-means and graph autoencoders — with a
supervised graph model found every unsupervised method far behind
[@PerezCano2025], and unsupervised vertex representations built for
anti-money-laundering detection reach the same conclusion [@UnsupAML2025].

These studies leave three questions open that matter for practical use. The
first is reproducibility of the partition itself: results are usually reported
for one fit, and stability under refitting is rarely measured. The second is
time: Bitcoin usage changes quickly, as the arrival of Runes in April 2024
shows, but clusterings are seldom tested on later data. The third is meaning:
cluster names are often taken from rules that are functions of the same
features that were clustered, so agreement with those names is partly built in.

## 2.3. Supervised and graph-based detection

Supervised learning dominates work on illicit activity. On Bitcoin, the
Elliptic data set made graph convolutional networks and tree ensembles the
standard reference points [@Weber2019], and later work added wallet-level
labels and features [@Elmougy2023]; a recent survey covers graph learning on
blockchain data more widely [@Qi2024]. These models need labels, which are scarce and
quickly outdated, and they return a risk score rather than a description of
activity. Their benchmarks also carry their own problems: the construction of
the Elliptic features was never fully disclosed, and information leaks between
commonly used splits [@Safar2026]. The comparison of Section 2.2 makes the same point for detection [@PerezCano2025]. We use
Elliptic only as an external check of whether unsupervised profiles concentrate
labelled illicit activity.

## 2.4. Feature weighting and size constraints in K-means

K-means treats all features alike, so its result depends on scaling and on
redundant features. Feature-weighted variants learn weights inside the
clustering objective, and recent work refines the weighting of features for fully
unsupervised clustering [@FeatWeight2025].
Unsupervised filters such as the Laplacian score rank features by how well they
preserve local structure [@He2005]. Our weighting is simpler: it ranks features
by mutual information with a proxy partition and assigns weights that decay as
a power of the rank. Constrained K-means adds minimum or maximum cluster sizes to the objective [@BalancedClust2026]; we use recursive splitting, which is cheaper at
the scale of millions of rows but gives no guarantee.

## 2.5. Validating a clustering without labels

Internal indices such as the Silhouette [@Rousseeuw1987], Davies–Bouldin
[@DaviesBouldin1979] and Calinski–Harabasz [@Calinski1974] measure compactness and
separation, but they favour some cluster shapes and say nothing about meaning.
Stability-based validation asks whether similar partitions appear when the data are resampled [@StabilitySel2025]; Hennig proposed measuring it
cluster by cluster with the Jaccard similarity [@Hennig2007]. Partitions are
compared with the adjusted Rand index
[@Hubert1985], adjusted mutual information [@Vinh2010] and variation of
information [@Meila2007]. We combine these with two checks that need external
information: concentration of transaction properties that were not clustering
inputs, and transfer to later time periods.

## 2.6. Positioning

ZSH is not a new clustering algorithm. It combines a rank-power feature
weighting, a hierarchical initialisation and a size-constrained refinement
around standard K-means. The contribution of this article is the evaluation: a
frozen, reproducible pipeline tested against matched baselines, under
refitting, on later data, on annotations that were never inputs, and on
external illicit labels, with the results reported whether or not they favour
the method.

# 3. Data

## 3.1. Bitcoin transaction sample

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
blocks), and set the 241 later records aside (Table 1, Figure 1). Every model in this study is
fitted on development data only. Test-period data are used only through
models that were fitted earlier.

Table: **Table 1.** Data used in this study. The development and test periods are parts of the published Bitcoin transaction sample [@Korde2026]; the prospective sample was collected after the analysis code had been frozen; the Elliptic data follow the temporal split of the original study [@Weber2019].

| Data | Period | Transactions | Blocks or labels | Use |
|:-------------|:-------------|-------------:|-----------------:|:-------------|
| Bitcoin sample, development | 2022-07-13 to 2023-12-31 | 3,299,616 | 63,075 | fitting all models |
| Bitcoin sample, test | 2024-01-01 to 2024-09-07 | 2,584,530 | 36,259 | evaluation |
| Prospective sample | 2024-10 to 2026-08 | 456,292 | 18,400 | evaluation |
| Elliptic, training steps | time steps 1–34 | 136,265 | 29,894 labelled | fitting; ranking clusters |
| Elliptic, test steps | time steps 35–49 | 67,504 | 16,670 labelled | evaluation |

![**Figure 1.** Study periods. The analysis code was frozen before any test-period, prospective or Elliptic test result was computed.](figs/F2_design.png){width=6.5in}

## 3.2. Prospective sample

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

## 3.3. Data audit and feature set

The published sample has 27 derived features. An audit of the development
period showed several problems that change their meaning:

* Five behavioural flags — consolidation, distribution, peer-to-peer, batch
  payment and "coinjoin-like" — are fixed rules on the input and output counts
  (Table S1). For example, a transaction is "coinjoin-like" whenever it
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
to output value, and the OP_RETURN and replace-by-fee flags (Table 2).
The coinbase flag is removed by the second rule because only 0.037% of
development transactions are coinbase transactions.

Table: **Table 2.** The twelve clustering features, ranked by their mutual information with the proxy partition on development data, and their rank-power weights (s = 1.5).

| Rank | Feature | Mutual information | Weight |
|--------:|:-------------|------------:|--------:|
| 1 | Serialised size | 1.494 | 0.489 |
| 2 | Virtual size | 1.461 | 0.173 |
| 3 | Fee | 1.050 | 0.094 |
| 4 | Fee rate per byte | 0.978 | 0.061 |
| 5 | Fee rate per virtual byte | 0.962 | 0.044 |
| 6 | Output count | 0.742 | 0.033 |
| 7 | Total input value | 0.690 | 0.026 |
| 8 | Input-to-output value ratio | 0.678 | 0.022 |
| 9 | Mean output value | 0.628 | 0.018 |
| 10 | Input count | 0.608 | 0.015 |
| 11 | Replace-by-fee signalled (0/1) | 0.087 | 0.013 |
| 12 | OP_RETURN output (0/1) | 0.009 | 0.012 |

## 3.4. Annotations that are not clustering inputs

To ask what the clusters mean, we need descriptions of transactions that the
clustering never sees. We use five kinds (Table 3); we call L2–L5
the non-input annotations. They are not statistically independent of the
inputs (script types, for example, affect transaction size), but none of them
is an input or is defined from the inputs.

Table: **Table 3.** Annotations that are not clustering inputs, as shares of all transactions. L2: input script class (all inputs of one class, otherwise mixed); L3: OP_RETURN protocol; L4: GraphSense entity tag on any input or output address; L5: equal-output CoinJoin rule, which needs individual output values. Prospective shares use the design weights.

| Type | Annotation | Development (%) | Test (%) | Prospective (%, weighted) |
|:-----|:-------------|------------:|--------:|------------:|
| L2 | P2PKH inputs | 8.63 | 3.46 | 4.59 |
| L2 | P2SH inputs | 7.44 | 4.35 | 3.16 |
| L2 | P2WPKH inputs | 42.37 | 47.25 | 62.46 |
| L2 | P2WSH inputs | 3.66 | 1.43 | 2.16 |
| L2 | P2TR inputs | 35.90 | 40.94 | 25.41 |
| L2 | Mixed-script inputs | 1.96 | 2.54 | 1.33 |
| L2 | Coinbase | 0.04 | 0.03 | 0.03 |
| L3 | Runes | 0.00 | 40.29 | 39.20 |
| L3 | Omni | 0.04 | 0.03 | 0.00 |
| L3 | Other OP_RETURN | 1.24 | 1.35 | 1.76 |
| L4 | Exchange tag | 0.76 | 0.33 | 0.33 |
| L4 | Miner tag | 0.03 | 0.01 | 0.00 |
| L4 | CoinJoin tag | 0.00 | 0.00 | 0.00 |
| L5 | Equal-output CoinJoin |  |  | 0.08 |

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

## 3.5. Elliptic data

For external labels we use the Elliptic data set [@Weber2019]: 203,769
transactions in 49 time steps with 165 anonymised features, of which 4,545 are
labelled illicit and 42,019 licit. We follow the temporal split of the original
study: time steps 1–34 for fitting and 35–49 for testing. All transactions of
the training steps, labelled or not, are used to fit the clustering; labels
are used only to rank clusters and to evaluate them. Recent work reports that
the construction of the Elliptic features was not disclosed and that
information leaks between commonly used splits [@Safar2026]; we return to this
in Section 7.7.

## 3.6. External labels used only for validation

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

# 4. Method

## 4.1. Overview

ZSH assigns each transaction to a behavioural profile in four steps
(Figure 2): (i) transform and scale the features; (ii) weight the
features by the rank of their relevance; (iii) initialise K-means from a
hierarchical merge of micro-clusters and fit it; (iv) split clusters that hold
more than a set share of the data. New transactions are assigned to the
nearest final centroid. The name refers to these components: zeta-normalised
rank weighting, size-constrained refinement and hierarchical initialisation.

![**Figure 2.** The ZSH pipeline. The shaded steps form the method; the atypicality score is a separate component. All steps are fitted on development data, and later data pass through the frozen model.](figs/F1_pipeline.png){width=6.5in}

## 4.2. Preprocessing

Monetary values are expressed in satoshis. Each continuous feature x is
mapped to $\operatorname{sign}(x)\log(1+|x|)$ and then centred on its
development-period median and divided by its interquartile range; if the
interquartile range is zero, 1.349 times the standard deviation is used
instead. Binary features are left as 0/1. The resulting matrix is
$X \in \mathbb{R}^{n \times d}$ with $d = 12$.

## 4.3. Rank-power feature weights

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

## 4.4. Hierarchical initialisation

Mini-batch K-means with 160 clusters is fitted to $X_w$. The 160 centroids are
merged by Ward's criterion [@Ward1963] with their sizes as weights: the cost of
merging groups $a$ and $b$ is

$$\Delta(a, b) = \frac{n_a n_b}{n_a + n_b} \, \lVert \mathbf{c}_a - \mathbf{c}_b \rVert^2,$$ {#eq:ward}

where $n_a$ is the number of transactions in group $a$ and $\mathbf{c}_a$ its
mean. Merging stops at
$K_0 = 30$ groups, and the size-weighted group means are the initial centroids.

## 4.5. Fitting and size-constrained refinement

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

## 4.6. Profile description

Each profile is described from its own members: its share, the medians of
input and output counts, value, fee rate and virtual size, its input script
mix, its OP_RETURN and Runes shares, its exchange-tag share, the share of each
count rule, and the transaction closest to the centroid. We do not give
profiles behavioural names such as "mixer"; the descriptors state what the
members have in common.

## 4.7. Atypicality score

As a separate component we fit an Isolation Forest [@Liu2008] (200 trees, 256
samples per tree) to 500,000 development transactions in $X_w$ and use the
negative path-length score as an atypicality measure. The score has no
built-in threshold. Section 6.5 tests whether it carries information about
illicit activity.

# 5. Experimental design

## 5.1. Research questions

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
it appears; Table S2 lists all of them and says which is which. Two of the
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
  appear and listed in Table S2.

## 5.2. Baselines and matched comparisons

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

## 5.3. Measures

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

## 5.4. Environment

All experiments ran on one computer (Intel Core i9-13900HX, 32 logical cores,
64 GB RAM) with Python 3.12 and scikit-learn 1.9.1; the environment is pinned in the repository. The master seed
is 42 and every other seed is derived from it. Multithreaded K-means is not
guaranteed to give bit-identical results; a repeated run of the sensitivity
analysis reproduced every variant exactly except the upsampled corpus, whose
AP lifts differed by at most 0.15.

# 6. Results

## 6.1. The fitted profiles

On the development data the rank-power weights are dominated by transaction
size: serialised size receives 0.489 of the total weight and virtual size
0.173, followed by the fee (0.094) and the two fee rates (0.061 and 0.044)
(Figure S1). The OP_RETURN and replace-by-fee flags, whose mutual
information with the proxy partition is close to zero, receive the smallest
weights (0.012 and 0.013). Because the proxy partition is a K-means solution on
the same data, this ranking reflects which features already separate
transactions most strongly; it does not reflect relevance to any behavioural
question.

ZSH produced 31 profiles. One of the 30 initial clusters held 12.1% of the
development transactions and was split in two; after the final reassignment
the largest profile holds 8.7% (Table S3, Figure S2). Fitting
on 3.3 million transactions took 48 s, of which 20 s went to the feature
weights and 19 s to the initialisation; assigning the 2.6 million test
transactions took 2 s.

The profiles are easy to read from their members:

* Four profiles (P00, P02, P03, P09; 27% of development transactions) consist
  almost entirely of one-input, one-output transactions (≥ 98.1%) that spend
  pay-to-taproot inputs (≥ 98.8%) and carry small values (median 1,145 to 31,626 sat); they differ
  mainly in fee rate (median 6, 16, 61 and 210 sat/vB).
* Profiles P01, P05, P06, P10 and P11 are the familiar one-input, two-output
  payments from P2WPKH inputs, again separated by fee rate (3 to 185 sat/vB);
  P07, P15 and P22 are one-input, one-output P2WPKH transactions.
* P08 (4.6%) collects legacy P2PKH payments and P14 (2.7%) P2SH payments.
* Eight profiles are consolidations with a rising number of inputs
  (medians 3, 3, 4, 6, 10, 22, 63 and 200 inputs in P13, P25, P17, P20, P23,
  P26, P27 and P30). The largest of these, P27 (0.6%), has an exchange tag on
  40.7% of its transactions.
* Four profiles are fan-out payments with a median of 8, 25, 51 and 193
  outputs (P21, P24, P28, P29); P24 has exchange tags on 12.3% of its
  transactions.

These descriptions come from annotations that were not inputs (script
classes, tags) and from medians of the inputs. Table S4 gives
a one-line description of every profile together with the transaction closest
to its centroid. None of the profiles is
specific to OP_RETURN use: the highest OP_RETURN share in development data is
22.9% (P19).

## 6.2. RQ1 — What weighting and refinement change

**Comparison with standard methods.** Table 4 compares ZSH with six
methods fitted on the same 200,000 development transactions with the same
number of clusters (K = 31). In the common unweighted space ZSH has the worst
geometry among the K-means-type methods: Silhouette 0.207 against 0.285 for
K-means++, 0.269 for mini-batch K-means and 0.266 for Ward clustering, and a
Davies–Bouldin index of 1.92 against 1.19. In its own weighted space its
Silhouette is 0.302. The Gaussian mixture and BIRCH reach 0.156 and 0.182, and
HDBSCAN labels 44% of its sample as noise. The partial reproduction of
Vlahavas et al. has the highest Silhouette in its own three-dimensional space
(0.764 with k = 5) but among the lowest concentrations for most annotations.

Table: **Table 4.** Methods fitted on the same 200,000 development transactions with the same number of clusters. Geometry is measured on a separate 200,000-row development sample in the common unweighted space (Silh., Silhouette; DBI, Davies–Bouldin index; CH, Calinski–Harabasz index) and, for the Silhouette, also in each method's own space. The largest cluster and the median AP lift over the eleven non-input annotations refer to the test period. The last column counts annotations for which ZSH has a significantly higher or lower AP lift than the method (Holm-adjusted p < 0.05, 95% interval excluding zero). HDBSCAN chooses its own number of clusters and cannot assign new transactions, so it is described on its fitting sample only.

| Method | K | Fit (s) | Silh. | DBI | CH (10³) | Silh. own | Largest (%) | Median lift | Higher / lower |
|:-------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| ZSH | 31 | 20.6 | 0.207 | 1.92 | 36.7 | 0.302 | 15.3 | 7.31 |  |
| K-means++ | 31 | 6.8 | 0.285 | 1.19 | 54.1 | 0.286 | 18.7 | 6.67 | 8 / 3 |
| Mini-batch K-means | 31 | 5.9 | 0.269 | 1.24 | 51.1 | 0.269 | 17.7 | 4.69 | 8 / 3 |
| Gaussian mixture (diag.) | 31 | 22.1 | 0.156 | 2.64 | 17.8 | 0.156 | 39.1 | 4.04 | 7 / 4 |
| BIRCH | 31 | 26.9 | 0.182 | 1.31 | 20.4 | 0.182 | 62.9 | 2.33 | 9 / 2 |
| Ward (sample) | 31 | 27.8 | 0.266 | 1.22 | 52.0 | 0.267 | 20.9 | 5.56 | 9 / 2 |
| Vlahavas et al., partial (k = 5) | 5 | 16.4 | 0.181 | 2.00 | 10.2 | 0.764 | 54.9 | 1.79 | 10 / 1 |
| Vlahavas et al., partial (k = K) | 31 | 180.1 | 0.045 | 4.68 | 5.7 | 0.475 | 39.6 | 2.49 | 9 / 1 |
| HDBSCAN (own sample, 44% noise) | 70 | 24.1 | 0.275 | 0.89 | 3.3 |  |  |  |  |

On the test data (Figure S3), ZSH concentrates eight of the eleven
non-input annotations more strongly than K-means++, and all eight
differences are significant after Holm adjustment: P2PKH inputs (AP lift 23.1
against 15.7), P2SH inputs (5.8 against 2.7), Omni transactions (33.5 against
18.7), other OP_RETURN use (6.2 against 4.9), and smaller gains for P2WPKH,
P2WSH, P2TR and Runes. K-means++ is stronger on the other three: exchange tags
(30.2 against 19.3), coinbase transactions (43.1 against 21.7) and mixed-script
inputs (13.5 against 9.1). It is also stronger on the count rules, which are
functions of the inputs: for example, 38.8 against 22.3 for the
more-than-three-inputs-and-outputs rule. ZSH took 21 s to fit on the
200,000-row sample against 7 s for K-means++.

**Weighting.** The factorial design isolates the two components on all
development data (Table S5, Figure S4). With the
initialisation and K = 31 fixed and no refinement, rank-power weights changed
the concentration of all eleven non-input annotations (H1; all Holm-adjusted
p < 0.05; Table S6). The effect has both signs. It raised the AP lift for coinbase
(+10.0), P2PKH (+6.7), Omni (+9.7), P2SH (+2.9) and other OP_RETURN use (+2.5),
and lowered it for exchange tags (−3.4), mixed-script inputs (−2.6) and, by a
negligible amount, Runes (−0.08). The same pattern holds with refinement
switched on. Weighting also lowered the concentration of the count rules
(−9.4 for the many-input-many-output rule), which is expected because input and
output counts receive weights of 0.015 and 0.033.

The shape of the weight curve matters more than the use of mutual information
itself. Weights proportional to mutual information behaved much like uniform
weights (median AP lift 6.99 against 7.01), while rank-power weights from the
unsupervised Laplacian score produced a different trade-off: the strongest
exchange-tag concentration of all arms (34.8) but weak script-type
concentration (P2PKH 5.0) and a median AP lift of 4.55.

**Refinement.** At matched K, refinement barely changed concentration (H2): no
non-input annotation moved by more than 0.21 AP lift, and five of the eleven
differences (P2WSH, mixed inputs, Omni, other OP_RETURN, exchange tags) were not
significant. Its effect is on cluster sizes. Without refinement the largest
rank-power cluster held 12.1% of the development data; with refinement, 8.7%.
Refinement cost a little geometry (Silhouette 0.211 against 0.228 in the common
space). With uniform weights no cluster exceeded the cap, so refinement did
nothing and the secondary H2 contrast is identical by construction.

**Initialisation.** With uniform features, the hierarchical initialisation
mattered as much as the weights for some annotations. K-means++ with ten random
starts on the full development data (arm A9) reached an AP lift of 5.6 for
P2PKH inputs, whereas the same features with the hierarchical initialisation
(A4) reached 16.5. With rank-power weights the difference largely disappears
(Section 6.6).

## 6.3. RQ2 — Reproducibility and transfer

**Refitting (H3).** Table 5 and Figure 3 summarise 40 refits
of the whole pipeline on one million development transactions each, including
the proxy partition and the feature weights. The mutual-information ranking was
stable (mean Kendall τ with the full-data ranking 0.92 across block-bootstrap
resamples) and so was the number of profiles (31.2 on average, between 30 and
32). Agreement with the full-data partition on a fixed set of 200,000
development transactions averaged ARI 0.83 (s.d. 0.06) across bootstrap
resamples and 0.84 across seeds. For K-means++ with the same K the values were
0.76 and 0.77; for rank-power weights without refinement, 0.84 and 0.86.
Refits of ZSH agreed with each other at a mean pairwise ARI of 0.80 (lowest
pair 0.69).

Table: **Table 5.** Reproducibility of the whole pipeline. ARI, AMI and VI (variation of information) compare each refit with the fit on all development data on a fixed set of 200,000 development transactions (mean ± standard deviation for ARI); pairwise ARI compares refits with each other (lowest pair in brackets). Centroid shift: mean displacement of matched centroids relative to the within-cluster root-mean-square distance. Kendall τ: agreement of each refit's feature ranking with the full-data ranking. Last column: clusters whose mean best Jaccard similarity across the bootstrap refits is at least 0.75, between 0.5 and 0.75, and below 0.5 [@Hennig2007].

| Replicates | Method | K | ARI | AMI | VI | Pair. ARI | Shift | τ | Jaccard groups |
|:-------------|:-------------|--------:|-------------:|--------:|--------:|-------------:|--------:|--------:|-------------:|
| 30 block bootstraps | ZSH | 31.2 | 0.83 ± 0.06 | 0.89 | 1.02 | 0.80 (0.69) | 0.30 | 0.92 | 13 / 16 / 2 |
| 30 block bootstraps | K-means++ | 31.0 | 0.76 ± 0.05 | 0.85 | 1.36 | 0.77 (0.64) | 0.36 |  | 12 / 14 / 5 |
| 30 block bootstraps | Rank-power, no refinement | 31.0 | 0.84 ± 0.06 | 0.89 | 0.97 | 0.81 (0.66) | 0.25 | 0.92 | 15 / 13 / 3 |
| 10 seeds | ZSH | 31.2 | 0.84 ± 0.06 | 0.89 | 1.00 | 0.81 (0.71) | 0.36 | 0.91 |  |
| 10 seeds | K-means++ | 31.0 | 0.77 ± 0.05 | 0.85 | 1.34 | 0.79 (0.69) | 0.36 |  |  |
| 10 seeds | Rank-power, no refinement | 31.0 | 0.86 ± 0.06 | 0.89 | 0.95 | 0.83 (0.66) | 0.26 | 0.91 |  |
| 10 seeds, permuted columns | ZSH | 32.3 |  |  |  | 0.61 (0.36) |  |  |  |

![**Figure 3.** Reproducibility. (a) Agreement of seed and block-bootstrap refits with the fit on all development data. (b) Mean best Jaccard similarity of each cluster across the 30 bootstrap refits; dotted lines mark the thresholds 0.5 and 0.75.](figs/F7_stability.png){width=6.5in}

These numbers need a reference. On data whose columns had been permuted
independently, so that no joint structure remained, ten seed-only refits of
ZSH still agreed at a mean pairwise ARI of 0.61 (s.d. 0.14). The corresponding
value on the real data is 0.81. Part of the reproducibility of any K-means
partition on these features therefore comes from the marginal distributions
alone (heavy tails and a few discrete values); the joint structure adds a
clear but not overwhelming amount.

How precisely is each profile's size known? Resampling blocks gives an interval
for every profile's share of each period (Table S7). The intervals are
narrow: the widest is 0.50 percentage points on development data, 0.73 in the
test period and 1.27 in the prospective sample, and no development interval is
wider than 7.7% of the share it surrounds. All 31 prospective shares lie outside the
corresponding development intervals. We report this as descriptive evidence of
movement only: the intervals describe each period separately, and an interval for
one period that excludes another period's point estimate is neither a confidence
interval for the between-period difference nor a test of it. We have not fitted
such a test, and the size of the change should not be read from this comparison. The cluster-wise Jaccard similarities behave differently.
Their means hide a wide spread across refits: between the 5th and the 95th
percentile they span 0.39 on average and 0.84 for the least stable profile, so a
profile's stability is itself uncertain and should be read as a range.

Cluster by cluster, 13 of the 31 profiles had a mean best Jaccard similarity of
at least 0.75 across the bootstrap refits, 16 lay between 0.5 and 0.75, and 2
fell below 0.5 (Figure 3b). Following Hennig's rule of thumb
[@Hennig2007], the first group can be treated as stable, the second as
indicating a pattern whose boundaries move, and the last as not reliable.
K-means++ had 12, 14 and 5 clusters in these groups. Centroids moved less for
ZSH than for K-means++ (mean matched displacement 0.30 against 0.36 of the
within-cluster root-mean-square distance).

**Transfer to later periods (H4).** The profiles were fitted once on 2022–2023 data
and then used to assign 2024 and 2025–2026 transactions. Figure S5 shows
their monthly shares. Activity was not stationary even within the development
period: the three small-value taproot profiles (P00, P02, P03) together held 2%
of the transactions in March 2023, 32% in May and 48% in August, and they fell
back in 2024. After the Runes protocol launched in April 2024, P05 and P07 grew sharply
in May and June 2024, and P10 and P22 became the largest profiles from July
2024.

Most transferred profiles kept their structural character. Their median input
counts, output counts and fee rates in 2024 stayed close to the development
values, and for 18 of the 31 profiles the Jensen–Shannon distance between the
development and 2024 input-script mixes was below 0.2 (Table 6). The
exceptions are the profiles that absorbed Runes: in 2024, Runes made up 82% of
P10, 83% of P22, 73% of P07 and 63% of P05, and their count-rule mix changed
accordingly (Jensen–Shannon distance 0.65–0.80). The frozen model had no
profile for this activity, so it placed Runes transactions with the structurally
closest existing groups: four profiles hold 80% of them, but none of these
profiles consists of Runes alone.

Table: **Table 6.** Transfer of the development-fitted partitions to later periods. ARI, AMI and the best Jaccard similarities compare the transferred partition with a refit of the same method on the later period. Kendall τ compares the feature ranking of the ZSH refit with the development ranking. The last two columns describe where the Runes transactions fall.

| Period | Method | Profiles | Refit K | ARI | AMI | Jaccard | ≥ 0.75 / < 0.5 | τ | Runes in 80% | Refit ≥ 90% Runes |
|:-----------|:---------|---------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|
| Test | ZSH | 31 | 36 | 0.21 | 0.44 | 0.18 | 0 / 31 | 0.48 | 4 | 2 (36%) |
| Test | K-means++ | 31 | 31 | 0.39 | 0.59 | 0.31 | 0 / 27 |  | 2 |  |
| Prospective | ZSH | 31 | 37 | 0.11 | 0.34 | 0.14 | 0 / 31 | 0.42 | 2 | 3 (54%) |
| Prospective | K-means++ | 31 | 31 | 0.28 | 0.41 | 0.15 | 0 / 31 |  | 2 |  |

A partition refitted on the 2024 data differs substantially from the
transferred one (ARI 0.21; mean best Jaccard 0.18; no profile above 0.75). The
refit found 36 profiles, two of which are at least 90% Runes and together hold
36% of the Runes transactions. Part of the difference comes from the weights
themselves: the mutual-information ranking on 2024 data agreed with the
development ranking only moderately (Kendall τ 0.48, against 0.92 between
resamples of the development data), with the input-to-output value ratio moving
from rank 8 to rank 1. Because the rank-power curve gives the top feature half
of the weight, such a change reshapes the distance. K-means++ with uniform
weights, which has no ranking to change, agreed better with its own refit
(ARI 0.39).

The prospective sample shows the same effect more strongly. Assigning its
456,292 transactions with the frozen model puts an estimated 45.0% of them in
one profile (P10) and 74.7% in three (P10, P12, P22); the largest profile of the
development fit held 8.7%, and of the test period 15.6%. Month by month the
concentration grows: P10 holds 65.7% of the transactions estimated for August
2026 (Figure S5). These three profiles are the low-fee-rate members of
their families, and the median fee rate of their prospective members is 1.3–2.6
sat/vB against 3.0–5.0 sat/vB in the development period. Nineteen of the 31
profiles receive less than 1% of the prospective transactions each, and five
receive almost none. The profiles keep their structural character only in part:
the input-script mix stays close to the development mix for 16 of the 31
profiles (Jensen–Shannon distance below 0.2) and the count-rule mix for 10.

A refit on the prospective data agrees with the transferred partition even less
than on the test data (ARI 0.11 against 0.21; mean best Jaccard 0.14). It
finds 37 profiles, three of which are at least 90% Runes and hold 54% of the
Runes transactions, whereas the transferred model spreads them over the
existing profiles. The feature ranking of the prospective refit also differs
from the development ranking (Kendall τ 0.42): the fee rate per byte moves to
rank 1 and the input-to-output value ratio to rank 2, while serialised size,
the dominant development feature, falls to rank 6. K-means++ with uniform
weights again agreed better with its own refit (ARI 0.28). A model fitted in
2023 therefore describes 2026 transactions poorly, and the size cap it
respected on development data does not hold when the mix of activity moves.

**Can the profiles be followed after a refit?** If refitting is the answer to drift,
the new profiles have to be recognisable as the old ones. They are not
(Table 7). Matching each of the 31 profiles to one cluster of the refit,
no profile reaches a Jaccard similarity of 0.5 in either period; the median best
match is 0.06 for the 2024 refit and 0.01 for the prospective refit, and the best
single case is 0.37 and 0.47. K-means++ does better in the near term — four of its
clusters, holding 41% of the test-period transactions, exceed 0.5 — but by 2026 only
one does, holding 0.3% of transactions. A refit therefore produces a new map rather
than an updated one, which is a practical limit on using these profiles for
monitoring: either the frozen assignment is kept and its drift accepted, or the
series of profile shares starts again at each refit.

Table: **Table 7.** Following profiles across a refit; exploratory, added after the freeze. Each of the 31 development profiles is paired with one cluster of the refit of the same period by minimum-cost assignment on the transactions they share; J is the Jaccard similarity of the two member sets. 'Their share' is the share of the period's transactions held by the profiles matched at 0.5. The 0.5 criterion is a minimum-continuity threshold: it asks only whether a profile remains recognisable across the refit, and is deliberately weaker than the 0.75 used in Section 6.3 as a criterion for cluster stability in Hennig's sense.

| Period | Method | Profiles | Refit clusters | Median J | Mean J | Matched at 0.5 | Their share (%) | Matched at 0.75 |
|:-----------|:---------|---------:|---------:|--------:|--------:|--------:|--------:|--------:|
| Test | ZSH | 31 | 36 | 0.06 | 0.12 | 0 | 0.0 | 0 |
| Test | K-means++ | 31 | 31 | 0.22 | 0.24 | 4 | 41.2 | 0 |
| Prospective | ZSH | 31 | 37 | 0.01 | 0.09 | 0 | 0.0 | 0 |
| Prospective | K-means++ | 31 | 31 | 0.01 | 0.10 | 1 | 0.3 | 0 |

**Unless the refit is constrained.** *This analysis is exploratory: it was added
after the analysis freeze, and it is not part of the pre-specified evidence for any
hypothesis (Table S2).* The refits above start from scratch: they
re-estimate the scaler, the feature ranking and the centroids, so both the space and
the partition move at once. A refit that keeps what the model has already learned
about the space and re-estimates only the centroids, starting Lloyd's algorithm from
the development centroids, behaves very differently (Table 8). In the test
period 24 of the 31 profiles can then be followed above a Jaccard similarity of 0.5,
holding 83.8% of the transactions, against none for the free refit, and the agreement
with the transferred partition rises from an ARI of 0.21 to 0.63. Two years out, 14
of 31 can be followed, holding 27.0% of the prospective transactions. The centroids
do move — 54 Lloyd iterations in the test period and 75 in the prospective sample,
with a mean displacement of 0.23 and 0.62 — so this is a genuine refit and not a
frozen model under another name.

Table: **Table 8.** A refit constrained to the previous centroids; exploratory, added after the freeze. Free refit: the whole pipeline refitted on the period, as in Table 7. Constrained: the scaler and the rank-power weights of the frozen model are kept and only the centroids are re-estimated, started at the development centroids. Followed: profiles whose minimum-cost match with the refit reaches a Jaccard similarity of 0.5, the minimum-continuity threshold of Table 7 rather than the stricter 0.75 stability criterion, and the share of the period transactions they hold. Median J: median Jaccard of the matched pairs. Attained: median share of the ceiling over the annotations of that period, for the frozen model and for the constrained refit.

| Period | ARI, free refit | ARI, constrained | Followed, free | Followed, constrained | Their share (%) | Median J | Attained, frozen (%) | Attained, constrained (%) |
|:-------------|--------:|------------:|----------:|------------:|--------:|--------:|----------:|------------:|
| Test period | 0.21 | 0.63 | 0/31 | 24/31 | 83.8 | 0.62 | 22.6 | 21.1 |
| Prospective sample | 0.11 | 0.34 | 0/31 | 14/31 | 27.0 | 0.44 | 10.8 | 26.1 |

What the constrained refit costs, and what it buys, differ by period. On the test
period it is close to a wash: the median attained share over the eleven annotations
goes from 22.6% to 21.1%, with P2PKH inputs falling from 80.9% to 71.7% and Runes
rising from 78.0% to 90.8%. On the prospective sample it is a clear gain: the median
rises from 10.8% to 26.1%, P2PKH inputs from 42.8% to 84.2%, Runes from 53.0% to
91.5% and other OP_RETURN use from 5.9% to 17.1%, with exchange tags the only
annotation to fall. This is consistent with centroid drift being an important contributor to what the
frozen model loses over two years, and with recentring recovering much of it.
Because the comparison is exploratory and unpaired — a single constrained refit
against a single free one, with no design that separates drift from the other
things a refit changes — it cannot establish that centroid drift is the principal
cause, and we do not claim it is. The practical implication stands: where a
deployed partition must keep its identity, recentring alone recovers a large part
of the loss. What a free refit loses, and the constrained refit keeps, is the feature ranking:
holding it fixed is what lets the profiles keep their identity.

## 6.4. RQ3 — What the profiles concentrate

Table 9 lists the cross-fitted concentration of the eleven
non-input annotations in the test period, and Figure 4 shows
the precision–coverage curves. For every annotation the lower 95% bound of the
AP lift of ZSH lies above one, so all eleven are concentrated above their base
rates. How much this means depends on what was attainable. Read against the
ceiling $1/\pi$, the profiles capture most of what a partition could capture
for the four annotations it captures best — 84% for P2TR inputs, 83% for P2WPKH, 81% for
P2PKH and 78% for Runes — but little for the rare ones: 24% for P2SH inputs,
23% for mixed-script inputs, 10% for P2WSH, 7% for exchange tags and under 1%
for coinbase and Omni transactions, whose large lifts (22 and 26) are small
fractions of ceilings above 2,900. Profiles built from counts, sizes, values
and fees therefore describe the bulk of ordinary activity well and rare
activity poorly, which is the opposite of what a triage tool would want.

Table: **Table 9.** Cross-fitted concentration of the non-input annotations by ZSH and by K-means++ with the same K, both fitted on all development data. AP lift: average precision of the cluster ranking divided by the base rate (KM++: the same quantity for K-means++). Prec.: precision at 25% coverage of the positives, reached with the number of top-ranked profiles given in the next column; dividing it by the base rate gives the enrichment. Intervals from 1,000 block bootstrap resamples; p-values Holm-adjusted over the annotations of each period.

| Period | Annotation | Positives | Base (%) | ZSH AP lift (95% CI) | Maximum lift | Attained (%) | Prec. (%) | Profiles | KM++ | Difference (95% CI) | p |
|:-----------|:-------------|-----------:|--------:|-----------------:|--------:|---------:|--------:|---------:|--------:|-----------------:|--------:|
| Test | Coinbase | 749 | 0.03 | 22.4 (20.2–24.4) | 3,451 | 0.6 | 0.9 | 1 | 14.2 | 8.2 (6.8–9.7) | 0.022 |
| Test | P2PKH inputs | 89,525 | 3.46 | 23.3 (23.1–23.6) | 28.9 | 80.9 | 99.9 | 1 | 5.6 | 17.7 (17.5–18.0) | 0.022 |
| Test | P2SH inputs | 112,449 | 4.35 | 5.4 (5.4–5.5) | 23.0 | 23.6 | 36.2 | 2 | 2.7 | 2.7 (2.7–2.8) | 0.022 |
| Test | P2WPKH inputs | 1,221,160 | 47.25 | 1.8 (1.8–1.8) | 2.1 | 83.4 | 92.5 | 1 | 1.4 | 0.32 (0.31–0.33) | 0.022 |
| Test | P2WSH inputs | 36,881 | 1.43 | 7.3 (7.2–7.4) | 70.1 | 10.4 | 11.4 | 2 | 6.8 | 0.47 (0.38–0.57) | 0.022 |
| Test | P2TR inputs | 1,058,045 | 40.94 | 2.0 (2.0–2.1) | 2.4 | 83.9 | 99.1 | 3 | 1.7 | 0.31 (0.30–0.32) | 0.022 |
| Test | Mixed-script inputs | 65,717 | 2.54 | 8.9 (8.8–9.0) | 39.3 | 22.6 | 29.0 | 2 | 14.0 | −5.2 (−5.3 to −5.0) | 0.022 |
| Test | Runes | 1,041,406 | 40.29 | 1.9 (1.9–2.0) | 2.5 | 78.0 | 82.2 | 2 | 2.0 | −0.05 (−0.05 to −0.04) | 0.022 |
| Test | Omni | 875 | 0.03 | 26.2 (23.0–29.3) | 2,953 | 0.9 | 1.1 | 1 | 15.7 | 10.5 (7.9–12.7) | 0.022 |
| Test | Other OP_RETURN | 35,010 | 1.35 | 7.2 (6.8–7.7) | 73.8 | 9.8 | 13.5 | 2 | 4.7 | 2.6 (2.3–2.9) | 0.022 |
| Test | Exchange tag | 8,649 | 0.33 | 20.8 (20.0–21.8) | 299 | 7.0 | 9.8 | 2 | 30.8 | −9.9 (−10.5 to −9.4) | 0.022 |
| Prospective | P2PKH inputs | 25,190 | 4.59 | 9.3 (8.9–9.7) | 21.8 | 42.8 | 97.1 | 2 | 2.4 | 7.0 (6.6–7.3) | 0.018 |
| Prospective | P2SH inputs | 16,956 | 3.16 | 3.4 (3.2–3.6) | 31.7 | 10.8 | 15.9 | 5 | 2.3 | 1.1 (1.0–1.4) | 0.018 |
| Prospective | P2WPKH inputs | 275,948 | 62.46 | 1.3 (1.3–1.3) | 1.6 | 82.3 | 85.0 | 3 | 1.2 | 0.07 (0.07–0.08) | 0.018 |
| Prospective | P2WSH inputs | 11,386 | 2.16 | 3.3 (3.1–3.5) | 46.3 | 7.1 | 7.1 | 3 | 3.1 | 0.23 (0.11–0.34) | 0.018 |
| Prospective | P2TR inputs | 116,926 | 25.41 | 2.4 (2.4–2.5) | 3.9 | 62.3 | 65.7 | 3 | 2.1 | 0.36 (0.33–0.39) | 0.018 |
| Prospective | Mixed-script inputs | 6,935 | 1.33 | 7.9 (7.3–8.3) | 75.4 | 10.5 | 12.5 | 5 | 10.0 | −2.1 (−3.0 to −1.6) | 0.018 |
| Prospective | Runes | 137,087 | 39.20 | 1.4 (1.3–1.4) | 2.6 | 53.0 | 54.7 | 1 | 1.3 | 0.05 (0.04–0.06) | 0.018 |
| Prospective | Other OP_RETURN | 9,245 | 1.76 | 3.3 (3.0–3.8) | 56.8 | 5.9 | 4.7 | 4 | 2.7 | 0.67 (0.30–1.07) | 0.018 |
| Prospective | Exchange tag | 1,936 | 0.33 | 29.6 (25.3–33.4) | 301 | 9.8 | 14.7 | 1 | 31.8 | −2.2 (−4.6 to −0.2) | 0.048 |

![**Figure 4.** Enrichment (precision divided by base rate) against coverage of the positives for selected annotations, averaged over the two cross-fitting directions.](figs/F9_curves.png){width=6.5in}

* P2PKH inputs are the clearest case. A single profile (P08), holding 1.9% of
  the test transactions, contains 54% of all transactions that spend only
  P2PKH inputs, and 99.9% of its members do so (AP lift 23.3).
* P2WPKH inputs, P2TR inputs and Runes each cover 40–47% of the test
  transactions. Their AP lift cannot be large, because precision cannot exceed
  one; the values are 1.8, 2.0 and 1.9. For P2TR the top three profiles reach a
  precision of 99.1% at 25% coverage.
* Coinbase and Omni transactions are rare (749 and 875 in the test period).
  Their enrichment at 25% coverage is about 32, but the precision there is only
  about 1%. The profiles narrow the search for these transactions to about 2%
  of the data without isolating them.
* Transactions with an exchange tag make up 0.33% of the test period. The two
  profiles that rank highest for this tag contain a quarter of the tagged
  transactions in 0.9% of the data, with a precision of 9.8%.

The same comparison with K-means++ fitted on all development data at the same
K (arm A9) gives the pattern already seen in the matched-sample comparison of
Section 6.2. ZSH is ahead for eight annotations, most clearly P2PKH inputs
(23.3 against 5.6), Omni (26.2 against 15.7) and coinbase (22.4 against 14.2),
and behind for three: exchange tags (20.8 against 30.8), mixed-script inputs
(8.9 against 14.0) and, by a negligible margin, Runes (1.94 against 1.98). All
eleven differences are significant after Holm adjustment (adjusted p ≤ 0.022,
the smallest value attainable with 1,000 resamples and eleven annotations).
Rank-power K-means without refinement at the same K stays within 0.21 of ZSH
for every annotation. Uniform weights with refinement lie close to K-means++,
except for P2PKH inputs (17.3), where the hierarchical initialisation helps.

In the prospective sample, nine of the twelve annotations had enough positives
for the rule of Section 5.3 (at least 200 in each half); coinbase transactions
(284), Omni transactions (16) and equal-output CoinJoins (388)
were too rare to evaluate. All nine remaining annotations are again
concentrated above their base rates, with lower bounds above one, so the
frozen profiles still carry information about properties they never saw two
years after they were fitted. These prospective estimates are design-weighted and
their intervals come from a month-stratified block bootstrap (Section 5.3); the
unweighted, sample-specific versions are in Table S8. The strength is
lower than in the test period for seven of the nine: the AP lift for P2PKH inputs
falls from 23.3 to 9.3, for P2WSH inputs from 7.3 to 3.3 and for other OP_RETURN
use from 7.2 to 3.3, while P2TR inputs (2.4) and exchange tags (29.6) are higher
than in 2024. Two profiles still contain a quarter of the P2PKH-input
transactions at a precision of 97%, and one profile contains a quarter of the
exchange-tagged transactions at a precision of 15%. ZSH concentrates seven of the
nine annotations more strongly than K-means++ at the same K; it is weaker for
mixed-script inputs (7.9 against 10.0) and for exchange tags (29.6 against 31.8),
and all nine differences are significant after Holm adjustment (adjusted
p ≤ 0.048).

**How good is the "coinjoin-like" rule?** The published sample marks a
transaction as coinjoin-like when it has more than three inputs and more than
three outputs; 110,349 transactions (1.9%) meet this rule. We retrieved the
individual output values of 5,000 randomly chosen flagged and 5,000 unflagged
transactions and applied the equal-output rule of Section 3.4 (Table
S9). Of the flagged transactions, 482 (9.6%, 95% CI 8.8–10.5%)
met it; of the unflagged ones, one did. Weighted back to the whole sample,
about 0.20% of the transactions (CI 0.17–0.25%) are equal-output CoinJoins, and
the count rule captures an estimated 90% of them (CI 74–100%; the interval is
wide because it rests on a single unflagged positive). The rule is thus
sensitive but imprecise: about nine in ten coinjoin-like transactions do not
have the equal-output structure. The rate among flagged transactions was lower
in the test period than in the development period (8.1% against 11.3%).

The equal-output transactions we found look like real CoinJoins. The four most
frequent equal values were 0.001, 0.01, 0.05 and 0.5 BTC, the denominations of
the Samourai Whirlpool pools; 178 of the 482 transactions used one of them. The
median number of equal outputs was five, which is the fixed structure of a
Whirlpool transaction [@Stutz2022]. No transaction in either sample matched a
GraphSense coinjoin tag; the only tags found were exchange tags (48 of 10,000
transactions). The prospective sample, where individual output values are available for every
transaction, allows the same comparison without sampling. Of its 456,292
transactions, 388 meet the equal-output rule and 5,152 meet the count rule.
Weighted by the design, equal-output CoinJoins make up 0.077% of transactions
(95% CI 0.067–0.086%), against 0.20% in 2022–2024, and the count rule flags
0.96% (0.90–1.01%), against 1.9%. The rule is again sensitive and imprecise:
its recall is 97.0% (95.0–98.6%) and its precision 7.8% (6.8–8.8%). Both rates
fall over the period: the count rule flags 1.9% of transactions in December
2024 and 0.55% in August 2026, and the equal-output share falls from 0.19% in
October 2024 to between 0.03% and 0.10% in 2026. The equal values are also
different. In 2022–2024 the four Whirlpool denominations were the most frequent; in the
prospective sample the most frequent equal values are powers of two and three
and round decimal amounts (1,062,882 = 2 × 3¹² sat, 354,294 = 2 × 3¹¹ sat,
262,144 = 2¹⁸ sat, 100,000 sat), which is the output pattern of a different
family of implementations. Two events fall just before the prospective period:
the seizure of the Samourai service in April 2024 [@DOJ2024] and the closure of
the Wasabi coordinator in June 2024 [@zkSNACKs2024]. We report the timing
without claiming a causal link, since our data cover only the period after
both. No transaction in the prospective sample matched a GraphSense coinjoin
tag.

**An external check of the CoinJoin family.** Both rules above are rules. A list of
Wasabi 2.x CoinJoin transactions with a known coordinator [@Svenda2026] gives labels
from the source instead (Table S10). It covers coordinators active after
mid-2024, so it matches none of our development transactions, 175 of the test period
and 45 of the prospective sample. On these, the count rule has a recall of 98.9%
(95% CI 95.9–99.7%) in the test period and 100% (92.1–100%) in the prospective
sample, and the equal-output rule, which can be evaluated only where individual
output values are available, has a recall of 97.8% (88.4–99.6%). These are recall figures, and only recall figures:
both rules find nearly all of the *listed* CoinJoins. They say nothing about
precision, which is a separate question answered by a separate evaluation — the
equal-output criterion of Table S9, where the count rule's precision is
9.6% in the test period and 7.8% in the prospective sample. High recall against
this list and low precision against that criterion are consistent, and should not
be combined into a single claim about either rule.

Two limits on the external check bear directly on how far it generalises, and we
restate them here rather than only in Section 7.8. The list covers one
implementation, Wasabi 2.x, and only coordinators active after mid-2024; it
matches 175 test-period and 45 prospective transactions. Nothing here establishes
recall for CoinJoin activity in general, for other implementations, or for other
periods. Of the 388 equal-output transactions in the prospective sample, 44 are on
the external list, so most of the rest come from implementations or coordinators the
list does not cover — which is itself a reminder of how partial the list is.

The labelled CoinJoins also let us check the profiles against a source. In the test
period they fall into 11 of the 31 profiles, four of which hold 80% of them, and 41%
are in profile P29 alone, which holds 0.2% of the test transactions — a concentration
of about 200 times the base rate. In the prospective sample, 42% fall in P30 (0.1% of
transactions). Both profiles are extreme-count groups (P29: two inputs and 193
outputs at the median; P30: 200 inputs and one output), which is what a Wasabi
CoinJoin looks like structurally: the matched transactions have a median of 31 inputs
and 37 outputs. With 175 and 45 positives these numbers are below the 200 per fold
that our cross-fitted rule requires, so we report them as descriptive evidence rather
than as a test.

## 6.5. RQ4 — Illicit activity (Elliptic)

**Concentration.** On Elliptic, ZSH was fitted to all 136,265 transactions of
time steps 1–34. The weights concentrated on a few anonymised features (the
largest weight was 0.41) and the refinement could not break up the dominant
cluster within three levels: 64% of the training transactions remain in one of
the 52 profiles. Clusters were ranked by their illicit rate among the 29,894
labelled training transactions and scored on the 16,670 labelled test
transactions, of which 6.5% are illicit (Table 10, Figure
5).

Table: **Table 10.** Concentration of illicit transactions on Elliptic. Models were fitted on all transactions of time steps 1–34; clusters were ranked by their illicit rate among the labelled training transactions and evaluated on the labelled transactions of steps 35–49 (base rate 6.5%). AF-165: all features; LF-93: local features only. Precision is read at 25% coverage of the test illicit transactions. Intervals from 1,000 resamples of time steps. The random forest is a supervised reference trained on the labelled training transactions; its precision and recall refer to its default decision threshold.

| Features | Method | K | AP lift (95% CI) | Attained (%) | Enrichment (95% CI) | Prec. (%) | Difference (95% CI) | p |
|:---------|:-------------|--------:|-----------------:|---------:|-----------------:|-----------------:|-----------------:|--------:|
| AF-165 | ZSH | 52 | 1.57 (1.26–1.84) | 10.2 | 1.61 (1.25–1.88) | 10.5 |  |  |
| AF-165 | K-means++ | 52 | 1.12 (1.09–1.15) | 7.3 | 1.12 (1.09–1.15) | 7.3 | 0.45 (0.14–0.72) | 0.008 |
| AF-165 | Rank-power, no refinement | 52 | 1.32 (1.14–1.47) | 8.6 | 1.33 (1.12–1.48) | 8.6 | 0.25 (0.11–0.40) | 0.008 |
| AF-165 | Uniform + refinement | 45 | 1.35 (1.13–1.69) | 8.8 | 1.37 (1.11–1.74) | 8.9 | 0.22 (0.04–0.34) | 0.006 |
| LF-93 | ZSH | 52 | 1.54 (1.23–1.82) | 10.0 | 1.56 (1.19–1.86) | 10.1 |  |  |
| LF-93 | K-means++ | 52 | 1.12 (1.09–1.14) | 7.3 | 1.12 (1.09–1.14) | 7.3 | 0.43 (0.12–0.69) | 0.012 |
| AF-165 | Gaussian mixture (diag.) |  | failed to fit |  |  |  |  |  |
| AF-165 | Random forest (supervised) |  | 12.02 | 78.1 |  | 90.9 (recall 72.0) |  |  |

![**Figure 5.** Elliptic test steps. (a) Enrichment against coverage of the test illicit transactions. (b) Precision and recall, by time step, of the clusters that cover 25% of the training illicit transactions; the dotted line marks the dark-market closure at step 43.](figs/F10_elliptic.png){width=6.5in}

The concentration is weak. ZSH reached an AP lift of 1.57 (95% CI 1.26–1.84);
the smallest set of top-ranked clusters that covers a quarter of the test
illicit transactions has a precision of 10.5%, or 1.61 times the base rate
(CI 1.25–1.88). H5 is therefore supported in its stated form—test precision
exceeds the base rate at fixed coverage—but the size of the effect is small:
the average precision is 0.102, so the profiles attain 10.2% of the
concentration a perfect partition of the illicit labels would reach, against
78% for the supervised random forest.
ZSH was ahead of K-means++ at the same K (AP lift 1.12; difference 0.45,
CI 0.14–0.72, Holm p = 0.008), of rank-power weights without refinement (1.32)
and of uniform weights with refinement (1.35). Using only the 93 local features
gave almost the same result (1.54). The Gaussian mixture could not be fitted at
any of the three pre-declared regularisation values and is not reported.

For comparison, a random forest trained on the labelled training transactions
reached a test precision of 0.91, a recall of 0.72 and an average precision of
0.78 (AP lift 12.0). The clusters that rank highest on training labels do not
keep their illicit share in the test period; most test illicit transactions
fall in the large cluster, whose illicit share is modest. Per time step
(Figure 5b), the top ZSH clusters caught 78–98% of the illicit
transactions in each of steps 35–42 at a precision between 3% and 36%. From
step 43, when a dark market closed [@Weber2019], their recall fell to
between 0% and 38% and their precision to at most 4%.

**Held-out actors.** The split above separates the ranking and the evaluation in
time, but the same wallet can appear on both sides. With the actor annotations of
Elliptic++ we can ask what happens when it cannot. A clean separation is impossible
in this data set: the transaction–address graph is a single connected component that
holds 202,803 of the 202,804 transactions with address data, and a random split of
addresses leaves only 20.3% of the labelled transactions with all their addresses on
one side. What is possible is to restrict the ranking: 18,641 of the 29,699 labelled
training transactions with address data (62.8%) share no address with any labelled
transaction of the evaluation period. Ranking the clusters on those alone and scoring
on the 16,346 evaluation transactions with address data changes almost nothing: the
AP lift of ZSH is 1.59 (95% CI 1.30–1.88) against 1.57 with the full training period,
the precision at 25% coverage is 10.9% against 10.5%, and the baselines are unchanged
(K-means++ 1.12). The weak concentration on Elliptic is therefore not an artefact of
wallets shared between the ranking and the evaluation data.

**Atypicality.** Table S11 and Figure S6 test whether
atypicality scores rank illicit Elliptic test transactions above licit ones.
They do the opposite. The Isolation Forest score, fitted on the training time
steps in the ZSH space, had a ROC-AUC of 0.175 (95% CI 0.118–0.278); in the
unweighted space it was 0.182 (0.121–0.297), and the distance to the assigned
ZSH centroid gave 0.269 (0.199–0.373). All three upper bounds lie below 0.5,
so H6 is supported for these scores: the more atypical a transaction looks, the
less likely it is to be illicit. The local outlier factor was at chance
(0.508, 0.454–0.555), an inconclusive result under our decision rule. Illicit
Elliptic transactions are thus typical rather than unusual in these feature
spaces, which is consistent with the weak performance of unsupervised
detectors on this benchmark [@PerezCano2025].

On the Bitcoin data the same kind of score behaves differently, and usefully.
The Isolation Forest fitted on development data, before Runes existed, ranked
the Runes transactions of 2024 as atypical with a ROC-AUC of 0.926 (0.926–0.927),
and transactions with an exchange tag with 0.728 (0.723–0.734). In the prospective sample it ranked the 388
equal-output CoinJoins as atypical (0.907, 0.892–0.918) and the Runes
transactions with 0.952 (0.951–0.954). The score is therefore a reasonable
signal that structurally unusual activity, or activity the development data did
not contain, is present; it is not a signal of illicit activity. We treat these
Bitcoin results as exploratory.

## 6.6. Sensitivity

Table S12 and Figure 6 compare variants of the
pipeline, each fitted on the same 1,000,000 development transactions and
evaluated on the test period. The reference variant uses the settings of the
primary model. It also has 31 profiles, but some of its concentration values
differ from those of the model fitted on all development data (P2PKH 20.4
against 23.3, P2SH 2.7 against 5.4), a first sign that some annotations react
strongly to small changes in the fit.

![**Figure 6.** AP lift on the test period for each sensitivity variant and annotation (cell text). Colour shows the change relative to the reference variant in the top row on a log2 scale; K is given for each variant.](figs/F12_sensitivity.png){width=6.5in}

**Number of clusters.** The initial number of clusters K₀ determines K, and K
had the largest effect. With K₀ = 60 (K = 60) the AP lift was higher than the
reference for all eleven annotations, for example 94.3 for Omni and 47.4 for
coinbase; with K₀ = 10 (K = 20) it was lower for most. This is why methods are
compared only at the same K in this study.

**Weight decay.** A larger s gives more weight to the top-ranked size features.
Between s = 0.5 and s = 3, the AP lift for P2PKH inputs rose from 18.5 to 25.2
and for P2SH inputs from 2.6 to 8.4, while that for exchange tags fell from
24.0 to 16.0 and the Silhouette in the common space fell from 0.257 to 0.119.
Across the eleven annotations, larger values of s raised more AP lifts than
they lowered (nine against two at s = 3), but at the cost of exchange tags,
coinbase transactions and geometry. The trade-off described in Section 6.2 is
therefore gradual, and s = 1.5 is one point on it rather than a best value.

**Size cap and depth.** Before refinement, the largest cluster of the 1,000,000-row
fit held 13.5% of the rows. Caps of 15% or 20% therefore changed nothing and
gave the same 30 clusters as no refinement at all. A 5% cap produced 45
clusters and, as expected from the larger K, higher concentration for most
annotations. Allowing six instead of three levels of splitting gave the same
partition as the reference. The cap holds only for the data the model is
fitted on: in the test period the largest profile of the reference variant
held 18.2% of the transactions, because new activity (Section 6.3) falls into
existing profiles.

**Balanced corpus.** The first version of this work upsampled rare rule
strata before fitting. Repeating this, or using the equivalent sample weights,
changed the result more than any other setting: 39 profiles, a negative
Silhouette in the common space (−0.02 and −0.01) and an AP lift for P2PKH
inputs of 2.8 instead of 20.4. Coinbase transactions, which all carry an
OP_RETURN output with the witness commitment [@BIP141], were multiplied
together with the OP_RETURN stratum and ended up in profiles of their own
(average precision 1). Balancing thus trades the structure of the observed
data for a few rare groups.

**Initialisation and seeding.** With rank-power weights, K-means++ starts did
as well as the hierarchical start or slightly better: the AP lift was higher for
seven annotations and lower for two (P2PKH 21.4 against 20.4, exchange tags 23.1
against 20.8, Omni 24.7 against 29.0). The advantage of the hierarchical start
seen in Section 6.2 therefore holds for uniform weights but not for weighted
features. Semantic seeds built from the count rules raised the
AP lift for seven annotations and lowered it for four. The seed–Ward blend
(λ = 0.6) raised it for nine of the eleven, most for P2SH inputs (6.5 against
2.7) and Omni (37.3 against 29.0), and lowered it for exchange tags (19.9
against 20.9) and Runes. P2SH and Omni are also the annotations that respond most
strongly to s, K₀ and the cap in this table, and choosing a variant after
seeing test results would defeat the purpose of the frozen design. We therefore
report the blend as a promising variant for a future confirmatory study, not as
a replacement for the primary model.

**Size of the proxy partition.** The feature ranking, and therefore every
weight, comes from the mutual information between each feature and a proxy
partition of ten clusters. Table S13 refits the pipeline with 5, 20 and
50 proxy clusters. The ranking is not stable under this choice: Kendall's τ
with the reference ranking is 0.36 for five clusters, 0.88 for twenty and 0.76
for fifty, and the top-ranked feature changes from serialised size to virtual
size or to the fee rate per byte. The number of profiles stays at 31 (33 with
five proxy clusters). Concentration moves in both directions and by large
amounts: with five proxy clusters eight of the eleven annotations are
concentrated more strongly than in the reference and three less strongly
(P2PKH 26.1 against 20.4), with twenty the split is six and five, and with
fifty nine are weaker (P2PKH 9.2, Omni 7.5 against 29.0). The proxy size is
thus as consequential as the weight decay, and the value we fixed in advance is
not the one that concentrates annotations most. This is the clearest
consequence of judging relevance against a partition of the same features:
what counts as relevant depends on how finely that partition is drawn.

**How much could better weights buy?** Table 11 answers this with weights
taken from the annotation itself. The gain is small or absent. For P2SH inputs the
oracle weighting raises the AP lift from 2.7 to 5.7, which is 25% of the attainable
concentration instead of 12%. For P2WSH inputs nothing changes (7.1 against 7.2), and
for exchange tags the oracle reaches 23.5 against 20.8, moving attainment from 7.0% to
7.8%. For P2PKH inputs the oracle is worse than the unsupervised weights (13.4 against
20.4), because emphasising the features that separate P2PKH transactions produces a
partition that splits them across several clusters. Weights proportional to the oracle
mutual information behave similarly. Within this scheme, therefore, feature weighting
is close to exhausted: an oracle that knows the answer in advance cannot concentrate
these annotations much better than the unsupervised ranking, and never beyond a
quarter of what is attainable for the three rarer ones. Feature weighting is not the
lever. Section 6.8 asks whether the features or the objective is, and finds that for
most annotations the features carry far more than the profiles recover.

Table: **Table 11.** An upper bound for the weighting; exploratory, added after the freeze. Each row refits the pipeline on the same 1,000,000 development transactions with weights computed from the annotation itself and evaluates it on the test period. ZSH lift: the unsupervised reference fitted on the same sample. Attained: share of the ceiling 1/π, that is the average precision. 'Best other oracle' is the highest lift this annotation reaches under a weighting derived from one of the other three annotations.

| Annotation | Base (%) | ZSH lift | ZSH attained (%) | Oracle lift | Oracle attained (%) | Oracle MI-direct lift | Best other oracle |
|:------------|--------:|--------:|---------:|--------:|---------:|----------:|--------:|
| P2PKH inputs | 3.46 | 20.4 | 70.8 | 13.4 | 46.5 | 17.4 | 22.7 |
| P2SH inputs | 4.35 | 2.7 | 11.7 | 5.7 | 24.6 | 2.6 | 5.9 |
| P2WSH inputs | 1.43 | 7.1 | 10.1 | 7.2 | 10.3 | 7.3 | 9.1 |
| Exchange tag | 0.33 | 20.8 | 7.0 | 23.5 | 7.8 | 26.0 | 24.6 |

**Leaving families out of the seeds.** Table S14 tests the seeded variant further. If the seeds carry information about a
rule family, removing that family from the seeds should lower its
concentration. For the many-input-many-output and single-input-fan-out rules
this happened (AP lift 19.1 and 19.3 against 21.2 and 21.1 with all seeds), but
seeding with all families was already below the unseeded model (22.1 and
22.4). For fan-out payments seeding helped (5.5 against 4.1), and part of the
gain remained when the family was withheld (4.9). For the other families the
differences were below 0.3. Because the rule families are functions of the
input counts, these results describe how seeds move cluster boundaries; they
do not show that the profiles capture behaviour.

## 6.7. The same evaluation for seven clustering families

**What was done.** *This whole section is exploratory: the seven-family comparison
was added after the analysis freeze (Table S2).* Sections 6.2 to 6.6 compare
ZSH with K-means++ at the same K. The limits they report — concentration bounded by each base rate, weights
that cannot lift it, profiles that do not survive a refit — are therefore
statements about ZSH. To see which of them are statements about the task, the
whole evaluation was repeated for seven families: ZSH, K-means++, mini-batch
K-means, a diagonal Gaussian mixture, BIRCH, Ward clustering and the partial
reproduction of Vlahavas et al. [@Vlahavas2024], all with the same K = 31 and
the same development transactions (Table 12). Six of the seven
use the same twelve features; the Vlahavas reproduction uses the five features
of that study, as in Section 6.2, since reproducing it means reproducing its
feature set. BIRCH and the Vlahavas pipeline were fitted on a
random subsample of 1,000,000 transactions because their cost grows faster than
linearly in the number of rows, and Ward on its own 30,000-row subsample as in
Section 6.2; the other four use all 3,299,616.

Table: **Table 12.** The whole evaluation applied to seven clustering families; exploratory, added after the freeze. Every method is fitted on the development period at K = 31 and uses the same twelve features, with one exception: the partial reproduction of Vlahavas et al. uses the five features of that study, as in Table 4; BIRCH and the Vlahavas reproduction are fitted on a 1,000,000-transaction subsample and Ward on its own 30,000-row subsample, as in Table 4. Largest: share of the period transactions held by the largest cluster. Refit ARI: mean adjusted Rand index between the development partition and ten block-bootstrap refits. ARI: agreement between the transferred partition and a refit on the period itself. Matched: profiles whose minimum-cost match with that refit reaches a Jaccard similarity of 0.5. ZSH chooses its own number of clusters when refitted (37 in the test period, 38 in the prospective sample); the other families are refitted at K = 31.

| Method | Largest, test (%) | Largest, prosp. (%) | Refit ARI | ARI, test | ARI, prosp. | Matched, test | Matched, prosp. |
|:-------------|---------:|---------:|--------:|--------:|--------:|---------:|---------:|
| ZSH | 15.6 | 45.0 | 0.78 | 0.18 | 0.10 | 0/31 | 0/31 |
| K-means++ | 20.4 | 55.0 | 0.76 | 0.44 | 0.22 | 4/31 | 0/31 |
| Mini-batch K-means | 15.2 | 56.2 | 0.67 | 0.33 | 0.15 | 4/31 | 5/31 |
| Gaussian mixture (diag.) | 39.2 | 39.6 | 0.76 | 0.33 | 0.36 | 5/31 | 1/31 |
| BIRCH | 45.7 | 84.5 | 0.29 | 0.34 | 0.27 | 4/31 | 2/31 |
| Ward (sample) | 23.5 | 54.5 | 0.68 | 0.47 | 0.27 | 3/31 | 0/31 |
| Vlahavas et al., partial | 39.7 | 54.8 | 0.66 | 0.68 | 0.66 | 14/31 | 16/30 |

**Concentration against the ceiling.** Table 13 and
Figure 7 give the attained share of every family for every annotation
on the test period. Read across a row and the families differ; read down a
column and the annotation decides. For the three most frequent annotations every
family attains between 45.6% and 97.0% of the ceiling. For coinbase
transactions no family passes 2.5%, for Omni 0.9%, for exchange tags 10.3%,
for other OP_RETURN use 9.8% and for P2WSH inputs 20.1%. Seven families that
build clusters in different ways — around centroids, by
merging a hierarchy and by summarising a feature tree — fail on the same
annotations, which is what Section 6.4 implies: these annotations are too rare
for 31 clusters of these features to isolate, whatever builds the clusters.

Table: **Table 13.** Share of the attainable concentration reached by each family on the test period; exploratory, added after the freeze. Ceiling: the largest AP lift an annotation of this base rate can reach, 1/π; the attained share is the average precision itself. Columns, in order: ZSH, K-means++, mini-batch K-means, the diagonal Gaussian mixture, BIRCH, Ward and the partial reproduction of Vlahavas et al., all at K = 31.

| Annotation | Base (%) | Ceiling | ZSH | KM++ | MBK | GMM | BIRCH | Ward | VKV |
|:-------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Coinbase | 0.03 | 3451 | 0.6 | 0.4 | 0.2 | 0.1 | 0.1 | 2.5 | 0.1 |
| P2PKH inputs | 3.46 | 29 | 80.9 | 19.4 | 58.2 | 14.6 | 14.0 | 32.6 | 10.4 |
| P2SH inputs | 4.35 | 23 | 23.6 | 11.7 | 11.4 | 13.2 | 9.5 | 11.1 | 8.1 |
| P2WPKH inputs | 47.25 | 2 | 83.4 | 68.3 | 70.1 | 64.9 | 55.0 | 69.6 | 51.6 |
| P2WSH inputs | 1.43 | 70 | 10.4 | 9.7 | 9.0 | 20.1 | 3.1 | 10.8 | 3.0 |
| P2TR inputs | 40.94 | 2 | 83.9 | 71.2 | 72.9 | 71.9 | 50.0 | 72.4 | 45.6 |
| Mixed-script inputs | 2.54 | 39 | 22.6 | 35.7 | 29.7 | 33.3 | 32.7 | 31.1 | 7.3 |
| Runes | 40.29 | 2 | 78.0 | 79.9 | 80.8 | 97.0 | 62.5 | 79.7 | 96.7 |
| Omni | 0.03 | 2953 | 0.9 | 0.5 | 0.6 | 0.7 | 0.2 | 0.9 | 0.1 |
| Other OP_RETURN | 1.35 | 74 | 9.8 | 6.3 | 4.7 | 4.4 | 3.4 | 8.7 | 3.7 |
| Exchange tag | 0.33 | 299 | 7.0 | 10.3 | 8.4 | 9.9 | 9.8 | 9.5 | 5.8 |
| Median |  |  | 22.6 | 11.7 | 11.4 | 14.6 | 9.8 | 11.1 | 7.3 |

![**Figure 7.** Share of the attainable concentration reached on the test period by each clustering family; exploratory, added after the freeze. Cells give the attained share as a percentage of the ceiling 1/π; the base rate of each annotation is printed under its name. All families are fitted at K = 31 on the same twelve features, except the partial reproduction of Vlahavas et al., which uses the five features of that study.](figs/F13_bench.png){width=6.5in}

Within that band the families do differ, and the largest difference belongs to
the weighting. On P2PKH inputs ZSH attains 80.9% against 58.2% for mini-batch
K-means, 32.6% for Ward and 19.4% for K-means++. Its median attained share over
the eleven annotations is 22.6%, the highest of the seven (Gaussian mixture
14.6%, K-means++ 11.7%, mini-batch K-means 11.4%, Ward 11.1%, BIRCH 9.8%,
Vlahavas et al. 7.3%). Two annotations go the other way: mixed-script inputs,
where K-means++ attains 35.7% against 22.6%, and Runes, where the Gaussian
mixture reaches 97.0% and the Vlahavas pipeline 96.7% against 78.0%.

The advantage does not last. On the prospective sample the median attained
shares are 13.2% for the Gaussian mixture, 12.8% for mini-batch K-means, 11.5%
for BIRCH, 10.9% for K-means++, 10.8% for ZSH, 10.7% for Ward and 5.8% for the
Vlahavas pipeline. Only P2PKH inputs keep a clear gap (42.8% against 10.9% for
K-means++). Two years after the fit the families are no longer distinguishable
by this measure.

**Cluster sizes.** In the test period the largest cluster holds between 15.2%
(mini-batch K-means) and 45.7% (BIRCH) of the transactions, with ZSH at 15.6%.
In the prospective sample it holds between 39.6% (Gaussian mixture) and 84.5%
(BIRCH), with ZSH at 45.0%, and three clusters hold between 63.6% and 97.2%
of the transactions depending on the family. The concentration of later
activity into a few clusters reported in Section 6.3 is therefore not a
consequence of the size cap or of the weights. Every family shows it, and the
size cap only limits how far it goes.

**Reproducibility.** Ten block-bootstrap refits of each family, compared with
its own development partition on the same 200,000 transactions, give a mean ARI
of 0.78 for ZSH, 0.76 for the Gaussian mixture and for K-means++, 0.68 for
Ward, 0.67 for mini-batch K-means, 0.66 for the Vlahavas pipeline and 0.29 for
BIRCH. The first three are close enough to be read as one group. Two values
need care. The Vlahavas pipeline has a standard deviation of 0.32 across
replicates and one refit that agrees with its own development partition at 0.08,
so its trimmed k-means sometimes finds an entirely different partition of the
same data. BIRCH at 0.29 does not reproduce its own partition at all. The ZSH
value of 0.78 is lower than the 0.83 of Section 6.3, which used thirty
resamples, different resamples and a different reference sample; the difference
is smaller than the spread across replicates (s.d. 0.06).

**Following profiles across a refit.** Refitting each family on the later
period and matching its clusters to the transferred ones (Section 6.3) shows
that the families which can be followed are those whose partitions are close to
degenerate. The Vlahavas pipeline matches 14 of 31 clusters above a Jaccard
similarity of 0.5 in the test period, holding 89.6% of its transactions, and 16
of 30 in the prospective sample holding 97.6%; BIRCH matches 2 of 31 in the
prospective sample, but those 2 hold 84.7% of the transactions. In both cases a
single cluster holds most of the data, so the match is easy and says little.
Where the partition is balanced, almost nothing survives: ZSH matches no
profile in either period, K-means++ 4 of 31 in the test period (41.2% of
transactions) and none in the prospective sample, Ward 3 and none, mini-batch
K-means 4 and 5, the Gaussian mixture 5 and 1. ZSH is also the only family that
chooses its own number of clusters when refitted — 37 in the test period and 38
in the prospective sample against 31 for the others — which lowers the Jaccard
similarity of any match it can make. These are new refits with their own seeds;
the refits of Section 6.3 found 36 and 37 clusters, so the number itself moves
with the seed. The counts at the 0.5 threshold are
themselves sensitive to the refit seed: the K-means++ refit of Table
7, which used the seed of Section 6.3, left one prospective profile
just above the threshold holding 0.3% of the sample, and the refit here leaves
none.

**What the comparison separates.** Three of the results of this article belong
to ZSH: the highest median attained share on the test period, a much stronger
concentration of legacy P2PKH spending than any other family, and refits that
agree at least as well as any other family's. Three belong to the task and not
to the method: the ceiling on rare annotations, which no family approaches; the
collapse of the partition into a few clusters two years after the fit, which
every family shows; and the impossibility of following a balanced partition
through a refit, which only near-degenerate partitions avoid. Reporting the
first three without the second three would describe properties of unsupervised
transaction profiling as properties of one method.

## 6.8. Is the limit the features or the objective?

*This section is exploratory: the supervised benchmark and the address-level
comparison were both added after the analysis freeze (Table S2).*

Section 6.6 shows that weights computed from an annotation itself add little, and
Section 6.7 that six other clustering families reach the same low shares for rare
annotations. Two explanations remain: the twelve features do not carry the
annotation, or they carry it and the clustering does not find it. Replacing the
clustering with a supervised model on exactly the same twelve features separates
them (Table 14). A gradient-boosted tree is fitted for one annotation at a
time and its score is cut into 31 cells, which are then ranked and scored by the same
cross-fitted procedure as the profiles, so the columns are comparable.

Table: **Table 14.** What the same twelve features give a supervised model; exploratory, added after the freeze. For each annotation a gradient-boosted tree is fitted on the twelve clustering features and its score is cut into 31 cells, which are then ranked and scored exactly as the profiles are, so the columns are comparable. Equal cells: cells of equal size, which cannot isolate a rare annotation whatever the score. Benchmark: cell sizes free (K-means on the score), fitted on one block-parity half of the test period and scored on the other. A year earlier: cell sizes free, fitted on the development period and therefore using no test-period label. All values are shares of the ceiling 1/π. The benchmark is fitted for one annotation at a time; Section S1 reports a single partition built from all eleven.

| Annotation | Base (%) | Ceiling | Profiles (%) | Equal cells (%) | Benchmark (%) | A year earlier (%) |
|:-------------|--------:|--------:|---------:|--------:|----------:|--------:|
| Coinbase | 0.03 | 3451 | 0.6 | 49.9 | 88.9 | 90.4 |
| P2PKH inputs | 3.46 | 29 | 80.9 | 96.7 | 98.7 | 96.9 |
| P2SH inputs | 4.35 | 23 | 23.6 | 90.9 | 98.5 | 95.0 |
| P2WPKH inputs | 47.25 | 2 | 83.4 | 99.9 | 100.0 | 99.7 |
| P2WSH inputs | 1.43 | 70 | 10.4 | 43.6 | 95.9 | 89.1 |
| P2TR inputs | 40.94 | 2 | 83.9 | 99.9 | 100.0 | 99.5 |
| Mixed-script inputs | 2.54 | 39 | 22.6 | 73.6 | 95.6 | 89.1 |
| Omni | 0.03 | 2953 | 0.9 | 0.5 | 52.7 | 2.4 |
| Other OP_RETURN | 1.35 | 74 | 9.8 | 41.8 | 97.0 | 4.8 |
| Exchange tag | 0.33 | 299 | 7.0 | 7.4 | 46.6 | 28.2 |
| Median |  |  | 16.5 | 61.8 | 96.4 | 89.8 |

How the score is cut matters, and the table reports two ways of doing it. Cells of
equal size are the obvious choice and the wrong one: an annotation with a base rate of
0.03% cannot exceed about 1% purity in cells holding a thirty-first of the data each,
whatever the score says, so equal cells measure the cutting rule rather than the
features. A clustering is under no such constraint, so the benchmark that matters lets the
cell sizes vary. The difference is large exactly where it should be — 43.6% against
95.9% for P2WSH inputs, 7.4% against 46.6% for exchange tags — and negligible for the
common classes, where it changes nothing.

Read against the benchmark with free cell sizes, the features carry far more than the
profiles recover, for every annotation without exception. P2SH inputs are the clearest
case: the profiles attain 23.6% of the ceiling and a supervised split of the same
features attains 98.5%. Mixed-script inputs go from 22.6% to 95.6%, P2WSH inputs from
10.4% to 95.9%, other OP_RETURN use from 9.8% to 97.0% and coinbase transactions from
0.6% to 88.9%. Table 14 covers ten annotations rather than the eleven used elsewhere:
Runes is a test-period protocol and has no development-period positives, so the
transfer column cannot be computed for it. The median over those ten is 16.5% for the
profiles and 96.4% for the benchmark, lower than the 22.6% median quoted in Section 6.7
because that one includes Runes, which the profiles capture well. Even where the profiles do well they leave something: 80.9% against
98.7% for P2PKH inputs and about 84% against 100% for the two common witness classes.
The information is in the twelve numbers, and K-means at K = 31 does not put it into
separate clusters.

The last column of Table 14 makes the comparison without using any test-period
label at all: the model is fitted on the development period a year earlier. It reaches
95.0% for P2SH inputs, 96.9% for P2PKH, 89.1% for P2WSH and mixed-script inputs and
90.4% for coinbase transactions. The P2SH figure is the comparison quoted in the
abstract and the conclusion: 95.0% for the supervised benchmark against 23.6% for
the profiles is a statement about P2SH inputs specifically, not a general summary
of the eleven annotations, whose attained shares range from 0.6% to 83.9%. This is the cleaner comparison, because the in-period
benchmark is optimistic — with two block-parity folds, the labels of the half on which the
clusters are ranked have influenced the scores, and therefore the cell membership, of
the half on which they are scored. Both columns say the same thing, and the transfer
column says it without that caveat.

Two annotations behave differently in that last column, and the difference is temporal
rather than structural. Trained a year earlier, the benchmark falls from 97.0% to 4.8% for
other OP_RETURN use and from 52.7% to 2.4% for Omni, while the script classes lose only
a few points. Supervision on these features is not a stable alternative to the profiles:
for the annotations whose composition changes, it decays exactly as the profiles do
(Section 6.3). Exchange tags sit between the two: 46.6% within the period and 28.2% a
year earlier, against 7.0% for the profiles.

One difference between the columns should be kept in mind: the supervised benchmark is
fitted for one annotation at a time, whereas the profiles are a single partition that
has to serve all of them at once. We priced that constraint by building one partition
from all eleven annotations together — the eleven out-of-fold scores of a transaction
form its coordinates, and K-means cuts that space into 31 clusters. Sharing costs
little, and we price it against the benchmark this section argues for rather than the
one it rejects: the shared partition attains a median of 92.8% against 96.4% for the
per-annotation benchmark with free cell sizes, so sharing costs about four points of the
ceiling. It is higher than the single-annotation benchmark for two of the ten annotations
of Table 14 and equal for one, because eleven coordinates describe a transaction
better than one does. Against the equal-cell variant the shared partition looks far
stronger still (92.8% against 74.1%), but that comparison inherits the constraint we have
just rejected and we do not rest anything on it. The gap between the profiles and the benchmark is therefore not an artefact
of asking one partition to do eleven jobs.

**Do address-level features add anything?** *Exploratory, added after the freeze.*
The cached records of the prospective
collection carry the full transaction, so for those 456,292 transactions we added
twelve features the published sample does not contain: how often each input and output
address occurs elsewhere in the sample, whether the transaction pays back to one of its
own input addresses, the number and size of witness items, sigops, locktime and
sequence use, and the dispersion of the input and output values. Input script types
were not used, because they are the annotations of Section 3.4. Address reuse is
counted inside the sample, which holds up to 25 transactions from each of 800 blocks
per month, so every count is a lower bound on reuse in the chain.

They add a great deal for the one annotation that needs it (Table 15). For
exchange tags the supervised benchmark on the twelve features is 40.8% and on all
twenty-four 59.6%, a gain of 19 percentage points, which is what one would expect of a
property defined by the addresses a transaction touches. For the two OP_RETURN
annotations they add much less (82.0% to 86.4% for other OP_RETURN use, 99.4% to
99.7% for Runes), which is also expected: those are properties of the outputs
themselves, already visible to the twelve. Refitting the profiles on all twenty-four
features moves them very little — 9.4% against 9.8% for exchange tags, and 4.8%
against 5.9% for other OP_RETURN use — so the extra information does not reach the
profiles through the clustering objective either.

Table: **Table 15.** Address-level features against the twelve, on the prospective sample; exploratory, added after the freeze. Twelve features built from the cached transaction records — input and output address reuse inside the sample, paying back to an own input address, witness items and bytes, sigops, locktime and sequence use, and the dispersion of input and output values — are added to the twelve of Table 2. Frozen: the development model transferred. Refit 12 and Refit 24: the pipeline refitted on this period with twelve and with twenty-four features. Sup.: the supervised benchmark of Table 14 computed on this period, with cell sizes free. All values are shares of the ceiling 1/π. Input script types are excluded because they are annotations.

| Annotation | Positives | Base (%) | Ceiling | Frozen (%) | Refit 12 (%) | Refit 24 (%) | Sup. 12 (%) | Sup. 24 (%) |
|:-------------|----------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Runes | 137,087 | 39.20 | 3 | 53.0 | 89.7 | 91.9 | 99.4 | 99.7 |
| Other OP_RETURN | 9,245 | 1.76 | 57 | 5.9 | 2.8 | 4.8 | 82.0 | 86.4 |
| Exchange tag | 1,936 | 0.33 | 301 | 9.8 | 3.0 | 9.4 | 40.8 | 59.6 |

This is a bounded test. Reuse measured inside a sparse sample is a weak proxy for reuse
in the chain, so the 59.6% is likely an underestimate of what an entity-level view could
give. Taken with the rest of the section it points to a division of labour: for the
properties of a transaction's own structure the twelve features suffice and the
objective is what fails, while for properties of the parties behind it, richer features
are worth having and the clustering still has to be able to use them.

# 7. Discussion

## 7.1. What the weights do

The rank-power weights did not make the profiles "better" in general. They
moved the clustering towards transaction size and fees, and this changed which
properties the profiles concentrate: script types and OP_RETURN protocols
gained, exchange tags and multi-script inputs lost (Section 6.2). A method that
is judged on one annotation can therefore look strong or weak depending on the
choice of annotation. We think this is the central practical point for
unsupervised transaction profiling: the weights encode a view of what matters,
and that view should be stated and tested rather than presented as neutral.

The ranking is also self-referential. Mutual information is measured against a
K-means partition of the same features, so features that already dominate an
unweighted partition are ranked first and then weighted more. It also depends
on how fine that partition is: with five instead of ten proxy clusters the
ranking agrees with ours at a Kendall τ of only 0.36, and the concentration of
most annotations rises (Section 6.6). A practitioner who picks a different
proxy size gets a different distance, so this choice should be reported and
tested rather than left implicit. The zeta
normalisation contributes only the constant H_d(s); what shapes the result is
the decay of the weights with rank, and the choice of ranking signal
(mutual information or Laplacian score) moves the trade-off further.

## 7.2. Refinement and initialisation

Refinement is not a source of concentration. At the same number of clusters it
changed the AP lift of no non-input annotation by more than 0.21. Its value
is practical: on the data it is fitted on, it keeps any single profile from
absorbing a large part of the data, which matters when profiles are used to
split work. The cap says nothing about later data: in the prospective period
one profile of the frozen model held 45% of the transactions. The cap is only a
target. On the Bitcoin data one split was enough; on Elliptic three levels of
splitting left a cluster with 64% of the transactions, because K-means kept
separating small groups from a dense core. A size-constrained objective [@BalancedClust2026] would enforce the cap but would also force apart transactions
that are genuinely alike.

The hierarchical initialisation had a larger effect than we expected. With the
same uniform features and the same K, starting from merged micro-clusters
instead of ten K-means++ runs raised the AP lift for P2PKH inputs from 5.6 to
16.5. K-means has many local optima on data like these, and which optimum is
reached changes what the clusters mean. The sensitivity analysis qualifies
this. With rank-power weights, K-means++ starts did as well as the
hierarchical start or slightly better, so the hierarchical start helps mainly
when the features are unweighted. The number of clusters had a larger effect than any
other setting, and the balanced corpus used in the first version of this work
removed much of the structure that the profiles otherwise capture. The
seed–Ward blend concentrated most annotations somewhat more strongly than the
primary model; whether this holds up is a question for a new confirmatory
study, not for this one.

## 7.3. Reproducibility and change over time

Refitting the whole pipeline on resampled blocks reproduced the partition well
but not perfectly (ARI about 0.8), slightly better than K-means++. The
column-permuted reference shows why such numbers must be read with care: even
without joint structure, K-means partitions of these heavy-tailed features
agree at ARI 0.61 across seeds. About two fifths of the profiles were stable in
Hennig's sense and two were not; profile-level conclusions should rest on the
stable ones.

Transfer to later data is the weakest part of the method. The frozen
profiles kept their structural character in 2024, but the Runes transactions
were spread over existing profiles, and a refit on 2024 data produced a
different partition (ARI 0.21). Part of the difference comes from the feature
ranking, which changed with the data; because the top-ranked feature receives
almost half of the weight, a new ranking gives a new distance. In the
prospective sample the mismatch is larger: three profiles absorb three quarters
of the transactions of 2025 and 2026, and a refit agrees with the transferred
partition at an ARI of 0.11. What did survive is concentration: two years
after the fit, every annotation with enough positives in the prospective
sample was still concentrated above its base rate, although more weakly than
in 2024. The profiles therefore keep some meaning while losing their balance. The reason is visible in the profile descriptors.
The profiles separate transactions mainly by size and fee rate, and the fee
market of 2025–2026 is not the one of 2022–2023: the median fee rate of the
three profiles that absorb the traffic fell from 3–5 to 1–3 sat/vB. A model
fitted once should therefore be refitted when activity changes. That is easy to say
and hard to use: when we match the profiles of a free refit to the originals, none of
the 31 reaches a Jaccard similarity of 0.5 (Section 6.3). A free refit gives a new map,
not an updated one. Constraining it changes that. Keeping the scaler and the weights of
the frozen model and re-estimating only the centroids, started at the development
centroids, lets 24 of the 31 profiles be followed into 2024 and 14 into 2026, and on
the prospective sample it also recovers most of the concentration the frozen model had
lost — the median attained share rises from 10.8% to 26.1% and legacy spending from
42.8% to 84.2%. This is consistent with much of the instability of a free refit being the model
moving rather than the data, though the exploratory comparison of Section 6.3 cannot
separate the two; what the constrained refit holds fixed is the feature ranking, and
holding it fixed is what lets the profiles keep their identity. For a
monitoring system this is the practical recipe — freeze the space, re-estimate the
centroids, and match the profiles across refits — rather than a choice between a
drifting frozen model and a series that restarts. This is not a problem of one method: of the seven
families in Section 6.7, the only ones whose profiles can be followed through a
refit are those in which a single cluster already holds most of the data, where
the match is easy and uninformative.

## 7.4. What the profiles say about illicit activity

On Elliptic, unsupervised profiles carried little information about illicit
status once they were evaluated on later time steps. ZSH was better than the
unweighted baselines, but a precision of about 10% at a quarter of the illicit
transactions is far from a supervised random forest, which reached a precision
of 91% at a recall of 72%. The clusters that looked most illicit in training
did not stay so, and after the dark-market closure at step 43 the top clusters
found at most 38% of the illicit transactions of any time step. This
agrees with the recent comparison of unsupervised detectors and supervised
graph models on the same data [@PerezCano2025], and with the general difficulty of temporal shift on this
benchmark [@Weber2019]. Choosing the "most illicit" cluster on
the same labels that are used to score it would have given a much more
favourable impression; the temporal split prevents that.

The atypicality scores point the same way. On the Elliptic test steps,
illicit transactions received lower Isolation Forest scores than licit ones,
and the local outlier factor was at chance. On the Bitcoin data, the same kind
of score ranked the Runes transactions of 2024 as unusual. Atypicality thus
signals novelty relative to the fitting period, not illicit activity.

## 7.5. What limits the profiles

Three experiments bound the same gap from different sides, and together they say
where the limitation lies. The first is the ceiling itself. Reading concentration
against 1/π separates a partition that fails from an annotation that is simply rare
(Section 6.4): a lift of 23 out of a possible 29 and a lift of 21 out of a possible
299 look similar and are not. A study that reports only lifts cannot tell them apart.

The second is the weighting. Weights computed from the annotation itself — an oracle
that no unsupervised method has — lift P2SH attainment from 12% to 25%, leave P2WSH
unchanged and make P2PKH worse (Section 6.6). Within this scheme feature weighting is
close to exhausted, which is a negative result about the part of the method that gives
it its name.

The third is the feature set, and it is the one that changed our reading of the other
two. A supervised model on exactly the same twelve features, fitted a year earlier on
the development period and using no test-period label, attains 95.0% of the ceiling for
P2SH inputs where the profiles attain 23.6%, 89.1% against 22.6% for mixed-script
inputs and 90.4% against 0.6% for coinbase transactions (Section 6.8). Over the ten
annotations the median is 96.4% for a benchmark fitted within the period, 89.8% for one
fitted a year earlier, and 16.5% for the
profiles, with no annotation exempt. The information is in the twelve numbers. What the
profiles lack is not better features and not better weights but an objective that looks
for this structure: K-means minimises within-cluster variance, and a partition of
minimum variance is not a partition of maximum concentration. The same pattern appears
on Elliptic, where clusters of the 165 features attain 10.2% and a random forest on
those same features 78% (Section 6.5).

An earlier reading of ours did not survive this measurement, and we report it
because the correction is part of the result. The first version of this analysis cut the
supervised score into cells of equal size, which no clustering is obliged to do and
which caps what any rare annotation can reach; it made exchange tags and Omni look like
cases where the features were empty. With the cell sizes free, the benchmark for exchange
tags is 46.6% within the period and 28.2% a year earlier, against 7.0% for the profiles.
There is no annotation for which these features carry nothing.

Two of the results are properties of the task rather than of ZSH, which the benchmark
establishes (Section 6.7): every family's partition collapses into a few clusters in
the prospective period, and profiles survive a refit only where a single cluster
already holds most of the data. The third, the gap to the supervised benchmark, is a
property of the objective that all seven families share — for most of the
annotations tested. Exchange tags are the documented exception: there the
twelve-feature representation is itself a binding constraint, and richer
features relax it (Section 6.8).

Where richer features help is where the annotation is a property of the parties rather
than of the transaction. Adding address reuse, witness structure, sigops and value
dispersion to the twelve, on the prospective sample where the full records are
available, raises the supervised benchmark for exchange tags from 40.8% to 59.6%, while
adding almost nothing for the OP_RETURN annotations, which the twelve already describe
(Section 6.8). Refitting the profiles on all twenty-four features moves them barely at
all, so the extra information does not reach them through the clustering objective
either. Two things would therefore be worth building: a criterion that optimises the
concentration of held-out properties directly, or a partition shaped by a few labels
and then applied unsupervised; and, for entity-level properties specifically, features
built from the address graph rather than from the transaction alone.

## 7.6. Practical use

The results suggest three uses and one misuse. First, the profiles give a
compact, readable map of on-chain activity: a few dozen groups, each described
by its members' counts, sizes, fee rates and script types, and each assigned
by a nearest-centroid rule that is cheap to apply to new blocks. Second,
because the assignment rule is fixed, the monthly profile shares can be
monitored. A profile whose share or composition changes quickly, as happened
when Runes appeared, marks a change in activity that deserves a closer look;
the atypicality score of a model fitted before the change points in the same
direction. Third, for properties that the profiles concentrate strongly,
such as legacy P2PKH spending, the top-ranked profiles can narrow a manual
search, and the precision and coverage reported here tell the analyst what
to expect. The misuse would be to read a profile as a verdict about the
parties behind a transaction. Nothing in this study links a profile to
illicit activity, and on Elliptic the more atypical transactions were less
likely to be illicit.

Anyone who uses the profiles for triage should re-estimate precision and
coverage on their own labelled data, because both depend on the annotation,
on K and on the period.

## 7.7. Privacy and responsible use

The profiles describe transactions, not people, and nothing in this study links a profile
to an identity. That distinction is easy to lose in practice, so it is worth stating what
these profiles can and cannot support.

Transaction profiling does not deanonymise anyone by itself. A profile is a position in a
space of counts, sizes, values and fees, shared by tens of thousands of transactions;
knowing that a transaction sits in P05 says what it looks like, not who made it. The
risk is compositional: profiles become identifying when they are joined to address
clustering and attribution tags, which is exactly how the entity data used here as
annotations were built. Anyone combining the two inherits the error rates of both, and
those error rates are not small. The entity tags cover 0.3% of transactions in the test
period, and the profiles concentrate them at 7.0% of what is attainable — the weakest
result in this study.

The atypicality score deserves particular care, because its failure mode is the opposite
of what a user would assume. On the Elliptic data illicit transactions received *lower*
scores than licit ones (ROC-AUC 0.175), so treating a high score as suspicion would
invert the intended reading. What the score does detect is novelty relative to the
fitting period: it ranked Runes transactions as unusual in 2024 (0.926) and equal-output
CoinJoins in the prospective sample (0.907). Both are legitimate activity. A score that
reliably flags privacy-preserving transactions and new protocols, while failing on
illicit ones, would be a poor basis for any decision affecting a person, and a harmful
one if it directed scrutiny towards users of privacy tools.

Three consequences follow for anyone deploying this kind of profiling. Profiles should be
used to organise review, not to decide it: the precision and coverage reported here
(Section 6.4) are the honest expectation, and for rare properties they are low. Any
operational use should re-estimate those numbers on the deployer's own labels, because
they depend on the annotation, on K and on the period, and they decay — every family
tested here lost the structure of its partition within two years (Section 6.7). And a
profile assignment should never be recorded as a property of a party to a transaction,
because the model was never given, and cannot recover, who that party is.

## 7.8. Limitations and threats to validity

* The transaction sample is a sample: roughly 59 transactions per block, and
  the published file has construction problems that we could correct only
  where the raw fields allowed (Section 3.3).
* The annotations are proxies. Script types and OP_RETURN protocols are exact
  properties but not behaviours; entity tags cover few transactions and are
  dominated by exchanges; the equal-output rule is a heuristic with known false
  positives (for example, exchange payouts of equal amounts).
* Twelve features describe a transaction only through counts, sizes, values,
  fees and two flags. Witness data, address reuse and the transaction graph are
  not used.
* The Elliptic features are anonymised and partly undocumented, and leakage
  between common splits has been reported [@Safar2026]. Its actors cannot be
  separated: the transaction–address graph is one connected component, so no split
  of that data is free of shared wallets, and we could only restrict the ranking
  side (Section 6.5).
* The external CoinJoin labels cover one implementation (Wasabi 2.x) and only the
  period after mid-2024, matching 175 test-period and 45 prospective transactions.
  They establish the recall of the two rules *against that list* and say nothing
  about their precision, about earlier CoinJoin software, or about CoinJoin
  activity in general; precision is estimated separately and is low (9.6% and
  7.8%, Section 6.4).
* Several choices (s = 1.5, K₀ = 30, 10% cap, depth 3, 160 micro-clusters,
  and ten proxy clusters) were fixed in advance rather than tuned, and the
  sensitivity analysis shows that two of them, the weight decay and the proxy
  size, change the results substantially. For all settings, Section 6.6 shows how results move
  when they change. Concentration depends strongly on K, and the AP lift of a
  common annotation is bounded by the inverse of its base rate, so values are
  comparable only for the same annotation and the same K.
* The test period is dominated by one new protocol (Runes: 40% of its
  transactions), so the transfer results describe this particular change.
* The prospective sample is one page of up to 25 consecutive transactions
  from each of 800 blocks per month, so it describes months well but single
  blocks only partly. Every planned page was in fact obtained, so there is no
  non-response to adjust for.
* The freeze is not blindness, and for part of this evaluation it is weaker than
  that. The code was frozen before the reported runs, but an earlier submitted
  version of this work had already analysed the same Bitcoin corpus *and the
  complete labelled Elliptic data set*. The Elliptic results of Section 6.5 and
  hypotheses H5 and H6 are therefore a frozen reanalysis carried out with prior
  knowledge of the outcomes, not blind confirmatory tests, and we do not present
  them as such (Section 5.1). Only the 2024–2026 sample, which had not been mined
  when the plan was frozen, supports genuinely prospective claims. The settings,
  the feature set and the choice of annotations were likewise made by people who
  had already seen how similar choices behaved on these transactions. What the
  freeze protects is the evaluation from being tuned to its own result; it does
  not make any of this a blind test. Ten of the twenty-four analyses reported here were
  specified in the plan and fourteen were added afterwards (Table S2);
  they are labelled throughout, and the two that overturned earlier conclusions
  are identified as such.

# 8. Conclusions

This study asked what rank-power feature weighting and size-constrained
refinement change when Bitcoin transactions are clustered, and whether the
resulting profiles are reproducible, transferable and meaningful. The answers
are mixed, and we think they are more useful for that.

The weights do not improve the profiles across the board. They change which
properties the profiles concentrate: legacy and script-hash spending and
OP_RETURN protocols gain, exchange tags and mixed-script inputs lose.
Refinement keeps any profile from absorbing a large part of the data it is
fitted on, but it hardly changes what the profiles concentrate and it does not
hold for later data. With these choices, all eleven
annotations that were never inputs are concentrated above their base rates in
2024, and refits of the whole pipeline reproduce the partition fairly well.

The profiles do not survive a large change in activity unchanged. When Runes
transactions appeared, the frozen model spread them over existing profiles,
and a refit on the new data produced different profiles, partly because the
feature ranking itself changed. In the prospective sample of 2024–2026,
three profiles absorbed three quarters of the transactions, so the partition
that the 2023 model describes is no longer the partition of the data. The
profiles still concentrated every annotation we could evaluate above its base
rate, so they keep some meaning even when their sizes go wrong. On the
Elliptic data the
profiles carried little information about illicit status on later time
steps, and atypicality scores pointed the wrong way. The count rule behind
the "coinjoin-like" flag of the published sample identified equal-output
CoinJoins correctly about one time in ten.

Three findings sharpen this. Read against the ceiling that each base rate sets, the
profiles capture most of what is attainable for common transaction properties and
almost none of it for rare ones. Weights computed from the annotations themselves do
not change that, and neither would better features: a supervised model on the same
twelve features, fitted a year earlier, attains 95.0% of the ceiling for P2SH inputs
where the profiles attain 23.6%, and over the ten annotations its median is 89.8%
against 16.5% for the profiles, rising to 96.4% when it is fitted within the period. The information is present and the clustering
objective does not isolate it. And when the model is refitted freely, the new profiles cannot be matched to the old
ones; but a refit that keeps the learned space and re-estimates only the centroids
follows 24 of the 31 profiles into 2024 and recovers most of the concentration lost by
2026, so the series can be continued after all.
Repeating the whole evaluation for seven clustering families shows which of
these results belong to ZSH and which to the task. ZSH reaches the highest
median share of the attainable concentration and concentrates legacy spending
far more strongly than the others, but no family approaches the ceiling for
rare annotations, every family's partition collapses into a few clusters two
years after the fit, and profiles survive a refit only where one cluster
already holds most of the data.

Two lines of work follow. First, for most of the annotations tested neither the
weights nor the features are the lever, so the useful direction is the objective: a criterion that optimises the
concentration of held-out properties directly, or a partition shaped by a few
labels and then applied unsupervised. Richer features earn their place only for properties of
the parties rather than of the transaction: adding address reuse and witness structure
to the twelve, where the full records allowed it, raised the supervised benchmark for exchange
tags from 40.8% to 59.6% and moved the OP_RETURN annotations far less. For
that annotation the features, not only the objective, are a binding limit.
Second, profiles that are refitted over time need a way to be matched to their
predecessors, so that a change in activity can be told apart from a change in the
model; constraining the refit to the previous centroids does this well enough to be
worth building on.

::: {custom-style="MDPI_6.2_back_matter"}

**Supplementary Materials:** The following supporting information can be
downloaded at https://www.mdpi.com/article/[DOI]/s1. Table S1: Count-rule flags of the published sample; Table S2: Analyses of this study, by when they were specified and what kind of evidence they provide; Table S3: The 31 ZSH profiles; Table S4: One-line description of each profile and the development transaction closest to its centroid; Table S5: Arms of the factorial design, all fitted on the full development period; Table S6: Primary contrasts of H1 (rank-power against uniform weights, A3 − A4) and H2 (refinement against none, A1 − A3) on the test period: differences in AP lift with 95% block-bootstrap intervals and Holm-adjusted p-values; Table S7: Share of each profile in every period with a 95% interval from resampling blocks (1,000 resamples; the prospective sample uses its design weights), and the mean cluster-wise Jaccard similarity across the 30 block-bootstrap refits of Section 6.3 with its 5th…; Table S8: Design-weighted against unweighted concentration in the prospective sample; Table S9: The count rule 'more than three inputs and more than three outputs', published as 'coinjoin-like', compared with the equal-output CoinJoin rule; Table S10: The two CoinJoin rules against an external list of Wasabi 2.x CoinJoin transactions with a known coordinator [@Svenda2026]; exploratory, added after the freeze; Table S11: Atypicality scores against illicit status on the Elliptic test steps and, as exploratory analyses, against annotations of the Bitcoin data; Table S12: Sensitivity of ZSH to its settings; Table S13: Sensitivity to the size of the proxy partition; exploratory, added after the freeze; Table S14: Leave-one-family-out analysis of the seeded variant: AP lift of each count-rule family on the test period for the unseeded model, the model seeded with all families, and the model seeded without that family; Figure S1: Rank-power weights of the twelve features on development data, with the mutual information (MI) of each feature with the proxy partition; Figure S2: Profile descriptors on development data; Figure S3: AP lift of the eleven non-input annotations on the test period for methods fitted on the same development sample with the same K (log scale); Figure S4: Primary contrasts of the factorial design on the test period: (left) rank-power against uniform weights without refinement (H1); (right) refinement against none with rank-power weights (H2); Figure S5: Monthly shares of the development-fitted profiles; Figure S6: ROC curves of atypicality scores against illicit status on the Elliptic test steps. The
supplementary file also gives the full definitions of the measures summarised
in Section 5.3.

**Author Contributions:** Conceptualization, S.D.K.; methodology, S.D.K.;
software, S.D.K.; validation, S.D.K.; formal analysis, S.D.K.; investigation,
S.D.K.; resources, S.D.K.; data curation, S.D.K.; writing—original draft
preparation, S.D.K.; writing—review and editing, S.D.K. and N.M.S.;
visualization, S.D.K.; supervision, N.M.S.; project administration, N.M.S. All
authors have read and agreed to the published version of the manuscript.

**Funding:** This research received no external funding.

**Institutional Review Board Statement:** Not applicable. The study analyses
public, pseudonymous blockchain records and a public benchmark data set; no
data about identified persons were collected.

**Informed Consent Statement:** Not applicable.

**Data Availability Statement:** The Bitcoin transaction sample is available
from IEEE DataPort (https://doi.org/10.21227/bxmt-mn56). The Elliptic data set
is available from Kaggle
(https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). The GraphSense
TagPacks are available at https://github.com/graphsense/graphsense-tagpacks;
the address-to-category map used here was built from the repository on
30 August 2026 and its SHA-256 checksum is listed in the analysis plan. The
code, the analysis plan written before the confirmatory analyses, the list of
deviations, the prospective-data collector and all result files are available
at https://github.com/sagarkorde/ZSH (branch `v2`; tags `v2-plan`,
`v2-frozen` and `v2-results`). The prospective transaction sample, the raw
API responses, the collection manifest and a checksum list are archived at
{ZENODO_DOI}. Every table and figure can be
regenerated with `python RUN_ALL.py`.

**Use of Generative AI:** The authors used Claude Opus 5 (Anthropic) throughout
this revision, under their direction and review. The tool was used to write and
test the analysis code in the accompanying repository, including the clustering
pipeline, the evaluation measures, the prospective-data collector and the
scripts that build every table and figure; to run those experiments and report
their output; to propose analyses that the authors then approved, among them the
reading of concentration against its ceiling, the oracle-weight bound, the
comparison across clustering families and the supervised benchmark of Section 6.8;
to check every reference against publisher records; and to draft and edit the
text of this article, its supplementary file and the response to the reviewers.
The authors specified the research questions, the analysis plan and the design,
reviewed all code, results and text, decided what to report, and take full
responsibility for the content of this publication. No text, figure or number
was published without author review. The 300 numerical claims in this article
and its supplementary file are checked against the saved result files by a
script in the repository (`experiments/verify_manuscript_numbers.py`).

**Conflicts of Interest:** The authors declare no conflicts of interest. The
first author is also an author of the transaction data set used in this study.
:::

# References

<!-- REFERENCES -->
