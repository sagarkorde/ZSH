# Supplementary Materials

**ZSH: Rank-Weighted Profiling of Bitcoin Transactions and a Pre-Specified Evaluation of What the
Profiles Capture**

Sagar D. Korde and Narendra M. Shekokar

This file holds the tables and figures that the article cites as Table S1, Table S2, ... and
Figure S1, Figure S2, ..., together with the full definitions of the measures summarised in
Section 5.3 of the article. Section, table and figure numbers without an S refer to the article.

## S1. Measures in full

Section 5.3 of the article defines the measures that carry the main results:
geometry, concentration against its ceiling, stability, transfer and the
sensitivity variants. The measures of the secondary analyses are given here in
full. Section, table and figure numbers without an S refer to the article.


**Checking the count rule.** To measure how well the "coinjoin-like" count rule
identifies CoinJoin transactions, we drew 5,000 transactions that meet the rule
and 5,000 that do not, at random from the whole sample, retrieved their
individual output values from the Esplora interfaces, and applied the
equal-output rule. The two strata are weighted by their population sizes to
estimate the precision and recall of the count rule and the prevalence of
equal-output transactions; intervals come from a parametric bootstrap of the
two stratum proportions. In the prospective sample the rule is compared with
the equal-output rule on every transaction, using the design weights.

**External CoinJoin labels.** For the transactions of the external list that fall in
our sample we report the recall of the count rule and of the equal-output rule, and
how the labelled CoinJoins are distributed over the profiles. With fewer than 200
labelled transactions per fold, the cross-fitted rule of the paragraph above cannot be
applied, so these results are descriptive.

**Elliptic.** ZSH, K-means++ and a diagonal Gaussian mixture at the same K,
rank-power K-means without refinement, and uniform weights with refinement are
fitted to all transactions of time steps 1–34 with all 165 features; ZSH is
also fitted with the 93 local features only. Clusters with fewer than 20
labelled training members are ranked last. For each test time step we report
the precision and recall of the smallest set of top-ranked clusters that covers
a quarter of the illicit training transactions. As a reference for what labels
make possible, a random forest [@Breiman2001] with 50 trees and 50 features per
split, the setting of the original study [@Weber2019], is trained on the
labelled transactions of steps 1–34.

**Atypicality.** The Isolation Forest of Section 4.7 is fitted on the Elliptic
training steps in the ZSH space and in the unweighted space. We add the local
outlier factor [@Breunig2000] (20 neighbours, fitted on 20,000 training
transactions) and the squared distance to the assigned ZSH centroid. Each score
is evaluated by ROC-AUC against the illicit label on the test steps. On the
Bitcoin data we evaluate, as exploratory analyses, how the development-period
Isolation Forest ranks Runes, exchange-tagged and P2WSH transactions of the test
period and equal-output CoinJoins and Runes of the prospective sample.

**Actor-disjoint evaluation.** With the actor annotations, the Elliptic analysis is
repeated with the cluster ranking restricted to training-period transactions that
share no address with any transaction of the evaluation period, so that the clusters
are ranked on actors that never appear in the evaluation set. Everything else is as
above. We also report whether a fully actor-disjoint split of the data is possible at
all.

**An upper bound for the weighting.** The weights are learned without labels. To see
how much a better ranking could be worth, we replace the ranking with one computed
from an annotation itself: on development data we estimate the mutual information
between each feature and the annotation on a class-balanced sample, and use it either
with the same rank-power curve or directly as weights. The pipeline is then refitted
with those fixed weights and evaluated on the test period. Such an oracle is not
available to an unsupervised method; it bounds what any weighting of these twelve
features could achieve for that annotation within this clustering scheme. We do this
for four annotations that span the range of attainment: P2PKH, P2SH and P2WSH inputs
and the exchange tag. Added after the freeze.

**Following profiles across refits.** If a model has to be refitted when activity
changes, the new profiles must be matched to the old ones or the series breaks. We
test this directly: each profile of the development model is paired with one cluster
of the refit by minimum-cost assignment on the transactions they share, and the pair
is scored by the Jaccard similarity of the two member sets. A profile counts as
followed when its best match reaches 0.5. This analysis was added after the freeze.

**The same evaluation for other clustering families.** To separate what belongs to
ZSH from what belongs to the task, the whole evaluation is repeated for the six
comparison methods of Table 4 at K = 31: concentration with its ceiling on
both later periods, ten block-bootstrap refits against the development partition,
a refit on each later period, and the profile matching described above. Each family
is fitted on the same development transactions, except that BIRCH and the partial
reproduction of Vlahavas et al. are fitted on a random subsample of 1,000,000 rows
and Ward on 30,000, because their cost grows faster than linearly. HDBSCAN is left
out because it chooses its own number of clusters and cannot assign new
transactions. Added after the freeze.

**Leaving rule families out of the seeds.** For the seeded variant we also
withhold one rule family at a time from the seeds; for the coinbase, OP_RETURN and
replace-by-fee families the corresponding flag is removed from the features
as well and the whole pipeline is refitted. All variants are evaluated on the
test period against the reference variant.

**A refit constrained to the previous centroids.** Section 6.3 refits the whole
pipeline on a later period, which re-estimates the scaler, the feature ranking and the
centroids together. The constrained refit changes only the last of these: it keeps the
scaler and the rank-power weights of the frozen model, so the space is unchanged, and
runs Lloyd's algorithm on the later period from the development centroids until the
largest centroid moves less than 1e-6 or 100 iterations are reached. Clusters that lose
every member keep their previous centroid. The resulting partition is compared with the
transferred one by the agreement measures of Section 5.3 and by the same minimum-cost
matching, and its concentration is measured with the same cross-fitted procedure.

**One partition for all annotations.** The supervised benchmark of Section 6.8 is fitted for
one annotation at a time, which a single partition cannot match for every annotation at
once. To price that constraint we build one partition with full label knowledge: for
each of the eleven annotations a gradient-boosted tree is cross-fitted by block parity
as above, the eleven out-of-fold scores of a transaction are standardised to form its
coordinates, and K-means cuts that eleven-dimensional space into K* clusters. The
partition is then ranked and scored exactly as any other. It remains an upper bound
rather than a method, because building it needs labels for every annotation.


**Design weights in the prospective sample.** The prospective sample was drawn with
unequal inclusion probabilities: blocks were selected within months, and one page of up
to 25 consecutive transactions was taken from each selected block, so a transaction in a
large block stands for more of the chain than one in a small block. Every prospective
estimate in the article that refers to a population quantity therefore uses the design
weights, and its interval comes from a block bootstrap stratified by calendar month,
which resamples blocks within each month rather than across the whole period and so
respects the way the sample was built. Section 5.4 states which quantities are weighted
and which are not.

Table S8 reports the same prospective concentration results both ways, so
that the effect of the weighting can be read directly rather than inferred. Two details
matter when comparing the columns. Target eligibility is deliberately unweighted, that
is it counts sampled positives, so both analyses cover exactly the same annotations and
the columns are comparable row by row. Ranking a cluster on the calibration half uses
Kish effective counts $(\sum w)^2 / \sum w^2$ rather than weighted totals, so a cluster
holding few sampled transactions with large weights cannot be promoted on the strength
of its weights alone; the evaluation half then uses the weighted totals, which is what
makes the reported precision and lift population estimates. With equal weights every
quantity reduces exactly to its unweighted form, and we verified this as a regression
check against the previously reported unweighted results.

## Supplementary tables

Table: **Table S1.** Count-rule flags of the published sample. Each flag is a fixed rule on the input and output counts; the rules can overlap. The last column gives the accuracy with which a depth-3 decision tree recovers each rule from the other candidate features on development data.

| Flag in the published sample | Definition | Development (%) | Test (%) | Recovered by depth-3 tree (%) |
|:-------------|:-------------|------------:|--------:|----------:|
| coinjoin-like | inputs > 3 and outputs > 3 | 1.6 | 2.2 | 100.00 |
| batch payment | inputs = 1 and outputs > 5 | 2.7 | 1.9 | 100.00 |
| consolidation | inputs > outputs and inputs > 2 | 7.4 | 4.9 | 99.62 |
| distribution | outputs > inputs and outputs > 2 | 8.5 | 10.6 | 99.56 |
| peer-to-peer | inputs = 1 and outputs = 1 | 40.5 | 19.6 | 100.00 |

Table: **Table S2.** Analyses of this study, by when they were specified and what kind of evidence they provide. All non-exploratory analyses had their hypotheses written into the analysis plan and their code frozen (repository tag `v2-frozen`) before the reported runs. They are nevertheless labelled *frozen reanalysis* rather than confirmatory, because an earlier submitted version of this work had already examined the same Bitcoin corpus and the complete labelled Elliptic data set: the freeze prevents the analysis from being tuned to its result, but the outcomes were not unknown to the authors. Rows marked *prospective on 2024-26* are additionally prospective when applied to the 2024-2026 sample, whose transactions had not been mined at the freeze. Exploratory analyses were added after the freeze; each is recorded with its reason and commit in the deviations log and labelled where it appears.

| Analysis | Section | Status | Reason it was added |
|:-------------------------------------------|:--------|:-------------|:-------------------------------------------|
| Primary model and profiles | 6.1 | Frozen reanalysis |  |
| Method comparison at matched K | 6.2 | Frozen reanalysis |  |
| Weighting × refinement factorial (H1, H2) | 6.2 | Frozen reanalysis |  |
| Stability: seeds, block bootstrap, permuted null (H3) | 6.3 | Frozen reanalysis |  |
| Transfer to later periods (H4) | 6.3 | Frozen reanalysis; prospective on 2024-26 |  |
| Concentration of non-input annotations (H5) | 6.4 | Frozen reanalysis; prospective on 2024-26 |  |
| Count rule against the equal-output rule | 6.4 | Frozen reanalysis; prospective on 2024-26 |  |
| Elliptic with a temporal split | 6.5 | Frozen reanalysis |  |
| Atypicality on Elliptic (H6) | 6.5 | Frozen reanalysis |  |
| Sensitivity to every fixed setting | 6.6 | Frozen reanalysis |  |
| Feature-ranking drift across refits | 6.3 | Exploratory | the test refit agreed poorly with the transferred partition |
| Atypicality on Bitcoin annotations | 6.5 | Exploratory | descriptive counterpart to H6 |
| Size of the proxy partition | 6.6 | Exploratory | a reviewer asked for sensitivity to this choice |
| Profile-share and Jaccard intervals | 6.3 | Exploratory | a reviewer asked for profile-support intervals |
| Actor-disjoint Elliptic evaluation | 6.5 | Exploratory | reviewers asked for source-level held-out evaluation |
| External CoinJoin labels | 6.4 | Exploratory | a reviewer asked for validation against external labels |
| Concentration read against its ceiling | 5.3, 6.4 | Exploratory | AP lift alone cannot separate a weak partition from a rare annotation |
| Oracle-weight upper bound | 6.6 | Exploratory | to separate a poor weighting from an uninformative feature set |
| Profile matching across refits | 6.3 | Exploratory | refitting is useful only if profiles can be followed |
| Refit constrained to the previous centroids | 6.3 | Exploratory | to test the remedy this article proposes |
| The same evaluation for seven families | 6.7 | Exploratory | to attribute the limits to ZSH or to the task |
| Supervised benchmark on the same features | 6.8 | Exploratory | to separate the feature set from the objective |
| One partition serving all annotations | 6.8 | Exploratory | to price the constraint of sharing a partition |
| Address-level features | 6.8 | Exploratory | to test whether richer features lift the weakest case |
| Design-weighted prospective estimates | 5.4, 6.4 | Exploratory | the prospective sample was drawn with unequal inclusion probabilities |

Table: **Table S3.** The 31 ZSH profiles. Dev., Test and Prosp.: share of the transactions of each period. In, Out, Value (total input value) and Fee rate (sat/vB) are medians over the development members; the main input script with its share, the share signalling replace-by-fee (RBF) and the share with an exchange tag also refer to development members. The last column gives the Runes share among the test-period members.

| Profile | Dev. (%) | Test (%) | Prosp. (%) | In | Out | Value (sat) | Fee rate | Main input script (%) | RBF (%) | Exch. tag (%) | Runes, test (%) |
|:--------|--------:|--------:|--------:|--------:|--------:|-------------:|--------:|-----------:|--------:|--------:|--------:|
| P00 | 8.7 | 1.9 | 4.4 | 1 | 1 | 1,145 | 6.1 | P2TR 99 | 96 | 0.0 | 1 |
| P01 | 7.7 | 5.7 | 0.7 | 1 | 2 | 2,011,790 | 22.0 | P2WPKH 93 | 39 | 0.4 | 33 |
| P02 | 7.3 | 3.9 | 0.0 | 1 | 1 | 9,680 | 61.0 | P2TR 100 | 79 | 0.0 | 0 |
| P03 | 7.1 | 4.7 | 0.3 | 1 | 1 | 2,841 | 16.3 | P2TR 99 | 90 | 0.0 | 0 |
| P04 | 6.9 | 4.9 | 0.9 | 2 | 2 | 1,441,620 | 22.0 | P2WPKH 59 | 35 | 0.2 | 2 |
| P05 | 6.3 | 9.7 | 3.4 | 1 | 2 | 1,284,683 | 11.9 | P2WPKH 89 | 39 | 0.1 | 63 |
| P06 | 6.0 | 6.3 | 0.6 | 1 | 2 | 1,398,670 | 49.0 | P2WPKH 74 | 43 | 0.2 | 12 |
| P07 | 5.5 | 12.0 | 2.2 | 1 | 1 | 343,614 | 21.0 | P2WPKH 87 | 43 | 0.0 | 73 |
| P08 | 4.6 | 1.9 | 1.0 | 1 | 2 | 1,381,350 | 23.0 | P2PKH 100 | 18 | 0.3 | 0 |
| P09 | 3.8 | 0.9 | 0.0 | 1 | 1 | 31,626 | 210.0 | P2TR 100 | 83 | 0.0 | 20 |
| P10 | 3.8 | 15.6 | 45.0 | 1 | 2 | 1,217,577 | 3.0 | P2WPKH 69 | 29 | 0.0 | 82 |
| P11 | 3.3 | 2.9 | 0.1 | 1 | 2 | 2,321,695 | 184.8 | P2WPKH 58 | 54 | 0.3 | 43 |
| P12 | 3.0 | 1.9 | 18.6 | 2 | 2 | 481,840 | 5.0 | P2WPKH 40 | 25 | 0.0 | 3 |
| P13 | 2.9 | 2.1 | 2.2 | 3 | 2 | 1,633,915 | 17.1 | P2WPKH 54 | 42 | 0.1 | 5 |
| P14 | 2.7 | 2.6 | 1.4 | 1 | 2 | 3,839,606 | 18.0 | P2SH 61 | 48 | 0.4 | 13 |
| P15 | 2.3 | 2.8 | 0.3 | 1 | 1 | 298,775 | 93.7 | P2WPKH 78 | 39 | 0.0 | 42 |
| P16 | 2.3 | 2.0 | 1.2 | 1 | 3 | 7,867,984 | 26.0 | P2PKH 50 | 37 | 3.3 | 10 |
| P17 | 2.2 | 1.8 | 1.9 | 4 | 2 | 1,630,564 | 20.7 | P2WPKH 51 | 47 | 0.3 | 3 |
| P18 | 2.1 | 1.3 | 0.1 | 2 | 2 | 1,824,000 | 140.0 | P2WPKH 50 | 45 | 0.5 | 7 |
| P19 | 2.0 | 1.1 | 0.1 | 1 | 3 | 3,535,655 | 143.0 | P2PKH 31 | 61 | 0.8 | 12 |
| P20 | 1.8 | 1.5 | 1.4 | 6 | 2 | 2,517,592 | 18.0 | P2WPKH 39 | 36 | 1.3 | 2 |
| P21 | 1.8 | 2.1 | 0.7 | 2 | 8 | 7,853,022 | 50.1 | P2TR 27 | 30 | 4.9 | 6 |
| P22 | 1.4 | 7.4 | 11.1 | 1 | 1 | 134,689 | 4.1 | P2WPKH 66 | 26 | 0.0 | 83 |
| P23 | 0.9 | 0.7 | 0.7 | 10 | 2 | 5,438,197 | 20.0 | P2WPKH 30 | 41 | 1.9 | 2 |
| P24 | 0.9 | 0.6 | 0.4 | 1 | 25 | 22,511,400 | 27.4 | P2TR 35 | 46 | 12.3 | 6 |
| P25 | 0.8 | 0.3 | 0.3 | 3 | 1 | 1,974,719 | 19.0 | P2PKH 85 | 15 | 3.7 | 0 |
| P26 | 0.6 | 0.5 | 0.4 | 22 | 2 | 9,738,762 | 16.0 | P2WPKH 25 | 43 | 3.4 | 3 |
| P27 | 0.6 | 0.4 | 0.3 | 63 | 1 | 57,021,165 | 13.5 | P2PKH 50 | 25 | 40.7 | 1 |
| P28 | 0.4 | 0.3 | 0.2 | 1 | 51 | 154,953,986 | 20.4 | P2WPKH 43 | 19 | 7.1 | 3 |
| P29 | 0.3 | 0.2 | 0.1 | 2 | 193 | 249,928,176 | 13.2 | P2WPKH 60 | 24 | 0.2 | 1 |
| P30 | 0.2 | 0.1 | 0.1 | 200 | 1 | 54,453,558 | 7.1 | P2TR 27 | 46 | 1.1 | 2 |

Table: **Table S4.** One-line description of each profile and the development transaction closest to its centroid. Descriptions are built from the medians and the annotation shares of each profile's members (Section 4.6); the share is of development transactions. Transaction ids are printed in two halves so that they break across lines.

| Profile | Share (%) | Description of its members | Transaction closest to the centroid |
|:--------|--------:|:-------------------------------------------|:-------------------------------------------|
| P00 | 8.7 | 1-in/1-out; P2TR 99%; value 1,145 sat; 6.1 sat/vB; OneInOneOut 100% | 95d7aea443947cf12711e755b494c614 7e3687220c7fa5355aecc5507e037fdd |
| P01 | 7.7 | 1-in/2-out; P2WPKH 93%; value 0.020 BTC; 22.0 sat/vB | 92abafcd27b1d905f6e27f97c113cad6 cae022f1164fc5aee02afea103da4419 |
| P02 | 7.3 | 1-in/1-out; P2TR 100%; value 9,680 sat; 61.0 sat/vB; OneInOneOut 98% | 3ce1b372f82b4c93c4f645a0c2d777cc 213a3e4eca8820c066232b1bc0bbed44 |
| P03 | 7.1 | 1-in/1-out; P2TR 99%; value 2,841 sat; 16.3 sat/vB; OneInOneOut 99% | eec5afdbc62fa4a6cb56cbf27ca33687 0ec08753510eb4cad9611f5df264ed53 |
| P04 | 6.9 | 2-in/2-out; P2WPKH 59%; value 0.014 BTC; 22.0 sat/vB | c561f3058b2b72538748cd4c29a2a48f 6266880b29d42001b6622f7ed42fd67d |
| P05 | 6.3 | 1-in/2-out; P2WPKH 89%; value 0.013 BTC; 11.9 sat/vB | 9a7c85967ca241f080ae35792fd158ad 815572b4820259b849e24318fd0c1900 |
| P06 | 6.0 | 1-in/2-out; P2WPKH 74%; value 0.014 BTC; 49.0 sat/vB | 96dea0ee7811846433767592189125cf 2b0b3a1759f1bd7c83b21f564bf2a26b |
| P07 | 5.5 | 1-in/1-out; P2WPKH 87%; value 343,614 sat; 21.0 sat/vB; OneInOneOut 100% | 6b60181934103bb1014bed36b5a47f3d 8baf803194be45f822537082c8ac234d |
| P08 | 4.6 | 1-in/2-out; P2PKH 100%; value 0.014 BTC; 23.0 sat/vB | ea67f35a76b5db7b84f3e1110833a0fd eacf83b01fc8d02bbc2fc1f589e1bad5 |
| P09 | 3.8 | 1-in/1-out; P2TR 100%; value 31,626 sat; 210.0 sat/vB; OneInOneOut 99% | 9d2d1252d84b7f0c2c048e7a5840b895 3d831c28cde98b56cfe9c9d18a1b093c |
| P10 | 3.8 | 1-in/2-out; P2WPKH 69%; value 0.012 BTC; 3.0 sat/vB | c29b24c9bc2a246bf3ab8946759485d6 84453395b1da43dc62b18d4a4db49ba0 |
| P11 | 3.3 | 1-in/2-out; P2WPKH 58%; value 0.023 BTC; 184.8 sat/vB | 8bea8bf4dd8454890c5a31a659302001 b35e9ae1e2737f0aba7a4fbed6ef3ab5 |
| P12 | 3.0 | 2-in/2-out; P2WPKH 40%; value 481,840 sat; 5.0 sat/vB | 761073f5bd2c530dbb899372ba650548 6abd0c8f8da6f56e493627adb301e428 |
| P13 | 2.9 | 3-in/2-out; P2WPKH 54%; value 0.016 BTC; 17.1 sat/vB; FanIn 59% | ab694f2c7ab6b438229a552b090139e0 deb49b958491196e03df92c20ce2f0ce |
| P14 | 2.7 | 1-in/2-out; P2SH 61%; value 0.038 BTC; 18.0 sat/vB | d05d508562642da9ec72403725c140d5 ecbe9d1342c0782e6f7cfd516099b03f |
| P15 | 2.3 | 1-in/1-out; P2WPKH 78%; value 298,775 sat; 93.7 sat/vB; OneInOneOut 99% | 10de1c5daa8cb1a0197e2a04b1f30506 0c671efc5a359240fd833d354c3ae149 |
| P16 | 2.3 | 1-in/3-out; P2PKH 50%; value 0.079 BTC; 26.0 sat/vB | 00f2930b103c1d4a292942db586f9407 0bb9b3b8422fca60eb28d698cdfa4eeb |
| P17 | 2.2 | 4-in/2-out; P2WPKH 51%; value 0.016 BTC; 20.7 sat/vB; FanIn 68% | a98ad4a1bfb4b53665d3fa9e9081bda7 92922cb8e5ac15c3a585a10424d8b552 |
| P18 | 2.1 | 2-in/2-out; P2WPKH 50%; value 0.018 BTC; 140.0 sat/vB | 8700fe52ce0dd98efcf7ac70bd339d9d d8d3e5d2e7089b1628da72adc9beef7a |
| P19 | 2.0 | 1-in/3-out; P2PKH 31%; value 0.035 BTC; 143.0 sat/vB; FanOut 55% | 18955a7d53c0eda490d3f71c9018e2ae 71864ef20831d2ef909d7f2f0ca37e47 |
| P20 | 1.8 | 6-in/2-out; P2WPKH 39%; value 0.025 BTC; 18.0 sat/vB; FanIn 71% | 9fe193c615c0b9d196cc47fd2f5d561e 09366f1c78535eadd0deaf76f7bbea40 |
| P21 | 1.8 | 2-in/8-out; P2TR 27%; value 0.079 BTC; 50.1 sat/vB | e7e7f92df1ac536d1472a3de14b296a9 1ce00f3bdf28c7039a736eea918dd29c |
| P22 | 1.4 | 1-in/1-out; P2WPKH 66%; value 134,689 sat; 4.1 sat/vB; OneInOneOut 100% | 844589b19b5b8cf70522574a090b2a0a f1416980c3624a4e4a8b416c7836469a |
| P23 | 0.9 | 10-in/2-out; P2WPKH 30%; value 0.054 BTC; 20.0 sat/vB; FanIn 75% | 657220c0fd5925602a0b53ee35fce0a3 0a1692643394d901e264c7eef114be8c |
| P24 | 0.9 | 1-in/25-out; P2TR 35%; value 0.225 BTC; 27.4 sat/vB; SingleInFanOut 78% | b8f7ca4bc2446aa569c4a8177144e9aa 6b681cff8aebd51dac6fd5523698a1e5 |
| P25 | 0.8 | 3-in/1-out; P2PKH 85%; value 0.020 BTC; 19.0 sat/vB; FanIn 91% | e0bdb9e3ddbc17fea9ea47cab8d558f2 1fe7f0e135bb290f7128220d76d39751 |
| P26 | 0.6 | 22-in/2-out; P2WPKH 25%; value 0.097 BTC; 16.0 sat/vB; FanIn 67% | cb6302a585c62f6223ebbf582f93ffb3 1386e4652b3869097e56283627976b61 |
| P27 | 0.6 | 63-in/1-out; P2PKH 50%; value 0.570 BTC; 13.5 sat/vB; FanIn 80% | bf3c4cb5928e2a90f9f31e8354c0b1b8 9b41af10e6629b360156c6a963d9c5c7 |
| P28 | 0.4 | 1-in/51-out; P2WPKH 43%; value 1.550 BTC; 20.4 sat/vB; SingleInFanOut 56% | df30f478d9a337fad40d0bd979a511db c952090e168910719f7d125b432c7f3b |
| P29 | 0.3 | 2-in/193-out; P2WPKH 60%; value 2.499 BTC; 13.2 sat/vB | c17fa5a7f1cbbcdeab3e6488e76cc584 0e02d115724b002467b65a1e66c80221 |
| P30 | 0.2 | 200-in/1-out; P2TR 27%; value 0.545 BTC; 7.1 sat/vB; FanIn 60% | 205307a76ee98751002e851ee6d7a5ee e24a168d648e462b5a592d0a796c36f9 |

Table: **Table S5.** Arms of the factorial design, all fitted on the full development period. K* = 31 is the number of ZSH profiles; A5 uses the K of A2. Silh.: Silhouette in the common unweighted space. Median AP lift over the eleven non-input annotations in the test period.

| Arm | Weights | Refinement | Initialisation | K | Largest cluster (%) | Silh. | Median AP lift |
|:--------|:-------------|:-----------|:-------------|--------:|--------:|--------:|--------:|
| A1 (ZSH) | rank-power | yes | hierarchical | 31 | 8.7 | 0.211 | 7.27 |
| A2 | uniform | yes | hierarchical | 30 | 9.6 | 0.268 | 7.01 |
| A3 | rank-power | no | hierarchical | 31 | 12.1 | 0.228 | 7.27 |
| A4 | uniform | no | hierarchical | 31 | 9.6 | 0.263 | 7.09 |
| A5 | uniform | no, K of A2 | hierarchical | 30 | 9.6 | 0.269 | 7.01 |
| A6 | MI-proportional | yes | hierarchical | 30 | 9.9 | 0.258 | 6.99 |
| A7 | Laplacian rank-power | yes | hierarchical | 31 | 9.6 | 0.223 | 4.55 |
| A9 | uniform | no | K-means++ (10 starts) | 31 | 10.0 | 0.279 | 5.60 |

Table: **Table S6.** Primary contrasts of H1 (rank-power against uniform weights, A3 − A4) and H2 (refinement against none, A1 − A3) on the test period: differences in AP lift with 95% block-bootstrap intervals and Holm-adjusted p-values. With 1,000 resamples and eleven annotations, the smallest attainable adjusted p-value is 0.022.

| Contrast | Annotation | Difference in AP lift | 95% CI | p (Holm) |
|:-----------|:-------------|-----------:|----------------:|--------:|
| H1: A3 − A4 | Coinbase | 10.01 | 8.56–11.54 | 0.022 |
| H1: A3 − A4 | P2PKH inputs | 6.75 | 6.62–6.88 | 0.022 |
| H1: A3 − A4 | P2SH inputs | 2.87 | 2.81–2.95 | 0.022 |
| H1: A3 − A4 | P2WPKH inputs | 0.30 | 0.29–0.30 | 0.022 |
| H1: A3 − A4 | P2WSH inputs | 0.19 | 0.09–0.28 | 0.022 |
| H1: A3 − A4 | P2TR inputs | 0.29 | 0.28–0.30 | 0.022 |
| H1: A3 − A4 | Mixed-script inputs | −2.59 | −2.68 to −2.52 | 0.022 |
| H1: A3 − A4 | Runes | −0.08 | −0.08 to −0.07 | 0.022 |
| H1: A3 − A4 | Omni | 9.72 | 6.82–11.93 | 0.022 |
| H1: A3 − A4 | Other OP_RETURN | 2.46 | 2.04–2.92 | 0.022 |
| H1: A3 − A4 | Exchange tag | −3.37 | −3.99 to −2.78 | 0.022 |
| H2: A1 − A3 | Coinbase | 0.20 | 0.17–0.23 | 0.022 |
| H2: A1 − A3 | P2PKH inputs | 0.05 | 0.04–0.06 | 0.022 |
| H2: A1 − A3 | P2SH inputs | −0.05 | −0.07 to −0.04 | 0.022 |
| H2: A1 − A3 | P2WPKH inputs | 0.00 | 0.00–0.00 | 0.022 |
| H2: A1 − A3 | P2WSH inputs | 0.00 | 0.00 to 0.01 | 0.695 |
| H2: A1 − A3 | P2TR inputs | 0.00 | 0.00–0.00 | 0.022 |
| H2: A1 − A3 | Mixed-script inputs | 0.02 | 0.00 to 0.03 | 0.360 |
| H2: A1 − A3 | Runes | 0.01 | 0.01–0.01 | 0.022 |
| H2: A1 − A3 | Omni | −0.17 | −0.47 to 0.01 | 0.288 |
| H2: A1 − A3 | Other OP_RETURN | 0.02 | 0.00–0.03 | 0.120 |
| H2: A1 − A3 | Exchange tag | 0.21 | −0.26 to 0.68 | 0.695 |

Table: **Table S7.** Share of each profile in every period with a 95% interval from resampling blocks (1,000 resamples; the prospective sample uses its design weights), and the mean cluster-wise Jaccard similarity across the 30 block-bootstrap refits of Section 6.3 with its 5th and 95th percentiles. Exploratory, added after the freeze.

| Profile | Development (%) | Test (%) | Prospective (%) | Jaccard (5th–95th) |
|:--------|-----------------:|-----------------:|-----------------:|-----------------:|
| P00 | 8.71 (8.46–8.96) | 1.92 (1.85–1.99) | 4.39 (4.17–4.63) | 0.97 (0.96–1.00) |
| P01 | 7.66 (7.57–7.75) | 5.70 (5.59–5.81) | 0.75 (0.66–0.84) | 0.64 (0.32–0.99) |
| P02 | 7.31 (7.12–7.48) | 3.88 (3.74–4.01) | 0.01 (0.01–0.02) | 0.95 (0.89–0.99) |
| P03 | 7.07 (6.87–7.27) | 4.66 (4.52–4.81) | 0.29 (0.23–0.37) | 0.95 (0.90–0.99) |
| P04 | 6.94 (6.88–7.00) | 4.89 (4.83–4.96) | 0.85 (0.78–0.92) | 0.91 (0.87–0.97) |
| P05 | 6.32 (6.24–6.40) | 9.69 (9.46–9.91) | 3.41 (3.21–3.61) | 0.69 (0.39–0.98) |
| P06 | 6.03 (5.94–6.11) | 6.32 (6.21–6.43) | 0.61 (0.52–0.71) | 0.73 (0.37–0.93) |
| P07 | 5.46 (5.39–5.53) | 11.98 (11.72–12.21) | 2.16 (1.98–2.35) | 0.71 (0.41–0.98) |
| P08 | 4.59 (4.55–4.63) | 1.88 (1.85–1.90) | 1.04 (0.97–1.11) | 0.93 (0.91–0.98) |
| P09 | 3.82 (3.67–3.97) | 0.91 (0.84–0.99) | 0.00 (0.00–0.01) | 0.97 (0.94–0.99) |
| P10 | 3.81 (3.75–3.88) | 15.58 (15.20–15.93) | 44.97 (44.31–45.58) | 0.86 (0.61–0.99) |
| P11 | 3.29 (3.21–3.36) | 2.88 (2.77–3.00) | 0.09 (0.06–0.12) | 0.68 (0.57–0.77) |
| P12 | 3.00 (2.95–3.05) | 1.94 (1.89–1.98) | 18.60 (18.13–19.10) | 0.93 (0.87–0.97) |
| P13 | 2.85 (2.82–2.88) | 2.10 (2.07–2.13) | 2.23 (2.16–2.31) | 0.75 (0.56–0.88) |
| P14 | 2.70 (2.68–2.73) | 2.62 (2.58–2.66) | 1.38 (1.31–1.46) | 0.69 (0.12–0.95) |
| P15 | 2.32 (2.28–2.36) | 2.80 (2.72–2.89) | 0.25 (0.19–0.32) | 0.60 (0.23–0.90) |
| P16 | 2.31 (2.28–2.33) | 2.05 (2.02–2.08) | 1.19 (1.14–1.25) | 0.82 (0.77–0.87) |
| P17 | 2.15 (2.12–2.18) | 1.81 (1.78–1.84) | 1.87 (1.80–1.95) | 0.54 (0.25–0.80) |
| P18 | 2.08 (2.05–2.12) | 1.35 (1.31–1.38) | 0.07 (0.06–0.09) | 0.57 (0.44–0.76) |
| P19 | 1.95 (1.90–2.00) | 1.11 (1.07–1.14) | 0.06 (0.05–0.08) | 0.45 (0.19–0.70) |
| P20 | 1.80 (1.78–1.82) | 1.54 (1.51–1.56) | 1.37 (1.31–1.43) | 0.61 (0.33–0.85) |
| P21 | 1.79 (1.77–1.81) | 2.09 (2.06–2.12) | 0.69 (0.65–0.73) | 0.66 (0.55–0.89) |
| P22 | 1.42 (1.39–1.46) | 7.36 (7.17–7.57) | 11.15 (10.81–11.47) | 0.54 (0.21–0.99) |
| P23 | 0.92 (0.91–0.94) | 0.72 (0.71–0.74) | 0.74 (0.70–0.78) | 0.67 (0.36–0.83) |
| P24 | 0.91 (0.90–0.93) | 0.58 (0.57–0.59) | 0.45 (0.41–0.48) | 0.88 (0.73–0.95) |
| P25 | 0.79 (0.78–0.80) | 0.25 (0.25–0.26) | 0.29 (0.27–0.32) | 0.37 (0.16–0.85) |
| P26 | 0.60 (0.59–0.61) | 0.46 (0.45–0.47) | 0.36 (0.34–0.39) | 0.56 (0.39–0.80) |
| P27 | 0.56 (0.55–0.57) | 0.35 (0.34–0.36) | 0.29 (0.26–0.32) | 0.65 (0.43–0.89) |
| P28 | 0.36 (0.35–0.36) | 0.26 (0.25–0.27) | 0.17 (0.15–0.19) | 0.76 (0.21–0.94) |
| P29 | 0.26 (0.26–0.27) | 0.21 (0.20–0.21) | 0.12 (0.10–0.13) | 0.90 (0.83–0.97) |
| P30 | 0.20 (0.19–0.20) | 0.12 (0.11–0.12) | 0.13 (0.11–0.14) | 0.80 (0.59–0.97) |

Table: **Table S8.** Design-weighted against unweighted concentration in the prospective sample. The weighted columns are the estimates reported in the article: they use the design weights of the sampling scheme and a month-stratified block bootstrap, and they estimate the quantity for the sampled population. The unweighted columns treat the sampled transactions as the population and are therefore sample-specific; they are given so that the effect of the weighting can be read directly. Positives are sampled counts and are identical for both, because target eligibility is deliberately unweighted. Attained: average precision, the share of the ceiling 1/π.

| Annotation | Positives | Base wtd (%) | Base unwtd (%) | AP lift wtd (95% CI) | AP lift unwtd (95% CI) | Attained wtd (%) | Attained unwtd (%) | Prec@25% wtd | Prec@25% unwtd |
|:-------------------|----------:|--------:|--------:|-----------------:|-----------------:|---------:|---------:|---------:|---------:|
| P2PKH inputs | 25,190 | 4.595 | 5.521 | 9.31 (8.85–9.72) | 7.66 (7.34–7.96) | 42.8 | 42.3 | 97.1 | 97.0 |
| P2SH inputs | 16,956 | 3.158 | 3.716 | 3.41 (3.22–3.62) | 3.01 (2.83–3.18) | 10.8 | 11.2 | 15.9 | 16.0 |
| P2WPKH inputs | 275,948 | 62.457 | 60.476 | 1.32 (1.31–1.33) | 1.31 (1.30–1.31) | 82.3 | 79.0 | 85.0 | 81.3 |
| P2WSH inputs | 11,386 | 2.159 | 2.495 | 3.29 (3.08–3.47) | 3.22 (3.08–3.36) | 7.1 | 8.0 | 7.1 | 8.2 |
| P2TR inputs | 116,926 | 25.414 | 25.625 | 2.45 (2.41–2.49) | 2.36 (2.33–2.40) | 62.3 | 60.6 | 65.7 | 62.7 |
| Mixed-script inputs | 6,935 | 1.327 | 1.520 | 7.88 (7.26–8.27) | 6.76 (6.31–7.12) | 10.5 | 10.3 | 12.5 | 12.3 |
| Runes | 137,087 | 39.197 | 30.043 | 1.35 (1.34–1.37) | 1.39 (1.38–1.41) | 53.0 | 41.9 | 54.7 | 43.2 |
| Other OP_RETURN | 9,245 | 1.761 | 2.026 | 3.34 (2.97–3.79) | 3.15 (2.88–3.43) | 5.9 | 6.4 | 4.7 | 5.6 |
| Exchange tag | 1,936 | 0.332 | 0.424 | 29.60 (25.29–33.41) | 23.75 (20.24–26.73) | 9.8 | 10.1 | 14.7 | 14.3 |

Table: **Table S9.** The count rule 'more than three inputs and more than three outputs', published as 'coinjoin-like', compared with the equal-output CoinJoin rule. For 2022–2024, estimates combine random samples of flagged and unflagged transactions, weighted by stratum size, with parametric bootstrap intervals. For the prospective sample they use all transactions and the design weights, with intervals from resampling blocks within months.

| Data | Quantity | Estimate, % (95% CI) |
|:-------------|:-------------|-----------------:|
| 2022–2024 sample (stratified, n = 5,000 + 5,000) | Precision of the count rule | 9.6 (8.8–10.5) |
|  | Recall of the count rule | 90.2 (74.4–100.0) |
|  | Prevalence of equal-output CoinJoins | 0.20 (0.17–0.25) |
|  | Transactions with a GraphSense coinjoin tag | 0 |
| Prospective sample (all 456,292 transactions, weighted) | Precision of the count rule | 7.8 (6.8–8.8) |
|  | Recall of the count rule | 97.0 (95.0–98.6) |
|  | Prevalence of equal-output CoinJoins | 0.08 (0.07–0.09) |
|  | Share meeting the count rule | 0.96 (0.90–1.01) |
|  | Transactions with a GraphSense coinjoin tag | 0 |

Table: **Table S10.** The two CoinJoin rules against an external list of Wasabi 2.x CoinJoin transactions with a known coordinator [@Svenda2026]; exploratory, added after the freeze. Recall is the share of the listed transactions that the rule flags, with a Wilson interval; the equal-output rule needs individual output values and can therefore be evaluated only on the prospective sample. The list covers coordinators active after mid-2024, which is why it matches no development transaction.

| Period | Transactions | On the external list | Count rule recall (%) | Equal-output recall (%) |
|:-----------|-------------:|---------:|-----------------:|-----------------:|
| Development | 3,299,616 | 0 |  |  |
| Test | 2,584,530 | 175 | 98.9 (95.9–99.7) |  |
| Prospective | 456,292 | 45 | 100.0 (92.1–100.0) | 97.8 (88.4–99.6) |

Table: **Table S11.** Atypicality scores against illicit status on the Elliptic test steps and, as exploratory analyses, against annotations of the Bitcoin data. A ROC-AUC below 0.5 means that positives receive lower scores. H6 is supported when the upper 95% bound is at most 0.5. Intervals from 1,000 resamples of time steps (Elliptic) or 200 resamples of blocks (Bitcoin).

| Data | Score | Target | Positives | ROC-AUC (95% CI) | PR-AUC | Base (%) | Reading |
|:-------------|:-------------|:-------------|-----------:|-----------------:|--------:|--------:|:------------|
| Elliptic test steps | Isolation Forest, ZSH space | illicit | 1,083 | 0.175 (0.118–0.278) | 0.037 | 6.5 | H6 supported |
| Elliptic test steps | Isolation Forest, unweighted | illicit | 1,083 | 0.182 (0.121–0.297) | 0.037 | 6.5 | H6 supported |
| Elliptic test steps | Local outlier factor, ZSH space | illicit | 1,083 | 0.508 (0.454–0.555) | 0.060 | 6.5 | inconclusive |
| Elliptic test steps | Distance to ZSH centroid | illicit | 1,083 | 0.269 (0.199–0.373) | 0.041 | 6.5 | H6 supported |
| Bitcoin test | Isolation Forest, ZSH space | Runes | 1,041,406 | 0.926 (0.926–0.927) | 0.779 | 40.3 | exploratory |
| Bitcoin test | Isolation Forest, ZSH space | exchange tag | 8,649 | 0.728 (0.723–0.734) | 0.023 | 0.3 | exploratory |
| Bitcoin test | Isolation Forest, ZSH space | P2WSH inputs | 36,881 | 0.343 (0.339–0.346) | 0.010 | 1.4 | exploratory |
| Bitcoin prospective | Isolation Forest, ZSH space | equal-output CoinJoin | 388 | 0.907 (0.892–0.918) | 0.035 | 0.1 | exploratory |
| Bitcoin prospective | Isolation Forest, ZSH space | Runes | 137,087 | 0.952 (0.951–0.954) | 0.798 | 30.0 | exploratory |

Table: **Table S12.** Sensitivity of ZSH to its settings. Each variant was fitted on the same 1,000,000 development transactions with one setting changed and evaluated on the test period. The last column counts non-input annotations whose AP lift is significantly higher or lower than for the reference (Holm-adjusted p < 0.05, 95% interval excluding zero); with 2.6 million test transactions, even small differences are significant. Silhouettes are estimated from random draws, so identical partitions can differ in the third decimal.

| Variant | K | Largest, fit (%) | Largest, test (%) | Silh. | Median lift | Higher / lower |
|:-------------|--------:|---------:|---------:|--------:|--------:|--------:|
| Reference (primary settings) | 31 | 8.9 | 18.2 | 0.211 | 7.05 |  |
| s = 0.5 | 31 | 9.4 | 23.3 | 0.257 | 7.16 | 4 / 7 |
| s = 1 | 31 | 9.4 | 22.8 | 0.237 | 7.49 | 5 / 6 |
| s = 2 | 32 | 8.4 | 18.9 | 0.168 | 7.43 | 7 / 2 |
| s = 3 | 33 | 9.0 | 20.2 | 0.119 | 8.44 | 9 / 2 |
| K₀ = 10 | 20 | 8.8 | 21.5 | 0.192 | 6.68 | 2 / 8 |
| K₀ = 20 | 22 | 10.8 | 26.2 | 0.219 | 6.59 | 2 / 8 |
| K₀ = 40 | 40 | 8.9 | 14.4 | 0.232 | 8.19 | 11 / 0 |
| K₀ = 60 | 60 | 8.4 | 12.3 | 0.204 | 8.91 | 11 / 0 |
| c = 5% | 45 | 5.1 | 15.5 | 0.149 | 8.52 | 11 / 0 |
| c = 15% | 30 | 13.5 | 20.6 | 0.234 | 7.05 | 1 / 9 |
| c = 20% | 30 | 13.5 | 20.6 | 0.234 | 7.05 | 1 / 9 |
| no refinement | 30 | 13.5 | 20.6 | 0.234 | 7.05 | 1 / 9 |
| depth 6 | 31 | 8.9 | 18.2 | 0.214 | 7.05 | 0 / 0 |
| upsampled corpus | 39 | 8.9 | 18.9 | −0.020 | 3.78 | 1 / 9 |
| sample weights | 39 | 8.5 | 19.8 | −0.012 | 3.55 | 2 / 9 |
| K-means++ start | 31 | 8.8 | 15.9 | 0.211 | 7.80 | 7 / 2 |
| semantic seeds | 31 | 7.6 | 15.9 | 0.198 | 7.27 | 7 / 4 |
| seed–Ward blend | 31 | 9.6 | 19.7 | 0.225 | 7.99 | 9 / 2 |

Table: **Table S13.** Sensitivity to the size of the proxy partition; exploratory, added after the freeze. Each row refits the whole pipeline on the same 1,000,000 development transactions with the given number of proxy clusters and is evaluated on the test period. τ: Kendall correlation of the resulting feature ranking with the reference ranking. Silh.: Silhouette in the common unweighted space. Median lift: median AP lift over the eleven non-input annotations; P2PKH and Omni are given as the annotations that move most. The last column counts annotations that are significantly higher or lower than the reference (Holm-adjusted p < 0.05).

| Proxy clusters | K | τ | Top-ranked feature | Silh. | Median lift | P2PKH | Omni | Higher / lower |
|:-------------|--------:|--------:|:-------------|--------:|--------:|--------:|--------:|--------:|
| 5 | 33 | 0.36 | Virtual size | 0.153 | 8.89 | 26.1 | 34.8 | 8 / 3 |
| 10 (reference) | 31 | 1.00 | Serialised size | 0.210 | 7.05 | 20.4 | 29.0 |  |
| 20 | 31 | 0.88 | Virtual size | 0.216 | 7.52 | 14.7 | 35.5 | 6 / 5 |
| 50 | 31 | 0.76 | Fee rate per byte | 0.173 | 6.28 | 9.2 | 7.5 | 2 / 9 |

Table: **Table S14.** Leave-one-family-out analysis of the seeded variant: AP lift of each count-rule family on the test period for the unseeded model, the model seeded with all families, and the model seeded without that family. For the OP_RETURN and replace-by-fee families the corresponding flag was also removed from the features and the whole pipeline was refitted.

| Rule family | Feature removed | Unseeded | All seeds | Family withheld | Withheld − all seeds |
|:-------------|:-------------|---------:|--------:|---------:|---------:|
| Coinbase | none (not an input) | 23.38 | 23.84 | 23.84 | 0.00 |
| Many inputs and outputs | none | 22.10 | 21.20 | 19.06 | −2.14 |
| Single input, fan-out | none | 22.42 | 21.10 | 19.29 | −1.81 |
| Fan-in | none | 13.92 | 13.66 | 13.90 | 0.24 |
| Fan-out | none | 4.14 | 5.45 | 4.95 | −0.51 |
| One input, one output | none | 4.14 | 4.26 | 4.14 | −0.13 |
| OP_RETURN | OP_RETURN output (0/1) | 2.12 | 2.07 | 2.10 | 0.03 |
| Replace-by-fee | Replace-by-fee signalled (0/1) | 2.90 | 2.94 | 2.82 | −0.12 |

## Supplementary figures

![**Figure S1.** Rank-power weights of the twelve features on development data, with the mutual information (MI) of each feature with the proxy partition.](figs/F3_weights.png){width=3.6in}

![**Figure S2.** Profile descriptors on development data. Left block: medians of five features, log-transformed and rescaled to [0, 1] across profiles. Middle block: shares of members with each input script class, an OP_RETURN output, replace-by-fee signalling and an exchange tag. Right column: Runes share among the test-period members.](figs/F4_profiles.png){width=6.5in}

![**Figure S3.** AP lift of the eleven non-input annotations on the test period for methods fitted on the same development sample with the same K (log scale). Bars are 95% block bootstrap intervals for ZSH and K-means++; grey points are the other baselines.](figs/F5_methods.png){width=6.5in}

![**Figure S4.** Primary contrasts of the factorial design on the test period: (left) rank-power against uniform weights without refinement (H1); (right) refinement against none with rank-power weights (H2). Differences in AP lift with 95% intervals; filled points are significant after Holm adjustment.](figs/F6_factorial.png){width=6.5in}

![**Figure S5.** Monthly shares of the development-fitted profiles. Test and prospective transactions were assigned by the frozen model; prospective shares use the design weights. Vertical lines mark the start of the test period, the Runes launch and the start of the prospective sample.](figs/F8_drift.png){width=6.5in}

![**Figure S6.** ROC curves of atypicality scores against illicit status on the Elliptic test steps. Curves below the diagonal mean that illicit transactions receive lower scores.](figs/F11_atypicality.png){width=3.6in}

# References

<!-- REFERENCES -->
