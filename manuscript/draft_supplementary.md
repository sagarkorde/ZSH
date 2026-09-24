# Draft — Supplementary Materials

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
comparison methods of Table {T_methods} at K = 31: concentration with its ceiling on
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

Table {T_weighting} reports the same prospective concentration results both ways, so
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
