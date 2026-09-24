# Draft — Results (merged; {FUTURE} parts pending)

## 6. Results

### 6.1. The fitted profiles

On the development data the rank-power weights are dominated by transaction
size: serialised size receives 0.489 of the total weight and virtual size
0.173, followed by the fee (0.094) and the two fee rates (0.061 and 0.044)
(Figure {F_weights}). The OP_RETURN and replace-by-fee flags, whose mutual
information with the proxy partition is close to zero, receive the smallest
weights (0.012 and 0.013). Because the proxy partition is a K-means solution on
the same data, this ranking reflects which features already separate
transactions most strongly; it does not reflect relevance to any behavioural
question.

ZSH produced 31 profiles. One of the 30 initial clusters held 12.1% of the
development transactions and was split in two; after the final reassignment
the largest profile holds 8.7% (Table {T_profiles}, Figure {F_profiles}). Fitting
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
classes, tags) and from medians of the inputs. Table {T_representatives} gives
a one-line description of every profile together with the transaction closest
to its centroid. None of the profiles is
specific to OP_RETURN use: the highest OP_RETURN share in development data is
22.9% (P19).

### 6.2. RQ1 — What weighting and refinement change

**Comparison with standard methods.** Table {T_methods} compares ZSH with six
methods fitted on the same 200,000 development transactions with the same
number of clusters (K = 31). In the common unweighted space ZSH has the worst
geometry among the K-means-type methods: Silhouette 0.207 against 0.285 for
K-means++, 0.269 for mini-batch K-means and 0.266 for Ward clustering, and a
Davies–Bouldin index of 1.92 against 1.19. In its own weighted space its
Silhouette is 0.302. The Gaussian mixture and BIRCH reach 0.156 and 0.182, and
HDBSCAN labels 44% of its sample as noise. The partial reproduction of
Vlahavas et al. has the highest Silhouette in its own three-dimensional space
(0.764 with k = 5) but among the lowest concentrations for most annotations.

On the test data (Figure {F_methods}), ZSH concentrates eight of the eleven
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
development data (Table {T_factorial}, Figure {F_factorial}). With the
initialisation and K = 31 fixed and no refinement, rank-power weights changed
the concentration of all eleven non-input annotations (H1; all Holm-adjusted
p < 0.05; Table {T_contrasts}). The effect has both signs. It raised the AP lift for coinbase
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

### 6.3. RQ2 — Reproducibility and transfer

**Refitting (H3).** Table {T_stability} and Figure {F_stability} summarise 40 refits
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

These numbers need a reference. On data whose columns had been permuted
independently, so that no joint structure remained, ten seed-only refits of
ZSH still agreed at a mean pairwise ARI of 0.61 (s.d. 0.14). The corresponding
value on the real data is 0.81. Part of the reproducibility of any K-means
partition on these features therefore comes from the marginal distributions
alone (heavy tails and a few discrete values); the joint structure adds a
clear but not overwhelming amount.

How precisely is each profile's size known? Resampling blocks gives an interval
for every profile's share of each period (Table {T_support}). The intervals are
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
fell below 0.5 (Figure {F_stability}b). Following Hennig's rule of thumb
[@Hennig2007], the first group can be treated as stable, the second as
indicating a pattern whose boundaries move, and the last as not reliable.
K-means++ had 12, 14 and 5 clusters in these groups. Centroids moved less for
ZSH than for K-means++ (mean matched displacement 0.30 against 0.36 of the
within-cluster root-mean-square distance).

**Transfer to later periods (H4).** The profiles were fitted once on 2022–2023 data
and then used to assign 2024 and 2025–2026 transactions. Figure {F_drift} shows
their monthly shares. Activity was not stationary even within the development
period: the three small-value taproot profiles (P00, P02, P03) together held 2%
of the transactions in March 2023, 32% in May and 48% in August, and they fell
back in 2024. After the Runes protocol launched in April 2024, P05 and P07 grew sharply
in May and June 2024, and P10 and P22 became the largest profiles from July
2024.

Most transferred profiles kept their structural character. Their median input
counts, output counts and fee rates in 2024 stayed close to the development
values, and for 18 of the 31 profiles the Jensen–Shannon distance between the
development and 2024 input-script mixes was below 0.2 (Table {T_transfer}). The
exceptions are the profiles that absorbed Runes: in 2024, Runes made up 82% of
P10, 83% of P22, 73% of P07 and 63% of P05, and their count-rule mix changed
accordingly (Jensen–Shannon distance 0.65–0.80). The frozen model had no
profile for this activity, so it placed Runes transactions with the structurally
closest existing groups: four profiles hold 80% of them, but none of these
profiles consists of Runes alone.

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
2026 (Figure {F_drift}). These three profiles are the low-fee-rate members of
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
(Table {T_matching}). Matching each of the 31 profiles to one cluster of the refit,
no profile reaches a Jaccard similarity of 0.5 in either period; the median best
match is 0.06 for the 2024 refit and 0.01 for the prospective refit, and the best
single case is 0.37 and 0.47. K-means++ does better in the near term — four of its
clusters, holding 41% of the test-period transactions, exceed 0.5 — but by 2026 only
one does, holding 0.3% of transactions. A refit therefore produces a new map rather
than an updated one, which is a practical limit on using these profiles for
monitoring: either the frozen assignment is kept and its drift accepted, or the
series of profile shares starts again at each refit.

**Unless the refit is constrained.** *This analysis is exploratory: it was added
after the analysis freeze, and it is not part of the pre-specified evidence for any
hypothesis (Table {T_prespec}).* The refits above start from scratch: they
re-estimate the scaler, the feature ranking and the centroids, so both the space and
the partition move at once. A refit that keeps what the model has already learned
about the space and re-estimates only the centroids, starting Lloyd's algorithm from
the development centroids, behaves very differently (Table {T_warm}). In the test
period 24 of the 31 profiles can then be followed above a Jaccard similarity of 0.5,
holding 83.8% of the transactions, against none for the free refit, and the agreement
with the transferred partition rises from an ARI of 0.21 to 0.63. Two years out, 14
of 31 can be followed, holding 27.0% of the prospective transactions. The centroids
do move — 54 Lloyd iterations in the test period and 75 in the prospective sample,
with a mean displacement of 0.23 and 0.62 — so this is a genuine refit and not a
frozen model under another name.

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

### 6.4. RQ3 — What the profiles concentrate

Table {T_concentration} lists the cross-fitted concentration of the eleven
non-input annotations in the test period, and Figure {F_concentration} shows
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
unweighted, sample-specific versions are in Table {T_weighting}. The strength is
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
{T_heuristic}). Of the flagged transactions, 482 (9.6%, 95% CI 8.8–10.5%)
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
from the source instead (Table {T_cjsource}). It covers coordinators active after
mid-2024, so it matches none of our development transactions, 175 of the test period
and 45 of the prospective sample. On these, the count rule has a recall of 98.9%
(95% CI 95.9–99.7%) in the test period and 100% (92.1–100%) in the prospective
sample, and the equal-output rule, which can be evaluated only where individual
output values are available, has a recall of 97.8% (88.4–99.6%). These are recall figures, and only recall figures:
both rules find nearly all of the *listed* CoinJoins. They say nothing about
precision, which is a separate question answered by a separate evaluation — the
equal-output criterion of Table {T_heuristic}, where the count rule's precision is
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

### 6.5. RQ4 — Illicit activity (Elliptic)

**Concentration.** On Elliptic, ZSH was fitted to all 136,265 transactions of
time steps 1–34. The weights concentrated on a few anonymised features (the
largest weight was 0.41) and the refinement could not break up the dominant
cluster within three levels: 64% of the training transactions remain in one of
the 52 profiles. Clusters were ranked by their illicit rate among the 29,894
labelled training transactions and scored on the 16,670 labelled test
transactions, of which 6.5% are illicit (Table {T_elliptic}, Figure
{F_elliptic}).

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
(Figure {F_elliptic}b), the top ZSH clusters caught 78–98% of the illicit
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

**Atypicality.** Table {T_atypicality} and Figure {F_atypicality} test whether
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

### 6.6. Sensitivity

Table {T_sensitivity} and Figure {F_sensitivity} compare variants of the
pipeline, each fitted on the same 1,000,000 development transactions and
evaluated on the test period. The reference variant uses the settings of the
primary model. It also has 31 profiles, but some of its concentration values
differ from those of the model fitted on all development data (P2PKH 20.4
against 23.3, P2SH 2.7 against 5.4), a first sign that some annotations react
strongly to small changes in the fit.

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
partition of ten clusters. Table {T_proxy_k} refits the pipeline with 5, 20 and
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

**How much could better weights buy?** Table {T_oracle} answers this with weights
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

**Leaving families out of the seeds.** Table {T_loo} tests the seeded variant further. If the seeds carry information about a
rule family, removing that family from the seeds should lower its
concentration. For the many-input-many-output and single-input-fan-out rules
this happened (AP lift 19.1 and 19.3 against 21.2 and 21.1 with all seeds), but
seeding with all families was already below the unseeded model (22.1 and
22.4). For fan-out payments seeding helped (5.5 against 4.1), and part of the
gain remained when the family was withheld (4.9). For the other families the
differences were below 0.3. Because the rule families are functions of the
input counts, these results describe how seeds move cluster boundaries; they
do not show that the profiles capture behaviour.

### 6.7. The same evaluation for seven clustering families

**What was done.** *This whole section is exploratory: the seven-family comparison
was added after the analysis freeze (Table {T_prespec}).* Sections 6.2 to 6.6 compare
ZSH with K-means++ at the same K. The limits they report — concentration bounded by each base rate, weights
that cannot lift it, profiles that do not survive a refit — are therefore
statements about ZSH. To see which of them are statements about the task, the
whole evaluation was repeated for seven families: ZSH, K-means++, mini-batch
K-means, a diagonal Gaussian mixture, BIRCH, Ward clustering and the partial
reproduction of Vlahavas et al. [@Vlahavas2024], all with the same K = 31 and
the same development transactions (Table {T_bench_battery}). Six of the seven
use the same twelve features; the Vlahavas reproduction uses the five features
of that study, as in Section 6.2, since reproducing it means reproducing its
feature set. BIRCH and the Vlahavas pipeline were fitted on a
random subsample of 1,000,000 transactions because their cost grows faster than
linearly in the number of rows, and Ward on its own 30,000-row subsample as in
Section 6.2; the other four use all 3,299,616.

**Concentration against the ceiling.** Table {T_bench_attained} and
Figure {F_bench} give the attained share of every family for every annotation
on the test period. Read across a row and the families differ; read down a
column and the annotation decides. For the three most frequent annotations every
family attains between 45.6% and 97.0% of the ceiling. For coinbase
transactions no family passes 2.5%, for Omni 0.9%, for exchange tags 10.3%,
for other OP_RETURN use 9.8% and for P2WSH inputs 20.1%. Seven families that
build clusters in different ways — around centroids, by
merging a hierarchy and by summarising a feature tree — fail on the same
annotations, which is what Section 6.4 implies: these annotations are too rare
for 31 clusters of these features to isolate, whatever builds the clusters.

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
{T_matching}, which used the seed of Section 6.3, left one prospective profile
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

### 6.8. Is the limit the features or the objective?

*This section is exploratory: the supervised benchmark and the address-level
comparison were both added after the analysis freeze (Table {T_prespec}).*

Section 6.6 shows that weights computed from an annotation itself add little, and
Section 6.7 that six other clustering families reach the same low shares for rare
annotations. Two explanations remain: the twelve features do not carry the
annotation, or they carry it and the clustering does not find it. Replacing the
clustering with a supervised model on exactly the same twelve features separates
them (Table {T_ceiling}). A gradient-boosted tree is fitted for one annotation at a
time and its score is cut into 31 cells, which are then ranked and scored by the same
cross-fitted procedure as the profiles, so the columns are comparable.

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
0.6% to 88.9%. Table {T_ceiling} covers ten annotations rather than the eleven used elsewhere:
Runes is a test-period protocol and has no development-period positives, so the
transfer column cannot be computed for it. The median over those ten is 16.5% for the
profiles and 96.4% for the benchmark, lower than the 22.6% median quoted in Section 6.7
because that one includes Runes, which the profiles capture well. Even where the profiles do well they leave something: 80.9% against
98.7% for P2PKH inputs and about 84% against 100% for the two common witness classes.
The information is in the twelve numbers, and K-means at K = 31 does not put it into
separate clusters.

The last column of Table {T_ceiling} makes the comparison without using any test-period
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
ceiling. It beats the single-annotation benchmark for two of the ten annotations of
Table {T_ceiling} and matches it for one, so eleven coordinates are sometimes a better
description of a transaction than the one fitted for the annotation at hand, but not
usually. Against the equal-cell variant the shared partition looks far
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

They add a great deal for the one annotation that needs it (Table {T_richer}). For
exchange tags the supervised benchmark on the twelve features is 40.8% and on all
twenty-four 59.6%, a gain of 19 percentage points, which is what one would expect of a
property defined by the addresses a transaction touches. For the two OP_RETURN
annotations they add much less (82.0% to 86.4% for other OP_RETURN use, 99.4% to
99.7% for Runes), which is also expected: those are properties of the outputs
themselves, already visible to the twelve. Refitting the profiles on all twenty-four
features moves them very little — 9.4% against 9.8% for exchange tags, and 4.8%
against 5.9% for other OP_RETURN use — so the extra information does not reach the
profiles through the clustering objective either.

This is a bounded test. Reuse measured inside a sparse sample is a weak proxy for reuse
in the chain, so the 59.6% is likely an underestimate of what an entity-level view could
give. Taken with the rest of the section it points to a division of labour: for the
properties of a transaction's own structure the twelve features suffice and the
objective is what fails, while for properties of the parties behind it, richer features
are worth having and the clustering still has to be able to use them.
