# Draft — Discussion ({FUTURE} parts pending)

## 7. Discussion

### 7.1. What the weights do

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

### 7.2. Refinement and initialisation

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

### 7.3. Reproducibility and change over time

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
42.8% to 84.2%. The instability of a free refit is therefore not mainly the data moving
but the model moving with it: what the constrained refit holds fixed is the feature
ranking, and holding it fixed is what lets the profiles keep their identity. For a
monitoring system this is the practical recipe — freeze the space, re-estimate the
centroids, and match the profiles across refits — rather than a choice between a
drifting frozen model and a series that restarts. This is not a problem of one method: of the seven
families in Section 6.7, the only ones whose profiles can be followed through a
refit are those in which a single cluster already holds most of the data, where
the match is easy and uninformative.

### 7.4. What the profiles say about illicit activity

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

### 7.5. What limits the profiles

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
annotations the median is 96.4% for a bound fitted within the period and 16.5% for the
profiles, with no annotation exempt. The information is in the twelve numbers. What the
profiles lack is not better features and not better weights but an objective that looks
for this structure: K-means minimises within-cluster variance, and a partition of
minimum variance is not a partition of maximum concentration. The same pattern appears
on Elliptic, where clusters of the 165 features attain 10.2% and a random forest on
those same features 78% (Section 6.5).

Two earlier readings of ours did not survive this measurement, and we report them
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

### 7.6. Practical use

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

### 7.7. Privacy and responsible use

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

### 7.8. Limitations and threats to validity

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
  blocks only partly. Every sampled page was obtained.
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
  specified in the plan and fourteen were added afterwards (Table {T_prespec});
  they are labelled throughout, and the two that overturned earlier conclusions
  are identified as such.
