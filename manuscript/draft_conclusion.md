# Draft — Conclusions

## 8. Conclusions

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
where the profiles attain 23.6%, and over the ten annotations the median bound is
96.4% against 16.5% for the profiles. The information is present and the clustering
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
