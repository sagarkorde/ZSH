# Draft — Introduction

## 1. Introduction

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

The study was specified and its code frozen before the reported runs. The freeze
is not blindness: an earlier submitted version of this work had already examined the
same Bitcoin corpus and the labelled Elliptic data, so the analyses of those data are
a frozen reanalysis rather than a blind confirmatory test (Section 5.1). Only the new
sample of transactions from October 2024 to August 2026, collected after the freeze,
is prospective. The main contributions are:

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
5. Evidence on where the limit lies, added after the freeze and reported
   throughout as exploratory. Six other clustering families reach the same low
   shares of the ceiling for the least frequent annotations, so those limits
   belong to the task rather than to ZSH; and a supervised model on the same
   twelve features attains far more than any of them, which places the limit in
   the clustering objective rather than in the features — except for exchange
   tags, where richer features do raise what supervision can extract
   (Sections 6.7 and 6.8).
6. A public, versioned repository with the frozen analysis plan, the code, the
   prospective-data collector and all result files.

Section 2 reviews related work, Section 3 describes the data and the audit,
Section 4 the method, Section 5 the experimental design, Section 6 the results
and Section 7 their meaning and limits. Section 8 concludes.
