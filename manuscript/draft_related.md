# Draft — Related Work

## 2. Related Work

### 2.1. Entities and addresses

Most unsupervised analysis of Bitcoin works at the level of addresses and
entities. Common-input-ownership and change-address heuristics group addresses
that are probably controlled by the same party [@AddrClust2024], and platforms such as GraphSense combine such clustering with
attribution tags [@Haslhofer2021]. These methods answer *who* controls funds.
They do not say what a single transaction does, and privacy techniques such as
CoinJoin are designed to break their assumptions [@CoinJoinPriv2024]. The
profiles studied here are complementary: they describe transactions, and they
can be read next to entity information when it exists.

### 2.2. Transaction-level unsupervised analysis

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

### 2.3. Supervised and graph-based detection

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

### 2.4. Feature weighting and size constraints in K-means

K-means treats all features alike, so its result depends on scaling and on
redundant features. Feature-weighted variants learn weights inside the
clustering objective, and recent work refines the weighting of features for fully
unsupervised clustering [@FeatWeight2025].
Unsupervised filters such as the Laplacian score rank features by how well they
preserve local structure [@He2005]. Our weighting is simpler: it ranks features
by mutual information with a proxy partition and assigns weights that decay as
a power of the rank. Other work builds the size constraint into the clustering itself: MST-DHC balances
cluster sizes through a minimum spanning tree and needs no size parameter
[@BalancedClust2026]. We use recursive splitting, which is cheaper at the scale of
millions of rows but gives no guarantee.

### 2.5. Validating a clustering without labels

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

### 2.6. Positioning

ZSH is not a new clustering algorithm. It combines a rank-power feature
weighting, a hierarchical initialisation and a size-constrained refinement
around standard K-means. The contribution of this article is the evaluation: a
frozen, reproducible pipeline tested against matched baselines, under
refitting, on later data, on annotations that were never inputs, and on
external illicit labels, with the results reported whether or not they favour
the method.

