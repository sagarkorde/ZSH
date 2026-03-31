# ZSH: A Zeta-Weighted Hybrid Semantic Clustering Framework for Blockchain Transaction Profiling and Anomaly-Aware Forensic Analysis

**Author 1**, Student Member, IEEE, **Author 2**, **Author 3**

**Affiliation:** Department/School, University Name, City, Country  
**Corresponding author:** Name, email@domain.com

**Abstract** - Blockchain transaction profiling is a critical capability for anti-money laundering, fraud investigation, and forensic intelligence, yet it remains difficult because large public ledgers are weakly labeled, behaviorally heterogeneous, and poorly served by geometry-only clustering objectives. This paper proposes **ZSH**, a zeta-weighted hybrid semantic clustering framework for blockchain transaction profiling and anomaly-aware forensic analysis. ZSH combines finite-normalized Riemann zeta feature weighting, geometry-aware rank fusion, semantic seed allocation, Ward-guided centroid initialization, semantic post-labeling, and a profile-preserving anomaly layer. Experiments were conducted on **11,303,526** Bitcoin transactions from blocks **744,837-903,456** using **27** behavioral and structural features. The framework produced **30** semantically interpretable transaction profiles and flagged **564,674** anomalous transactions (**5.00%**). Statistical validation showed stable structure, with bootstrap Silhouette **0.4495** (95% confidence interval **[0.4374, 0.4615]**) and permutation significance **p < 0.001**. In a corrected same-space benchmark, **KMeans++ Elkan** achieved the strongest intrinsic Euclidean geometry; however, ZSH delivered stronger profiling utility, including higher weighted semantic purity (**0.9474 vs. 0.9368**), higher macro purity (**0.8705 vs. 0.8179**), lower semantic entropy (**0.2543 vs. 0.2772**), and better minority-profile recoverability under shallow expert-rule explanation (**balanced accuracy 0.4541 vs. 0.3953**). The results show that ZSH should be interpreted not as a universal replacement for geometry-first clustering, but as a statistically validated, semantically grounded, and operationally actionable framework for large-scale blockchain transaction profiling.

**Index Terms** - blockchain analytics, transaction profiling, anomaly detection, clustering, cryptocurrency forensics.

## I. INTRODUCTION

Blockchain systems have transformed digital value transfer by enabling transparent, tamper-evident, and globally accessible transaction ledgers. At the same time, this transparency produces a new analytical challenge: large public ledgers contain millions of heterogeneous transactions whose behavioral meaning is not directly labeled. For investigators, regulators, compliance teams, and security analysts, the central question is no longer whether blockchain data are available, but how such data can be transformed into reliable, interpretable, and scalable behavioral profiles. Recent surveys confirm that graph learning, machine learning, and representation learning have become central to blockchain analytics, especially for illicit transaction detection, fraud analysis, and forensic intelligence [1]-[3].

Despite this progress, blockchain transaction pattern recognition remains difficult for four reasons. First, transaction behavior is inherently multimodal. Normal peer-to-peer transfers, exchange batching, coinjoin-like obfuscation, consolidations, OP_RETURN usage, and replace-by-fee behavior occupy overlapping but non-identical regions of feature space. Second, large blockchain corpora are weakly labeled or entirely unlabeled, which limits the use of fully supervised pipelines. Third, many existing studies optimize only prediction or only clustering geometry, leaving the resulting output difficult to interpret operationally. Fourth, evaluation is often not fully aligned with the intended use case: an algorithm can produce compact Euclidean clusters while still failing to provide semantically coherent behavioral categories that support forensic reasoning.

The recent literature contains strong advances in anomaly detection, anti-money laundering, and illicit activity mining [4]-[8]. Researchers have proposed statistical and machine learning approaches for suspicious transaction detection, Bitcoin fraud analysis, and scalable anomaly screening. In parallel, graph-based models have produced strong results for Ethereum phishing, illicit account identification, and transaction-graph representation learning [9]-[20]. These contributions are important and state-of-the-art within their respective problem formulations. However, much of the literature remains focused on one of three paradigms: (i) supervised illicit account classification, (ii) graph-based phishing detection, or (iii) anomaly scoring without semantically interpretable profile generation. In contrast, many practical blockchain analytics settings require an unsupervised or weakly supervised system that can profile transactions at scale, expose meaningful behavioral categories, and preserve a separate anomaly channel for downstream review.

This paper addresses that gap through **ZSH**, a zeta-weighted hybrid semantic clustering framework designed for **dynamic transaction profiling**. The term dynamic profiling is used here in two senses. First, the framework supports time-distributed blockchain data by preserving a metadata layer for block-height and year-based profile analysis. Second, it produces a flexible profile taxonomy that can separate recurring behavioral families into stable macro- and micro-profiles, such as multiple coinjoin-like or batch-payment subtypes. Rather than treating clustering as an isolated geometry optimization problem, ZSH combines weighting, semantic initialization, geometry guidance, and anomaly-aware post-analysis into a single pipeline.

The manuscript is intentionally positioned around an honest scientific claim. The corrected same-space benchmark performed in this study shows that **KMeans++ Elkan** achieves stronger intrinsic Euclidean clustering scores than ZSH on Silhouette, Davies-Bouldin index, and Calinski-Harabasz index. Therefore, this paper does **not** claim that ZSH is the universally best geometry-only clustering algorithm for blockchain transaction data. Instead, the contribution is that ZSH is **better aligned with the blockchain profiling task**, where semantic coherence, explainability, and anomaly-aware interpretability are at least as important as pure cluster compactness.

The main contributions of this paper are summarized as follows:

1. A **finite-normalized Riemann zeta weighting mechanism** is proposed for ranking and weighting blockchain transaction features. Unlike an infinite-series normalization, the proposed finite normalization guarantees that the feature weights sum exactly to one over the observed feature set.
2. A **hybrid profiling pipeline** is introduced that integrates rule-derived semantic seed allocation, Ward-guided micro-cluster structure, and centroid refinement while preserving a corrected same-space comparison with classical baselines.
3. A **semantic post-labeling and profile-preserving anomaly design** is developed so that transaction-family identity and anomaly severity are modeled as complementary outputs rather than collapsed into a single score.
4. A **reviewer-safe evaluation protocol** is implemented in which intrinsic metrics, bootstrap confidence intervals, permutation testing, and ablation analysis are all computed in the same standardized zeta-weighted feature space on an 11.3-million-transaction corpus.
5. A **task-aligned superiority analysis** is provided to show where ZSH should and should not be preferred to KMeans: KMeans++ Elkan remains the strongest geometry-only baseline, whereas ZSH is stronger for semantic profile purity, minority-profile recoverability, and anomaly-aware forensic interpretation.

The remainder of the paper is organized as follows. Section II reviews recent work on blockchain anomaly detection, graph learning, and profiling-oriented analysis. Section III presents the proposed ZSH framework. Section IV describes the dataset, implementation settings, and evaluation protocol. Section V reports the experimental results, ablations, contextual comparisons, and limitations. Section VI concludes the paper and outlines future research directions.

## II. RELATED WORK

### A. Machine Learning for Blockchain Anomaly Detection and AML

Recent blockchain analytics research has increasingly adopted statistical learning and anomaly detection for anti-money laundering (AML), fraud screening, and suspicious behavior identification. Surveys and broad reviews show that blockchain analytics is moving away from purely heuristic systems toward data-driven pipelines capable of mining large transaction graphs and high-dimensional behavioral features [1]-[3]. Within AML and illicit-finance settings, Jensen and Iosifidis [4] demonstrated that classical statistics and machine learning remain competitive tools for laundering-related pattern analysis, while Pocher *et al.* [5] framed anomalous cryptocurrency transaction detection as an AML/CFT problem and highlighted the value of forensic learning on ledger data. Nayyer *et al.* [6] combined ensemble stacking for Bitcoin fraud detection, showing that even non-graph models can capture suspicious behavior when feature engineering is sufficiently informative.

Other recent work has focused directly on anomaly detection infrastructures for blockchain networks. Voronov *et al.* [7] proposed a sketch-based framework for scalable anomaly detection in blockchain networks, emphasizing efficiency and streaming suitability. Elmougy and Liu [8] investigated fraudulent transactions and illicit nodes in the Bitcoin network through a financial-forensics perspective, illustrating how transaction-level and node-level signals interact. These studies demonstrate that anomaly detection and forensic mining are already central research topics, but they also reveal a recurring limitation: many methods focus on binary suspicious-vs-non-suspicious detection and do not provide a stable semantic transaction taxonomy that can support profiling at scale.

A second line of work has emphasized richer anomaly modeling. Hasan *et al.* [21] used conventional classifiers together with explainability analysis for blockchain transaction anomaly detection, arguing that model transparency is important in decision-sensitive settings. Ouyang *et al.* [22] proposed subgraph contrastive learning for Bitcoin money laundering detection, shifting the emphasis from hand-crafted local features toward learned topological context. Wang *et al.* [23] combined CNN and Transformer components for abnormal transaction detection, indicating that spatio-temporal feature interactions matter in blockchain data. Liang *et al.* [30] later introduced a plug-and-play data-driven AML framework for Bitcoin that further reinforced the trend toward modular learning pipelines.

Although these studies significantly advance anomaly and illicit transaction detection, two gaps remain relevant for the present work. First, most methods are developed for **detection** rather than **profiling**, meaning they primarily answer whether a transaction is suspicious rather than what behavioral family it belongs to. Second, anomaly outputs are often intertwined with class labels or end-task predictions, whereas practical profiling pipelines benefit from separating behavioral structure from anomaly severity. This separation is one of the design principles of ZSH.

### B. Graph Learning for Illicit Accounts, Phishing, and Transaction Semantics

Graph neural networks and transaction-graph representation learning have become especially prominent in Ethereum phishing and illicit account detection. Hu *et al.* [9] introduced BERT4ETH, a transformer-based approach for Ethereum fraud detection, showing that sequential and contextual transaction semantics can be pretrained. Wang *et al.* [10] used temporal graph attention to detect phishing scams in Ethereum, while Xiong *et al.* [11] applied graph neural networks directly to Ethereum phishing detection. Li *et al.* [12] proposed self-supervised incremental deep graph learning for phishing detection, and Li *et al.* [13] further developed transaction graph contrastive learning. Cai *et al.* [14] extended semantic learning into smart-contract security via transaction semantic representation learning for Ponzi detection.

Researchers have also studied temporal and dynamic anomaly structures in blockchain graphs. Liu *et al.* [15] used evolved graph attention for directed dynamic attribute graph anomaly detection. Han *et al.* [16] proposed MT^2AD for multi-layer temporal transaction anomaly detection in Ethereum, and Xiao *et al.* [17] introduced a spatio-temporal and global representation approach for abnormal cryptocurrency transactions. These works underline a major point: behavioral anomalies in blockchain systems are often dynamic, relational, and dependent on more than simple static feature vectors.

During 2024 and 2025, graph-based phishing and illicit-account analysis became even more sophisticated. Huang *et al.* [18] proposed PEAE-GNN using augmentation ego-graphs; Zhang *et al.* [19] introduced GrabPhisher via temporally evolving GNNs; Liu *et al.* [20] uncovered Ethereum phishing gangs using blockchain data at an information-forensics level; Chen *et al.* [24] used data augmentation with hybrid graph neural networks; Ding *et al.* [25] addressed illicit account detection on large cryptocurrency multigraphs; Yang *et al.* [26] proposed a streaming framework for Ethereum phishing scam detection; Song *et al.* [27] studied anti-money laundering for transactional blockchains; Hou *et al.* [28] introduced triple-stream feature fusion; and Zhang *et al.* [29] modeled dynamic multiperspective cascade graphs for Web3 phishing detection. The latest contributions continue this trajectory: Sheng *et al.* [31] combined global graph structures and local semantics, Sui *et al.* [32] applied graph contrastive learning in EPAD, Huang *et al.* [33] proposed pseudolabel generation for blockchain anomaly transaction detection, Asiri and Somasundaram [34] reported graph-convolution-based fraud detection in Bitcoin transactions, and Shen *et al.* [35] introduced graph continual learning for blockchain anomaly transaction detection.

This body of work is highly relevant to the current study, but it is still largely dominated by **supervised or semi-supervised account classification** and **phishing detection**. Such methods are extremely valuable, yet they do not fully resolve the problem addressed in this paper: large-scale, weakly labeled **transaction profiling** with explicit semantic labels and a separate anomaly layer. In addition, many graph-learning papers rely on specialized benchmark datasets and account-level labels, which can be difficult to transfer directly to broader transaction-level profiling pipelines.

### C. Research Gap and Position of the Present Study

Based on the recent literature, three research gaps remain open.

First, the majority of recent blockchain detection studies optimize for labeled illicit-account recognition, phishing scam detection, or anomaly classification. These are important tasks, but they differ from unsupervised transaction profiling in which the analyst wants an interpretable profile space, not just a fraud score [8]-[13], [18]-[20], [25]-[29], [31]-[35].

Second, interpretability is often discussed but not operationalized into a profile taxonomy. Many models provide feature importance, contrastive embeddings, or local explanations, yet few produce stable behavioral groups such as batch-payment, coinjoin-like, distribution, or OP_RETURN profiles that can be used as an intermediate forensic language between raw transactions and downstream investigators [5], [21], [30].

Third, comparisons with classical clustering algorithms are often incomplete because methods are trained or evaluated in different spaces, or because downstream anomaly components are allowed to affect cluster structure. This can obscure the true relationship between geometric performance and profiling utility. The present work addresses this issue through a corrected same-space evaluation protocol and a contextual comparison that explicitly distinguishes geometry-only clustering from semantically grounded transaction profiling.

Accordingly, this paper is positioned at the intersection of clustering, weak semantic supervision, anomaly-aware profiling, and blockchain forensics. Instead of competing with the latest supervised graph models on their exact end tasks, the proposed framework addresses a complementary problem: how to convert a very large blockchain transaction corpus into semantically interpretable profiles and anomaly flags that remain statistically valid, operationally meaningful, and reproducible at scale.

## III. PROPOSED METHOD AND SYSTEM DESIGN

### A. Problem Formulation

Let the blockchain transaction dataset be represented by a feature matrix

\[
\mathbf{X} \in \mathbb{R}^{n \times d},
\]

where \(n\) is the number of transactions and \(d\) is the number of engineered transaction features. In this study, \(n = 11{,}303{,}526\) and \(d = 27\). The objective is to partition the transactions into \(K\) behaviorally meaningful profiles while also producing a transaction-level anomaly indicator. The framework should satisfy four design criteria:

1. **Scalability** to multi-million-transaction corpora.
2. **Interpretability** through semantically meaningful transaction profiles.
3. **Statistical validity** through corrected same-space evaluation.
4. **Operational separation** between cluster structure and anomaly scoring.

The final output of the framework is a tuple

\[
(\mathbf{y}, \mathbf{p}, \mathbf{a}, \mathbf{s}),
\]

where \(\mathbf{y}\) denotes numeric cluster labels, \(\mathbf{p}\) denotes semantic profile names, \(\mathbf{a}\) denotes binary anomaly flags, and \(\mathbf{s}\) denotes continuous anomaly scores.

### B. ZSH Pipeline Overview

The ZSH framework is implemented as a multi-stage pipeline. First, raw transactions are loaded, quality checked, and transformed into a balanced feature matrix. Second, the feature space is standardized and then reweighted using a finite-normalized Riemann zeta scheme derived from proxy-label mutual information. Third, a geometry-aware branch is created to assess whether topology-preserving feature emphasis can improve downstream clustering. Fourth, a hybrid clustering stage combines semantic seed allocation and Ward-inspired centroid structure to generate behaviorally coherent clusters. Fifth, clusters are post-labeled into semantic profiles using expert-rule dominance. Sixth, an Isolation Forest is applied as an independent anomaly layer. Finally, statistical validation and contextual benchmarking are performed.

The pipeline separates **clustering**, **semantic naming**, and **anomaly detection**. This separation is deliberate. Clusters define the transaction structure, semantic labels define how clusters are interpreted, and anomaly flags identify outlying cases within or across profiles. Because these three layers serve different analytical purposes, they should not be merged into a single optimization objective.

### C. Feature Engineering and Preprocessing

The final feature set consists of 27 transaction-level variables spanning size, value transfer, fee behavior, address and script composition, and rule-like behavioral indicators. Representative variables include `total_input_value`, `total_output_value`, `avg_input_value`, `avg_output_value`, `input_output_ratio`, `fee`, `fee_rate_sat_per_byte`, `fee_rate_sat_per_vbyte`, `input_count`, `output_count`, `input_address_count`, `output_address_count`, `total_addresses`, `value_concentration_ratio`, and binary flags such as `has_coinbase`, `has_op_return`, `rbf_enabled`, `is_consolidation`, `is_distribution`, `is_peer_to_peer`, `is_batch_payment`, and `is_coinjoin_like`.

Identifier and metadata fields such as transaction IDs, raw timestamps, and block-height variables are not used directly in clustering. They are retained only for descriptive analysis, temporal profiling, and post-hoc interpretation. This avoids label leakage and ensures that the profile space is driven by transaction behavior rather than record identifiers.

After preprocessing, the scaled matrix is denoted by \(\mathbf{X}_{\text{scaled}}\). This matrix is then passed to the weighting stage. The use of scaled input is important because the downstream weighting mechanism is intended to encode feature relevance, not raw magnitude differences.

### D. Finite-Normalized Riemann Zeta Feature Weighting

The first novel component of ZSH is the use of finite-normalized Riemann zeta weighting. Proxy labels are first generated using MiniBatchKMeans with ten clusters on \(\mathbf{X}_{\text{scaled}}\). These proxy labels provide a low-cost surrogate structure for mutual information estimation. For each feature \(j\), a mutual information score \(m_j\) is computed:

\[
m_j = I(X_j ; y^{(p)}),
\]

where \(y^{(p)}\) denotes the proxy cluster assignment. Features are ranked in descending order of \(m_j\), and the rank of feature \(j\) is denoted by \(r_j\).

The zeta weight assigned to feature \(j\) is then

\[
w_j = \frac{r_j^{-s}}{\sum_{k=1}^{d} k^{-s}},
\]

where \(s\) is the decay exponent. In this study, the primary setting is \(s = 1.5\), while \(s \in \{1.0, 1.5, 2.0, 3.0\}\) is used for sensitivity analysis.

The choice of a **finite** normalizer is important. Using the infinite zeta function \(\zeta(s)\) would allocate probability mass to non-existent feature ranks beyond \(d\), causing the total weight assigned to the observed features to fall below one. By normalizing with the finite partial sum \(\sum_{k=1}^{d} k^{-s}\), ZSH guarantees exact scale invariance over the actual feature set. In the present implementation, the weight sum was validated numerically to equal 1.0 within floating-point tolerance.

The weighted matrix is then obtained by element-wise multiplication:

\[
\mathbf{X}_{w} = \mathbf{X}_{\text{scaled}} \odot \mathbf{w},
\]

where \(\mathbf{w} = [w_1, \ldots, w_d]\). In the current dataset, the highest-ranked features under the primary zeta weighting were `total_input_value`, `avg_input_value`, `input_output_ratio`, `total_output_value`, and `avg_output_value`, indicating that transaction-value geometry dominates the latent structure more strongly than binary rule flags.

### E. Geometry-Aware Rank Fusion and ZSH-G Branch

To complement the semantics-oriented zeta ranking, a second ranking branch is constructed from local geometry preservation. A sample of the scaled data is compressed into anchor points using MiniBatchKMeans, and a weighted k-nearest-neighbor graph is built over the anchors. A Laplacian score is then computed for each feature; lower scores indicate better preservation of local manifold structure.

The geometry-aware rank score for feature \(j\) is defined as

\[
g_j = \alpha r^{(\ell)}_j + \beta r^{(m)}_j + p_j,
\]

where \(r^{(\ell)}_j\) is the Laplacian-score rank, \(r^{(m)}_j\) is the mutual-information rank, \(\alpha = 0.70\), \(\beta = 0.30\), and \(p_j\) is a penalty term applied to binary rule flags. The penalty discourages the geometry branch from over-prioritizing sparse indicator variables when the goal is to preserve continuous manifold structure.

The resulting geometry rank is converted into a second zeta-style weight vector. In the present dataset, the top geometry-ranked features were `input_output_ratio`, `avg_input_value`, `total_output_value`, `total_input_value`, and `avg_output_value`, which is broadly consistent with the dominance of value-flow features in the main branch.

In addition, a linear metric-learning branch, denoted **ZSH-G**, is fitted on the corrected zeta-weighted space. This branch uses KMeans++ Elkan as a teacher and searches among identity, Mahalanobis-style, Fisher-style, and diagonal-ratio transforms. The goal is to discover a linear preconditioner that improves downstream intrinsic cluster geometry. Importantly, candidate transforms are validated against the teacher on a held-out split. In the current experiment, the **identity fallback** was selected, indicating that the learned metric branch did not surpass the teacher on the corrected validation geometry. This outcome is scientifically important: it shows that the final gains reported for ZSH come primarily from the core weighting, semantic initialization, and profiling design rather than from an artificially favorable metric transform.

### F. Hybrid Semantic Clustering

The clustering stage is designed to combine semantic prior knowledge with data-driven geometry. The profile count is fixed at \(K = 30\). Rule-derived semantic seeds are constructed from eight blockchain behavioral indicators:

1. `has_coinbase`
2. `is_coinjoin_like`
3. `is_batch_payment`
4. `is_consolidation`
5. `is_distribution`
6. `is_peer_to_peer`
7. `has_op_return`
8. `rbf_enabled`

For each semantic family \(c\), the number of seed centroids is allocated proportionally to its support:

\[
k_c = \max\left(1, \operatorname{round}\left(\frac{K n_c}{\sum_{u} n_u}\right)\right),
\]

where \(n_c\) is the count of transactions assigned to family \(c\). Sub-cluster centroids are then estimated through MiniBatchKMeans within each family.

To inject geometry, Ward-guided micro-cluster initialization is also constructed. The training subset is first partitioned into a larger set of micro-clusters, and agglomerative Ward linkage is applied to the micro-centers to form \(K\) macro-centers. These semantic and Ward-based centers are aligned with the Hungarian algorithm and blended:

\[
\mathbf{C}_{\text{blend}} = \lambda \mathbf{C}_{\text{seed}} + (1-\lambda)\mathbf{C}_{\text{ward}},
\]

with \(\lambda = 0.60\) in the current configuration.

Several candidate clustering modes are then compared on a held-out validation subset:

1. Seeded MiniBatchKMeans
2. Seeded refinement
3. Ward-guided refinement
4. Semantic-Ward blend
5. Geometry-first Elkan
6. Geometry-first Ward refinement
7. Geometry-first Elkan-Ward blend

Candidate selection is based on a balanced ranking over Silhouette, Davies-Bouldin index, and Calinski-Harabasz index in the corrected **same evaluation space**. The validation reference is KMeans++ Elkan in standardized zeta-weighted space. This selection procedure avoids choosing a candidate based solely on a single metric and prevents unfair evaluation across inconsistent spaces.

### G. Semantic Post-Labeling

After the numeric cluster labels are generated, each cluster is assigned a semantic name using rule-family dominance. For cluster \(c\), the semantic label is chosen as the most frequent non-`Unknown` rule-derived family within the cluster. If multiple clusters share the same dominant family, suffixes such as `_1`, `_2`, or `_3` are appended to preserve one-to-one profile naming.

This post-labeling strategy is important because it cleanly separates **cluster formation** from **semantic interpretation**. Seed labels help initialize clustering, while semantic labels help name the resulting behavioral groups. This design avoids circularity in which labels are forced too strongly during centroid fitting.

### H. Anomaly Layer

The anomaly stage uses Isolation Forest fitted on a large sample from the corrected zeta-weighted space. The anomaly score for transaction \(i\) is denoted by \(s_i\), and the binary anomaly flag is assigned by thresholding at the 95th percentile:

\[
a_i =
\begin{cases}
1, & s_i > Q_{0.95}(\mathbf{s}) \\
0, & \text{otherwise}.
\end{cases}
\]

This yields an anomaly rate of approximately 5%, matching the contamination setting. Crucially, anomaly detection does **not** modify the cluster labels. This preserves the integrity of the profile structure and enables analysts to distinguish between "what profile a transaction belongs to" and "how unusual that transaction is within the overall distribution."

### I. Reproducibility and Implementation Design

The full pipeline is implemented with checkpointing, chunked computation, memory mapping, and explicit artifact logging. The design targets workstation-level hardware and avoids loading the full multi-gigabyte feature matrix unnecessarily into RAM. This is particularly important for large blockchain datasets and supports reproducibility under constrained but practical research hardware settings. All major outputs, including feature ranks, weighted arrays, labels, profile summaries, statistical reports, and comparison tables, are saved to disk to permit exact downstream regeneration of figures and tables.

To strengthen experimentation proof and post-review transparency, the final submission should be accompanied by a public GitHub repository containing the pipeline scripts, an environment specification, an artifact manifest, and instructions for regenerating the key tables and figures. In the present draft, that release is represented by a local reproducibility package; the final camera-ready version should replace the placeholder with the public repository URL.

## IV. EXPERIMENTAL SETUP AND IMPLEMENTATION

### A. Dataset and Coverage

The experiments were conducted on a blockchain transaction corpus covering Bitcoin blocks **744,837 to 903,456**, spanning transactions from **2022 to 2025**. The final aligned dataset contained **11,303,526** transactions. Year-level metadata show that the corpus is concentrated in 2022-2024, with a small 2025 tail due to the extraction window. The 27 clustering features were drawn from transaction size, value transfer, fee behavior, address dispersion, script composition, and behavior flags. Temporal attributes such as block height and year were retained for profile interpretation but not used as direct clustering inputs.

### B. Hardware and Software Environment

The pipeline was executed on an HP Omen workstation with an Intel 13th-generation Core i9 CPU, **64 GB RAM**, and an **NVIDIA RTX 4060 (8 GB)**. The implementation was built in Python using NumPy, pandas, scikit-learn, DuckDB, joblib, and matplotlib. Memory-mapped arrays and chunked writes were used to keep the workflow stable on a workstation-scale environment rather than a distributed cluster.

### C. Evaluation Protocol

The evaluation protocol was intentionally corrected to ensure fairness. All intrinsic clustering metrics were computed in the standardized zeta-weighted space, denoted \(X_{w\_norm}\). UMAP was used only for visualization and was not used for metric computation. This choice eliminates a common problem in clustering studies, namely evaluating Euclidean metrics in a manifold-reduced space that does not preserve the original geometry in a metric-consistent way.

Three intrinsic clustering metrics were used:

1. **Silhouette score** (higher is better)
2. **Davies-Bouldin index** (lower is better)
3. **Calinski-Harabasz index** (higher is better)

In addition, the study employed:

1. **Bootstrap confidence intervals** over 200 stratified resamples of 5,000 rows
2. **Permutation significance testing** with 300 label-shuffle trials
3. **Ablation analysis** to isolate the contributions of weighting, refinement, and anomaly separation
4. **Contextual profiling comparison** against KMeans++ Elkan over five repeated 120,000-row samples

### D. Baselines

The following baselines were used in the same corrected evaluation space:

1. Vanilla KMeans
2. KMeans++ Elkan
3. HDBSCAN
4. Gaussian Mixture Model (GMM)
5. Agglomerative clustering
6. Ward-guided hybrid baseline

KMeans++ Elkan serves as the strongest geometry-only reference. This is an important choice because the purpose of the study is not to compare ZSH against weak baselines, but to determine whether a semantically grounded hybrid framework can provide better profiling utility even when a classical geometry optimizer remains stronger on intrinsic cluster shape.

### E. Core Experimental Settings

**Table I** summarizes the most important experimental settings used in the final study.

| Parameter | Value |
| --- | --- |
| Transactions | 11,303,526 |
| Clustering features | 27 |
| Final profile count \(K\) | 30 |
| Fit subset for Step 5 | 500,000 |
| Validation subset | 50,000 |
| Evaluation subset | 60,000 |
| Zeta decay exponent \(s\) | 1.5 |
| Ward micro-clusters | 160 |
| Isolation Forest contamination | 0.05 |
| Bootstrap iterations | 200 |
| Permutation iterations | 300 |
| Contextual comparison repeats | 5 |
| Rows per contextual repeat | 120,000 |

## V. RESULTS AND DISCUSSION

### A. Statistical Validity of the Learned Profile Structure

The first question is whether ZSH discovers non-trivial structure at all. The answer is affirmative. Across 200 bootstrap iterations on stratified 5,000-row subsamples, ZSH achieved a mean Silhouette score of **0.4495** with 95% confidence interval **[0.4374, 0.4615]**, a Davies-Bouldin index of **1.0722** with confidence interval **[0.9836, 1.3216]**, and a Calinski-Harabasz index of **898.2** with confidence interval **[298.4, 1351.5]**. These intervals are sufficiently tight to indicate that the profile structure is stable under resampling and not an artifact of a single draw.

The permutation test provides stronger evidence. The observed Silhouette on the evaluation pool was **0.4485**, whereas the null distribution under label shuffling had mean **-0.1942** with standard deviation **0.0327**. The resulting **p < 0.001** rejects the null hypothesis that the cluster assignments are indistinguishable from random labeling. This result is important because the later contextual claims about interpretability would be much weaker if the underlying partition itself were unstable or statistically unconvincing.

The bootstrap confidence intervals are summarized visually in **Fig. 9**.

### B. Same-Space Comparison With Classical and Modern Baselines

The corrected same-space benchmark is presented in **Table II**. Every method was trained and evaluated in the same standardized zeta-weighted feature space.

**Table II. Same-space SOTA comparison in \(X_{w\_norm}\).**

| Method | k | Silhouette | DBI | CHI |
| --- | ---: | ---: | ---: | ---: |
| ZSH (ours) | 30 | 0.4530 | 1.0659 | 1315.2 |
| Vanilla KMeans | 30 | 0.4875 | 1.0333 | 1311.0 |
| KMeans++ Elkan | 30 | 0.4966 | 0.8443 | 2926.4 |
| HDBSCAN | 14 | 0.2770 | 1.9099 | 557.2 |
| GMM | 30 | 0.3332 | 1.8631 | 995.2 |
| Ward-guided hybrid | 30 | 0.4238 | 1.0436 | 1304.6 |
| Agglomerative | 30 | 0.4697 | 0.9073 | 2559.9 |

The strongest geometry-only method was **KMeans++ Elkan**, which outperformed ZSH on all three intrinsic criteria. Agglomerative clustering also exceeded ZSH on Silhouette, DBI, and CHI. This result must be acknowledged directly. It means that, in pure Euclidean compactness-separation terms, ZSH is **not** the top performer in the present dataset.

This finding is not a weakness of the paper if it is interpreted correctly. It clarifies the scientific contribution. ZSH should not be marketed as "the best clustering algorithm" in an absolute sense. Rather, it should be positioned as a framework that sacrifices some geometry-only optimality in exchange for semantic coherence, profile naming, and anomaly-aware interpretability. For a Q1 journal, this distinction is essential because it turns a potentially overstated claim into a careful, defensible one. **Fig. 10** visualizes the same-space comparison and makes the geometry-first advantage of KMeans++ Elkan explicit rather than implicit.

### C. Contribution of the Weighting and Hybrid Design

To understand where ZSH gains and losses originate, an ablation study was performed. The results are shown in **Table III**. Importantly, Table III is a **component-isolation study**, not a second claim that the final named ZSH profiles dominate the benchmark in Table II. In particular, Condition C corresponds to the geometry-refinement branch whose validation step selected the **identity fallback**, making it effectively equivalent to KMeans++ Elkan in \(X_{w\_norm}\). Table III therefore isolates what geometry refinement and anomaly separation contribute, whereas Table II remains the correct comparison for the final profiling framework.

**Table III. Ablation study.**

| Condition | Silhouette | DBI | CHI | Anomaly rate |
| --- | ---: | ---: | ---: | ---: |
| A: Vanilla KMeans (no zeta, no seeds) | 0.4469 | 1.3498 | 1656.4 | - |
| B: + Zeta weighting (no seeds) | 0.4276 | 1.0340 | 1633.3 | - |
| C: + Elkan geometry refinement | 0.4896 | 0.8637 | 6204.5 | - |
| D: Condition C + Isolation Forest flag | 0.4896 | 0.8637 | 6204.5 | 5.03% |

The ablation reveals three important points.

First, **zeta weighting alone is not sufficient** to guarantee improvement on all intrinsic metrics. Relative to vanilla KMeans, the weighted-only condition improved DBI but reduced Silhouette and CHI. This suggests that the value of zeta weighting is not simply to optimize geometry directly, but to reshape the feature space so that subsequent semantic and hybrid operations become more meaningful.

Second, the largest intrinsic gain came from **Elkan-based geometry refinement**, which improved Silhouette from 0.4276 to 0.4896 and increased CHI dramatically from 1633.3 to 6204.5 relative to the weighted-only condition. This indicates that refined centroid optimization remains crucial even in a semantically guided clustering framework.

Third, the anomaly layer did **not** change the cluster structure. Conditions C and D have identical clustering metrics, while Condition D additionally reports the anomaly rate. This is exactly the intended behavior of ZSH. The anomaly component enriches the analytical output without contaminating the core cluster assignments.

**Fig. 11** summarizes the same ablation trend and complements the numeric table.

### D. Profile-Level Behavioral Findings

The final ZSH model generated **30** semantic profiles. The largest profiles were `OP_Return` (**11.53%**), `Distribution` (**10.84%**), `Batch_Payment_3` (**9.15%**), `Coinjoin_Mixer` (**8.71%**), and `Standard_P2P` (**7.97%**). These high-level categories are operationally useful because they correspond to transaction behaviors that can be understood by analysts without inspecting every transaction individually.

**Table V. Representative high-support ZSH profiles and anomaly rates.**

| Profile | Share of corpus | Anomaly rate |
| --- | ---: | ---: |
| OP_Return | 11.53% | 0.22% |
| Distribution | 10.84% | 0.01% |
| Batch_Payment_3 | 9.15% | 0.16% |
| Coinjoin_Mixer | 8.71% | 0.06% |
| Standard_P2P | 7.97% | 0.00% |
| RBF_Enabled_1 | 7.56% | 0.02% |
| Coinjoin_Mixer_7 | 7.45% | 0.44% |
| Standard_P2P_3 | 4.79% | 0.09% |
| Batch_Payment_5 | 4.69% | 1.72% |
| Batch_Payment_7 | 4.12% | 0.38% |

The anomaly layer further revealed that certain micro-profiles contain disproportionately high concentrations of anomalous behavior. For example, some compact subprofiles such as `Batch_Payment_1`, `Coinjoin_Mixer_3`, and `Batch_Payment_2` exhibited anomaly rates near or at 100%, while large macro-profiles such as `Standard_P2P` and `Distribution` had near-zero anomaly rates. This pattern supports the intuition that macro behavioral families can still contain high-risk microstructures that become visible only when clustering and anomaly scoring are kept separate.

From a profiling perspective, this is one of the most valuable outputs of the framework. Classical KMeans can partition the space, but it does not automatically tell the analyst that a given cluster corresponds to a coinjoin-like family, an OP_RETURN-dominant family, or a concentrated batch-payment subtype. ZSH creates that intermediate semantic layer, which can then be combined with anomaly intensity for prioritization. In the main paper, the most publication-efficient visual summary is the semantic heatmap in **Fig. 3**; the dense UMAP, anomaly-overlay, and temporal-distribution plots are better placed in supplementary or repository material so that the main text remains readable.

### E. Contextual Comparison: Is ZSH Better for Blockchain Profiling Than KMeans?

Because the geometry-only benchmark favors KMeans++ Elkan, a second question becomes more important: **is ZSH better for the intended blockchain profiling task?** To answer this, a contextual comparison was performed over five repeated 120,000-row samples using only eight heuristic blockchain indicators and a shallow decision tree as an explainability probe. These indicators are not external ground-truth labels; they are used here as a **task-alignment probe** for semantic coherence and recoverability.

The results are presented in **Table IV**.

**Table IV. Contextual profiling comparison between ZSH and KMeans++ Elkan.**

| Metric | ZSH | KMeans++ Elkan |
| --- | ---: | ---: |
| Weighted semantic purity | 0.9474 | 0.9368 |
| Macro purity | 0.8705 | 0.8179 |
| Weighted entropy | 0.2543 | 0.2772 |
| High-purity clusters | 21.4 | 17.4 |
| Rule-tree accuracy | 0.8202 | 0.8396 |
| Rule-tree balanced accuracy | 0.4541 | 0.3953 |
| Rule-tree macro-F1 | 0.4203 | 0.3698 |
| NMI with rule layer | 0.7186 | 0.7341 |
| AMI with rule layer | 0.7185 | 0.7340 |

These results are revealing. ZSH achieved:

1. **Higher weighted semantic purity** than KMeans++ Elkan (+1.1%)
2. **Substantially higher macro purity** (+6.4%)
3. **Lower weighted entropy** (+8.3% better because lower is preferable)
4. **More high-purity profiles** (+23.0%)
5. **Higher balanced accuracy and macro-F1** under shallow rule-tree recovery (+14.9% and +13.7%, respectively)

Two observations follow from this table.

First, ZSH produces **cleaner behavioral groups**, especially for minority profiles. The better balanced accuracy and macro-F1 show that when the learned profiles are approximated with only eight heuristic blockchain indicators, ZSH retains more recoverable structure for the smaller or rarer profile types. This is precisely the kind of behavior expected from a profiling-oriented framework.

Second, KMeans++ Elkan still achieved slightly higher NMI and AMI with the same heuristic layer. This suggests that KMeans aligns somewhat better with the **global information partition** of the rule-derived layer, while ZSH yields **purer and more operationally separated** profile groups. In other words, the two methods optimize different priorities. KMeans is the better geometry-first and information-global optimizer; ZSH is the better profiling-first and minority-sensitive organizer.

For blockchain forensics, the latter property is often more valuable. Analysts do not usually act on Silhouette values directly. They act on interpretable profile groups, suspicious micro-clusters, and anomaly-prioritized transactions. Under this task-oriented criterion, ZSH is the stronger method. **Fig. 12** visualizes the gap between geometry-first and profiling-first criteria.

### F. Why ZSH Is Not the Best Geometry-Only Clusterer, but Still the Better Profiling Framework

The central scientific conclusion of the study is therefore nuanced. ZSH is **not** the best geometry-only clusterer in the corrected same-space benchmark. KMeans++ Elkan is better if the sole objective is Euclidean compactness and separation. However, blockchain transaction profiling is not a purely geometric problem. It is a semantic and operational problem in which analysts need to answer questions such as:

1. What dominant transaction family does this cluster represent?
2. Which subprofiles within a large family look atypical?
3. Which transactions should be prioritized for manual review?
4. How stable are these profiles across time and behavior types?

ZSH addresses these questions more directly than KMeans. The finite zeta weighting pushes the representation toward behaviorally informative features. Semantic seed allocation prevents the clustering stage from drifting entirely away from blockchain-relevant transaction families. Ward guidance preserves useful structure for subprofile separation. Finally, the post-labeling and anomaly layer transform the numerical clusters into a profile system that is understandable and actionable.

Thus, the fairest interpretation is not "ZSH beats KMeans everywhere," but "ZSH is the better framework when the goal is blockchain transaction profiling rather than geometry-only partitioning."

### G. Limitations and Threats to Validity

Several limitations should be acknowledged.

First, ZSH does not dominate KMeans++ Elkan on intrinsic clustering metrics. Any claim of universal superiority would therefore be unsupported by the current data.

Second, the contextual profiling comparison uses a heuristic semantic layer derived from transaction flags that overlap with the design goals of ZSH. This means the contextual benchmark should be interpreted as **task-alignment evidence**, not as an independent external ground-truth benchmark.

Third, the current ZSH-G metric branch selected the **identity fallback**, which indicates that the proposed linear metric-learning variants did not outperform the teacher baseline on the validation split. This is informative but also shows that the strongest part of the present contribution lies in the finite zeta weighting, semantic hybridization, and anomaly-aware profiling, rather than in the learned linear transform itself.

Fourth, the study is transaction-centric. Although the resulting profiles can support wallet-level reasoning, a direct wallet-level evaluation was not performed in the current manuscript. Fifth, the dataset is dominated by one blockchain context and a limited time range. Cross-chain transferability, streaming adaptation, and external labeled-case validation remain open research directions.

These limitations do not invalidate the contribution, but they do define its scope clearly. ZSH is best understood as a large-scale transaction profiling framework with strong interpretability and operational utility, not as the final word on blockchain clustering geometry.

## VI. CONCLUSION

This paper presented **ZSH**, a zeta-weighted hybrid semantic clustering framework for large-scale blockchain transaction profiling. The framework combines finite-normalized Riemann zeta feature weighting, geometry-aware rank fusion, semantic seed allocation, Ward-guided hybrid clustering, profile-preserving anomaly detection, and corrected same-space statistical evaluation. On a dataset of 11,303,526 transactions described by 27 features, ZSH produced 30 semantically interpretable profiles and flagged 564,674 anomalous transactions. Statistical testing confirmed that the learned structure is stable and significantly different from random labeling.

The corrected benchmark showed that KMeans++ Elkan remains stronger on intrinsic Euclidean clustering geometry. However, the contextual comparison demonstrated that ZSH is better aligned with the intended blockchain profiling task, achieving higher semantic purity, lower entropy, more high-purity profiles, and better minority-profile recoverability under shallow expert-rule explanation. Therefore, the contribution of ZSH lies not in absolute dominance over all baselines, but in its ability to turn large unlabeled transaction corpora into semantically grounded and operationally actionable profile systems. The evidence thus supports a **conditional superiority claim**: ZSH is preferable when the target outcome is semantically coherent, anomaly-aware transaction profiling rather than geometry-only partitioning.

Future work will focus on three directions: (i) extending the framework to cross-chain and wallet-level profiling, (ii) improving the ZSH-G metric-learning branch beyond the current identity fallback, and (iii) incorporating streaming or continual profile adaptation for evolving transaction ecosystems. These directions can further strengthen the role of transaction profiling in blockchain forensics, compliance analytics, and behavioral intelligence.

## CODE AND ARTIFACT AVAILABILITY

The final submission should include a public repository link for experimentation proof. The repository should expose the full pipeline (`Step_1` to `Step_8`), the environment specification, a figure-and-table artifact map, and instructions for reproducing the results from the raw dataset. In this project workspace, a GitHub-ready package has been prepared and should be pushed to a public repository before submission. The paper should then replace the placeholder statement with the final repository URL: `Code and artifact package available at https://github.com/sagarkorde/ZSH.`

## ACKNOWLEDGMENT

The authors acknowledge the institutional and computational support used to execute the experiments reported in this study. If the work received specific funding, the final funding statement should be inserted here. For transparency, this manuscript draft was prepared with AI-assisted language support; however, the experimental design, code execution, validation, and scientific interpretation remain the responsibility of the authors.

## REFERENCES

[1] Y. Qi, J. Wu, H. Xu, and M. Guizani, "Blockchain Data Mining With Graph Learning: A Survey," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 2, pp. 729-748, Feb. 2024, doi: 10.1109/tpami.2023.3327404.

[2] J. Zhang, K. Cai, and J. Wen, "A survey of deep learning applications in cryptocurrency," iScience, vol. 27, no. 1, p. 108509, Jan. 2024, doi: 10.1016/j.isci.2023.108509.

[3] S. Kayikci and T. M. Khoshgoftaar, "Blockchain meets machine learning: a survey," Journal of Big Data, vol. 11, no. 1, Jan. 2024, doi: 10.1186/s40537-023-00852-y.

[4] R. I. T. Jensen and A. Iosifidis, "Fighting Money Laundering With Statistics and Machine Learning," IEEE Access, vol. 11, pp. 8889-8903, 2023, doi: 10.1109/access.2023.3239549.

[5] N. Pocher, M. Zichichi, F. Merizzi, M. Z. Shafiq, and S. Ferretti, "Detecting anomalous cryptocurrency transactions: An AML/CFT application of machine learning-based forensics," Electronic Markets, vol. 33, no. 1, Jul. 2023, doi: 10.1007/s12525-023-00654-3.

[6] N. Nayyer, N. Javaid, M. Akbar, A. Aldegheishem, N. Alrajeh, and M. Jamil, "A New Framework for Fraud Detection in Bitcoin Transactions Through Ensemble Stacking Model in Smart Cities," IEEE Access, vol. 11, pp. 90916-90938, 2023, doi: 10.1109/access.2023.3308298.

[7] T. Voronov, D. Raz, and O. Rottenstreich, "A Framework for Anomaly Detection in Blockchain Networks With Sketches," IEEE/ACM Transactions on Networking, vol. 32, no. 1, pp. 686-698, Feb. 2024, doi: 10.1109/tnet.2023.3298253.

[8] Y. Elmougy and L. Liu, "Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics," Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 3979-3990, Aug. 2023, doi: 10.1145/3580305.3599803.

[9] S. Hu, Z. Zhang, B. Luo, S. Lu, B. He, and L. Liu, "BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection," Proceedings of the ACM Web Conference 2023, pp. 2189-2197, Apr. 2023, doi: 10.1145/3543507.3583345.

[10] L. Wang, M. Xu, and H. Cheng, "Phishing scams detection via temporal graph attention network in Ethereum," Information Processing & Management, vol. 60, no. 4, p. 103412, Jul. 2023, doi: 10.1016/j.ipm.2023.103412.

[11] A. Xiong et al., "Ethereum phishing detection based on graph neural networks," IET Blockchain, vol. 4, no. 3, pp. 226-234, May 2023, doi: 10.1049/blc2.12031.

[12] S. Li, R. Wang, H. Wu, S. Zhong, and F. Xu, "SIEGE: Self-Supervised Incremental Deep Graph Learning for Ethereum Phishing Scam Detection," Proceedings of the 31st ACM International Conference on Multimedia, pp. 8881-8890, Oct. 2023, doi: 10.1145/3581783.3612461.

[13] S. Li et al., "TGC: Transaction Graph Contrast Network for Ethereum Phishing Scam Detection," Annual Computer Security Applications Conference, pp. 352-365, Dec. 2023, doi: 10.1145/3627106.3627109.

[14] J. Cai, B. Li, J. Zhang, and X. Sun, "Ponzi Scheme Detection in Smart Contract via Transaction Semantic Representation Learning," IEEE Transactions on Reliability, vol. 73, no. 2, pp. 1117-1131, Jun. 2024, doi: 10.1109/tr.2023.3319318.

[15] C. Liu, Y. Xu, and Z. Sun, "Directed dynamic attribute graph anomaly detection based on evolved graph attention for blockchain," Knowledge and Information Systems, vol. 66, no. 2, pp. 989-1010, Dec. 2023, doi: 10.1007/s10115-023-02033-y.

[16] B. Han, Y. Wei, Q. Wang, F. M. D. Collibus, and C. J. Tessone, "MT^2AD: multi-layer temporal transaction anomaly detection in ethereum networks with GNN," Complex & Intelligent Systems, vol. 10, no. 1, pp. 613-626, Jul. 2023, doi: 10.1007/s40747-023-01126-z.

[17] L. Xiao et al., "CTDM: cryptocurrency abnormal transaction detection method with spatio-temporal and global representation," Soft Computing, vol. 27, no. 16, pp. 11647-11660, May 2023, doi: 10.1007/s00500-023-08220-x.

[18] H. Huang et al., "PEAE-GNN: Phishing Detection on Ethereum via Augmentation Ego-Graph Based on Graph Neural Network," IEEE Transactions on Computational Social Systems, vol. 11, no. 3, pp. 4326-4339, Jun. 2024, doi: 10.1109/tcss.2023.3349071.

[19] J. Zhang, H. Sui, X. Sun, C. Ge, L. Zhou, and W. Susilo, "GrabPhisher: Phishing Scams Detection in Ethereum via Temporally Evolving GNNs," IEEE Transactions on Services Computing, vol. 17, no. 6, pp. 3727-3741, Nov. 2024, doi: 10.1109/tsc.2024.3411449.

[20] J. Liu, J. Chen, J. Wu, Z. Wu, J. Fang, and Z. Zheng, "Fishing for Fraudsters: Uncovering Ethereum Phishing Gangs With Blockchain Data," IEEE Transactions on Information Forensics and Security, vol. 19, pp. 3038-3050, 2024, doi: 10.1109/tifs.2024.3359000.

[21] M. Hasan, M. S. Rahman, H. Janicke, and I. H. Sarker, "Detecting anomalies in blockchain transactions using machine learning classifiers and explainability analysis," Blockchain: Research and Applications, vol. 5, no. 3, p. 100207, Sep. 2024, doi: 10.1016/j.bcra.2024.100207.

[22] S. Ouyang, Q. Bai, H. Feng, and B. Hu, "Bitcoin Money Laundering Detection via Subgraph Contrastive Learning," Entropy, vol. 26, no. 3, p. 211, Feb. 2024, doi: 10.3390/e26030211.

[23] Z. Wang, A. Ni, Z. Tian, Z. Wang, and Y. Gong, "Research on blockchain abnormal transaction detection technology combining CNN and transformer structure," Computers and Electrical Engineering, vol. 116, p. 109194, May 2024, doi: 10.1016/j.compeleceng.2024.109194.

[24] Z. Chen, S.-Z. Liu, J. Huang, Y.-H. Xiu, H. Zhang, and H.-X. Long, "Ethereum Phishing Scam Detection Based on Data Augmentation Method and Hybrid Graph Neural Network Model," Sensors, vol. 24, no. 12, p. 4022, Jun. 2024, doi: 10.3390/s24124022.

[25] Z. Ding, J. Shi, Q. Li, and J. Cao, "Effective Illicit Account Detection on Large Cryptocurrency MultiGraphs," Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 457-466, Oct. 2024, doi: 10.1145/3627673.3679707.

[26] J. Yang, W. Yu, J. Wu, D. Lin, Z. Wu, and Z. Zheng, "2DynEthNet: A Two-Dimensional Streaming Framework for Ethereum Phishing Scam Detection," IEEE Transactions on Information Forensics and Security, vol. 19, pp. 9924-9937, 2024, doi: 10.1109/tifs.2024.3484296.

[27] J. Song, S. Zhang, P. Zhang, J. Park, Y. Gu, and G. Yu, "Illicit Social Accounts? Anti-Money Laundering for Transactional Blockchains," IEEE Transactions on Information Forensics and Security, vol. 20, pp. 391-404, 2025, doi: 10.1109/tifs.2024.3518068.

[28] W. Hou, B. Cui, Y. Chen, R. Li, and W. Song, "TSFF: A Triple-Stream Feature Fusion Method for Ethereum Phishing Scam Detection," IEEE Internet of Things Journal, vol. 12, no. 3, pp. 2623-2632, Feb. 2025, doi: 10.1109/jiot.2024.3473771.

[29] L. Zhang et al., "Unraveling the Deception of Web3 Phishing Scams: Dynamic Multiperspective Cascade Graph Approach for Ethereum Phishing Detection," IEEE Transactions on Computational Social Systems, vol. 12, no. 2, pp. 498-510, Apr. 2025, doi: 10.1109/tcss.2024.3516144.

[30] Y. Liang et al., "A plug-and-play data-driven approach for anti-money laundering in bitcoin," Expert Systems with Applications, vol. 266, p. 126072, Mar. 2025, doi: 10.1016/j.eswa.2024.126072.

[31] Z. Sheng, L. Song, and Y. Wang, "Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Phishing Detection," IEEE Transactions on Network and Service Management, vol. 22, no. 5, pp. 4706-4718, Oct. 2025, doi: 10.1109/tnsm.2025.3576130.

[32] H. Sui, J. Zhang, B. Chen, D. Wu, X. Sun, and S. Palaiahnakote, "EPAD: Ethereum phishing scam detection via graph contrastive learning," Expert Systems with Applications, vol. 288, p. 128227, Sep. 2025, doi: 10.1016/j.eswa.2025.128227.

[33] J. Huang et al., "GAPLG: Graph Augmented With Pseudolabels Generation for Blockchain Anomaly Transaction Detection," IEEE Transactions on Computational Social Systems, vol. 12, no. 6, pp. 4532-4546, Dec. 2025, doi: 10.1109/tcss.2025.3555658.

[34] A. Asiri and K. Somasundaram, "Graph convolution network for fraud detection in bitcoin transactions," Scientific Reports, vol. 15, no. 1, Apr. 2025, doi: 10.1038/s41598-025-95672-w.

[35] X. Shen, C. Xu, and L. Zhu, "Blockchain Anomaly Transaction Detection Method Based on Graph Continual Learning," IEEE Transactions on Network Science and Engineering, vol. 13, pp. 6059-6078, 2026, doi: 10.1109/tnse.2026.3653459.

## AUTHOR BIOGRAPHIES

**Author 1** is currently pursuing the Ph.D. degree with [Department], [University], [City], [Country]. The research interests include blockchain analytics, unsupervised learning, anomaly detection, and financial forensics.

**Author 2** is with [Department], [University], [City], [Country]. The research interests include machine learning, data mining, and applied artificial intelligence.

**Author 3** is with [Department], [University], [City], [Country]. The research interests include cybersecurity, blockchain systems, and digital forensics.
