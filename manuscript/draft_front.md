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
does; unsupervised profiles describe that activity without labels. They are rarely
tested against information the clustering never saw, and a lift over the base rate
cannot separate a weak method from a rare property. We evaluate ZSH, which weights
twelve features by the rank of their mutual information with a proxy partition, starts
K-means from merged micro-clusters and caps cluster size; the plan and code were frozen
before evaluation. It was fitted on 3.3 million transactions from 2022–2023 and tested
on 2.6 million from 2024 and 456,292 collected to August 2026. ZSH produced 31 profiles,
reaching 78–84% of the ceiling each base rate sets for the three dominant single-script
classes, near 23% for P2SH and mixed-script inputs, and under 11% for the five least
frequent annotations (base rates 1.4% to 0.03%). Neither oracle weights nor six other
clustering families changed this, yet for P2SH inputs supervision on the same features
reached 95% against the profiles' 24%. Refits agreed at an adjusted Rand index of 0.83,
yet by 2026 three profiles held three quarters of the sample and none survived an
unconstrained refit; constraining it to the previous centroids followed 24 of 31
profiles in 2024 but only 14 of 31 in 2026. For most annotations the binding limit is
the clustering objective rather than the twelve features, though exchange tags improve
markedly with richer features. These profiles are a readable map of activity, not a
detector.
:::

::: {custom-style="MDPI_1.8_keywords"}
**Keywords:** Bitcoin; cluster stability; clustering; CoinJoin; distribution shift; Elliptic data set; feature weighting; transaction profiling.
:::
