# Draft — Back matter

::: {custom-style="MDPI_6.2_back_matter"}

**Supplementary Materials:** The following supporting information can be
downloaded at https://www.mdpi.com/article/[DOI]/s1. {SUPP_LIST}. The
supplementary file also gives the full definitions of the measures summarised
in Section 5.3.

**Author Contributions:** Conceptualization, S.D.K.; methodology, S.D.K.;
software, S.D.K.; validation, S.D.K.; formal analysis, S.D.K.; investigation,
S.D.K.; resources, S.D.K.; data curation, S.D.K.; writing—original draft
preparation, S.D.K.; writing—review and editing, S.D.K. and N.M.S.;
visualization, S.D.K.; supervision, N.M.S.; project administration, N.M.S. All
authors have read and agreed to the published version of the manuscript.
<!-- AUTHOR CHECK: confirm roles; unchanged from the first submission. -->

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
<!-- AUTHOR ACTION: create the Zenodo (or IEEE DataPort) record for the
prospective sample and replace {ZENODO_DOI}. -->

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
<!-- AUTHOR CHECK: MDPI asks for a precise description of AI use. Edit this so
it matches what you did yourself and what the tool did; remove anything here
that overstates the tool's role or understates yours. -->

**Conflicts of Interest:** The authors declare no conflicts of interest. The
first author is also an author of the transaction data set used in this study.
:::
