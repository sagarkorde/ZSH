"""Assemble manuscript.md from the section drafts, the manuscript tables and the figures,
then build the MDPI .docx with build_manuscript.py.

* Table and figure keys in the text ({T_key}, {F_key}) are numbered in order of first
  mention; each table or figure is placed after the paragraph (or list) of its first
  mention. Appendix items are numbered A1, A2, ... and placed in the appendix.
* HTML comments are removed; comments that start with AUTHOR are collected in
  author_notes.md.
* Remaining {…} placeholders outside math are listed at the end and make the build fail
  unless --draft is given.

Usage: python assemble.py [--draft]
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
def _find_repo():
    """Repository root: $ZSH_REPO, else the parent when this lives in manuscript/."""
    env = os.environ.get("ZSH_REPO")
    if env:
        return Path(env)
    if (HERE.parent / "results").is_dir():
        return HERE.parent
    raise SystemExit(
        "Cannot locate the repository. Set ZSH_REPO to the checkout that holds "
        "results/, or run this script from manuscript/ inside it.")


REPO = _find_repo()
TABLES = REPO / "results" / "manuscript"
FIGS = REPO / "results" / "figures"
# Where the built .docx files go. Defaults to the repository root; set
# ZSH_DOCX_OUT to the folder you assemble the submission in.
OUT_DIR = Path(os.environ.get("ZSH_DOCX_OUT", HERE.parent))
DRAFT = "--draft" in sys.argv

SECTIONS = ["draft_front.md", "draft_intro.md", "draft_related.md", "draft_methods.md",
            "draft_results.md", "draft_discussion.md", "draft_conclusion.md", "draft_backmatter.md"]

# Tables and figures published as Supplementary Materials (a separate file) rather than inside
# the article. They are numbered S1, S2, ... in order of first mention in the article.
SUPP = ["T_rules", "T_profiles", "T_factorial", "T_heuristic", "T_cjsource", "T_atypicality",
        "T_sensitivity", "T_proxy_k", "T_contrasts", "T_loo", "T_support", "T_representatives",
        "F_weights", "F_profiles", "F_factorial", "F_atypicality", "F_methods", "F_drift",
        "T_prespec", "T_weighting"]

TAB = {
    "T_data": ("T_data.md", "Data used in this study. The development and test periods are parts of the "
               "published Bitcoin transaction sample [@Korde2026]; the prospective sample was collected after "
               "the analysis code had been frozen; the Elliptic data follow the temporal split of the "
               "original study [@Weber2019]."),
    "T_rules": ("T_rules.md", "Count-rule flags of the published sample. Each flag is a fixed rule on the input "
                "and output counts; the rules can overlap. The last column gives the accuracy with which a "
                "depth-3 decision tree recovers each rule from the other candidate features on development data."),
    "T_features": ("T_features.md", "The twelve clustering features, ranked by their mutual information with "
                   "the proxy partition on development data, and their rank-power weights (s = 1.5)."),
    "T_annotations": ("T_annotations.md", "Annotations that are not clustering inputs, as shares of all "
                      "transactions. L2: input script class (all inputs of one class, otherwise mixed); L3: "
                      "OP_RETURN protocol; L4: GraphSense entity tag on any input or output address; L5: "
                      "equal-output CoinJoin rule, which needs individual output values. Prospective shares use "
                      "the design weights."),
    "T_profiles": ("T_profiles.md", "The 31 ZSH profiles. Dev., Test and Prosp.: share of the transactions of "
                   "each period. In, Out, Value (total input value) and Fee rate (sat/vB) are medians over the "
                   "development members; the main input script with its share, the share signalling "
                   "replace-by-fee (RBF) and the share with an exchange tag also refer to development members. "
                   "The last column gives the Runes share among the test-period members."),
    "T_methods": ("T_methods.md", "Methods fitted on the same 200,000 development transactions with the same "
                  "number of clusters. Geometry is measured on a separate 200,000-row development sample in the "
                  "common unweighted space (Silh., Silhouette; DBI, Davies–Bouldin index; CH, Calinski–Harabasz "
                  "index) and, for the Silhouette, also in each method's own space. The largest cluster and the "
                  "median AP lift over the eleven non-input annotations refer to the test period. The last "
                  "column counts annotations for which ZSH has a significantly higher or lower AP lift than the "
                  "method (Holm-adjusted p < 0.05, 95% interval excluding zero). HDBSCAN chooses its own number "
                  "of clusters and cannot assign new transactions, so it is described on its fitting sample only."),
    "T_factorial": ("T_factorial.md", "Arms of the factorial design, all fitted on the full development period. "
                    "K* = 31 is the number of ZSH profiles; A5 uses the K of A2. Silh.: Silhouette in the common "
                    "unweighted space. Median AP lift over the eleven non-input annotations in the test period."),
    "T_stability": ("T_stability.md", "Reproducibility of the whole pipeline. ARI, AMI and VI (variation of "
                    "information) compare each refit with the fit on all development data on a fixed set of "
                    "200,000 development transactions (mean ± standard deviation for ARI); pairwise ARI compares "
                    "refits with each other (lowest pair in brackets). Centroid shift: mean displacement of "
                    "matched centroids relative to the within-cluster root-mean-square distance. Kendall τ: "
                    "agreement of each refit's feature ranking with the full-data ranking. Last column: clusters "
                    "whose mean best Jaccard similarity across the bootstrap refits is at least 0.75, between 0.5 "
                    "and 0.75, and below 0.5 [@Hennig2007]."),
    "T_transfer": ("T_transfer.md", "Transfer of the development-fitted partitions to later periods. ARI, AMI "
                   "and the best Jaccard similarities compare the transferred partition with a refit of the same "
                   "method on the later period. Kendall τ compares the feature ranking of the ZSH refit with the "
                   "development ranking. The last two columns describe where the Runes transactions fall."),
    "T_concentration": ("T_concentration.md", "Cross-fitted concentration of the non-input annotations by ZSH "
                        "and by K-means++ with the same K, both fitted on all development data. AP lift: average "
                        "precision of the cluster ranking divided by the base rate (KM++: the same quantity for "
                        "K-means++). Prec.: precision at 25% coverage of the positives, reached with the number of "
                        "top-ranked profiles given in the next column; dividing it by the base rate gives the "
                        "enrichment. Intervals from 1,000 block bootstrap resamples; p-values Holm-adjusted over "
                        "the annotations of each period."),
    "T_heuristic": ("T_heuristic.md", "The count rule 'more than three inputs and more than three outputs', "
                    "published as 'coinjoin-like', compared with the equal-output CoinJoin rule. For 2022–2024, "
                    "estimates combine random samples of flagged and unflagged transactions, weighted by stratum "
                    "size, with parametric bootstrap intervals. For the prospective sample they use all "
                    "transactions and the design weights, with intervals from resampling blocks within months."),
    "T_elliptic": ("T_elliptic.md", "Concentration of illicit transactions on Elliptic. Models were fitted on "
                   "all transactions of time steps 1–34; clusters were ranked by their illicit rate among the "
                   "labelled training transactions and evaluated on the labelled transactions of steps 35–49 "
                   "(base rate 6.5%). AF-165: all features; LF-93: local features only. Precision is read at 25% "
                   "coverage of the test illicit transactions. Intervals from 1,000 resamples of time steps. The "
                   "random forest is a supervised reference trained on the labelled training transactions; its "
                   "precision and recall refer to its default decision threshold."),
    "T_atypicality": ("T_atypicality.md", "Atypicality scores against illicit status on the Elliptic test steps "
                      "and, as exploratory analyses, against annotations of the Bitcoin data. A ROC-AUC below 0.5 "
                      "means that positives receive lower scores. H6 is supported when the upper 95% bound is at "
                      "most 0.5. Intervals from 1,000 resamples of time steps (Elliptic) or 200 resamples of blocks "
                      "(Bitcoin)."),
    "T_sensitivity": ("T_sensitivity.md", "Sensitivity of ZSH to its settings. Each variant was fitted on the "
                      "same 1,000,000 development transactions with one setting changed and evaluated on the test "
                      "period. The last column counts non-input annotations whose AP lift is significantly "
                      "higher or lower than for the reference (Holm-adjusted p < 0.05, 95% interval excluding "
                      "zero); with 2.6 million test transactions, even small differences are significant. "
                      "Silhouettes are estimated from random draws, so identical partitions can differ in the "
                      "third decimal."),
    "T_cjsource": ("T_cjsource.md", "The two CoinJoin rules against an external list of Wasabi 2.x CoinJoin "
                   "transactions with a known coordinator [@Svenda2026]; exploratory, added after the freeze. Recall is the "
                   "share of the listed transactions that the rule flags, with a Wilson interval; the equal-output "
                   "rule needs individual output values and can therefore be evaluated only on the prospective "
                   "sample. The list covers coordinators active after mid-2024, which is why it matches no "
                   "development transaction."),
    "T_oracle": ("T_oracle.md", "An upper bound for the weighting; exploratory, added after the freeze. Each row refits the "
                 "pipeline on the same 1,000,000 development transactions with weights computed from the "
                 "annotation itself and evaluates it on the test period. ZSH lift: the unsupervised reference "
                 "fitted on the same sample. Attained: share of the ceiling 1/π, that is the average precision. "
                 "'Best other oracle' is the highest lift this annotation reaches under a weighting derived from "
                 "one of the other three annotations."),
    "T_matching": ("T_matching.md", "Following profiles across a refit; exploratory, added after the freeze. Each of the 31 "
                   "development profiles is paired with one cluster of the refit of the same period by "
                   "minimum-cost assignment on the transactions they share; J is the Jaccard similarity of the "
                   "two member sets. 'Their share' is the share of the period's transactions held by the "
                   "profiles matched at 0.5. The 0.5 criterion is a minimum-continuity threshold: it asks only "
                   "whether a profile remains recognisable across the refit, and is deliberately weaker than the "
                   "0.75 used in Section 6.3 as a criterion for cluster stability in Hennig's sense."),
    "T_bench_battery": ("T_bench_battery.md", "The whole evaluation applied to seven clustering families; exploratory, added after the freeze. Every method is fitted on the development period at K = 31 and uses the same twelve features, with one exception: the partial reproduction of Vlahavas et al. uses the five features of that study, as in Table {T_methods}; BIRCH and the Vlahavas reproduction are fitted on a 1,000,000-transaction subsample and Ward on its own 30,000-row subsample, as in Table {T_methods}. Largest: share of the period transactions held by the largest cluster. Refit ARI: mean adjusted Rand index between the development partition and ten block-bootstrap refits. ARI: agreement between the transferred partition and a refit on the period itself. Matched: profiles whose minimum-cost match with that refit reaches a Jaccard similarity of 0.5. ZSH chooses its own number of clusters when refitted (37 in the test period, 38 in the prospective sample); the other families are refitted at K = 31."),
    "T_warm": ("T_warm.md", "A refit constrained to the previous centroids; exploratory, added after the freeze. Free refit: the whole pipeline refitted on the period, as in Table {T_matching}. Constrained: the scaler and the rank-power weights of the frozen model are kept and only the centroids are re-estimated, started at the development centroids. Followed: profiles whose minimum-cost match with the refit reaches a Jaccard similarity of 0.5, the minimum-continuity threshold of Table {T_matching} rather than the stricter 0.75 stability criterion, and the share of the period transactions they hold. Median J: median Jaccard of the matched pairs. Attained: median share of the ceiling over the annotations of that period, for the frozen model and for the constrained refit."),
    "T_prespec": ("T_prespec.md", "Analyses of this study, by when they were specified and what kind of evidence they provide. All non-exploratory analyses had their hypotheses written into the analysis plan and their code frozen (repository tag `v2-frozen`) before the reported runs. They are nevertheless labelled *frozen reanalysis* rather than confirmatory, because an earlier submitted version of this work had already examined the same Bitcoin corpus and the complete labelled Elliptic data set: the freeze prevents the analysis from being tuned to its result, but the outcomes were not unknown to the authors. Rows marked *prospective on 2024-26* are additionally prospective when applied to the 2024-2026 sample, whose transactions had not been mined at the freeze. Exploratory analyses were added after the freeze; each is recorded with its reason and commit in the deviations log and labelled where it appears."),
    "T_ceiling": ("T_ceiling.md", "What the same twelve features give a supervised model; exploratory, added after the freeze. For each annotation a gradient-boosted tree is fitted on the twelve clustering features and its score is cut into 31 cells, which are then ranked and scored exactly as the profiles are, so the columns are comparable. Equal cells: cells of equal size, which cannot isolate a rare annotation whatever the score. Benchmark: cell sizes free (K-means on the score), fitted on one block-parity half of the test period and scored on the other. A year earlier: cell sizes free, fitted on the development period and therefore using no test-period label. All values are shares of the ceiling 1/π. The benchmark is fitted for one annotation at a time; Section S1 reports a single partition built from all eleven."),
    "T_bench_attained": ("T_bench_attained.md", "Share of the attainable concentration reached by each family on the test period; exploratory, added after the freeze. Ceiling: the largest AP lift an annotation of this base rate can reach, 1/π; the attained share is the average precision itself. Columns, in order: ZSH, K-means++, mini-batch K-means, the diagonal Gaussian mixture, BIRCH, Ward and the partial reproduction of Vlahavas et al., all at K = 31."),
    "T_richer": ("T_richer.md", "Address-level features against the twelve, on the prospective sample; exploratory, added after the freeze. Twelve features built from the cached transaction records — input and output address reuse inside the sample, paying back to an own input address, witness items and bytes, sigops, locktime and sequence use, and the dispersion of input and output values — are added to the twelve of Table 2. Frozen: the development model transferred. Refit 12 and Refit 24: the pipeline refitted on this period with twelve and with twenty-four features. Sup.: the supervised benchmark of Table {T_ceiling} computed on this period, with cell sizes free. All values are shares of the ceiling 1/π. Input script types are excluded because they are annotations."),
    "T_proxy_k": ("T_proxy_k.md", "Sensitivity to the size of the proxy partition; exploratory, added after the freeze. "
                  "Each row refits the whole pipeline on the same 1,000,000 development transactions with the "
                  "given number of proxy clusters and is evaluated on the test period. τ: Kendall correlation of "
                  "the resulting feature ranking with the reference ranking. Silh.: Silhouette in the common "
                  "unweighted space. Median lift: median AP lift over the eleven non-input annotations; P2PKH and "
                  "Omni are given as the annotations that move most. The last column counts annotations that are "
                  "significantly higher or lower than the reference (Holm-adjusted p < 0.05)."),
}
APP_TAB = {
    "T_contrasts": ("T_contrasts.md", "Primary contrasts of H1 (rank-power against uniform weights, A3 − A4) "
                    "and H2 (refinement against none, A1 − A3) on the test period: differences in AP lift with "
                    "95% block-bootstrap intervals and Holm-adjusted p-values. With 1,000 resamples and eleven "
                    "annotations, the smallest attainable adjusted p-value is 0.022."),
    "T_loo": ("T_loo.md", "Leave-one-family-out analysis of the seeded variant: AP lift of each count-rule family "
              "on the test period for the unseeded model, the model seeded with all families, and the model "
              "seeded without that family. For the OP_RETURN and replace-by-fee families the corresponding flag "
              "was also removed from the features and the whole pipeline was refitted."),
    "T_support": ("T_support.md", "Share of each profile in every period with a 95% interval from resampling "
                  "blocks (1,000 resamples; the prospective sample uses its design weights), and the mean "
                  "cluster-wise Jaccard similarity across the 30 block-bootstrap refits of Section 6.3 with its "
                  "5th and 95th percentiles. Exploratory, added after the freeze."),
    "T_weighting": ("T_weighting.md", "Design-weighted against unweighted concentration in the prospective "
                    "sample. The weighted columns are the estimates reported in the article: they use the "
                    "design weights of the sampling scheme and a month-stratified block bootstrap, and they "
                    "estimate the quantity for the sampled population. The unweighted columns treat the "
                    "sampled transactions as the population and are therefore sample-specific; they are given "
                    "so that the effect of the weighting can be read directly. Positives are sampled counts "
                    "and are identical for both, because target eligibility is deliberately unweighted. "
                    "Attained: average precision, the share of the ceiling 1/π."),
    "T_representatives": ("T_representatives.md", "One-line description of each profile and the development "
                          "transaction closest to its centroid. Descriptions are built from the medians and the "
                          "annotation shares of each profile's members (Section 4.6); the share is of development "
                          "transactions. Transaction ids are printed in two halves so that they break across "
                          "lines."),
}
FIG = {
    "F_pipeline": ("F1_pipeline.png", 6.5, "The ZSH pipeline. The shaded steps form the method; the "
                   "atypicality score is a separate component. All steps are fitted on development data, and "
                   "later data pass through the frozen model."),
    "F_design": ("F2_design.png", 6.5, "Study periods. The analysis code was frozen before any test-period, "
                 "prospective or Elliptic test result was computed."),
    "F_weights": ("F3_weights.png", 3.6, "Rank-power weights of the twelve features on development data, with "
                  "the mutual information (MI) of each feature with the proxy partition."),
    "F_profiles": ("F4_profiles.png", 6.5, "Profile descriptors on development data. Left block: medians of "
                   "five features, log-transformed and rescaled to [0, 1] across profiles. Middle block: shares of "
                   "members with each input script class, an OP_RETURN output, replace-by-fee signalling and an "
                   "exchange tag. Right column: Runes share among the test-period members."),
    "F_methods": ("F5_methods.png", 6.5, "AP lift of the eleven non-input annotations on the test period for "
                  "methods fitted on the same development sample with the same K (log scale). Bars are 95% block "
                  "bootstrap intervals for ZSH and K-means++; grey points are the other baselines."),
    "F_factorial": ("F6_factorial.png", 6.5, "Primary contrasts of the factorial design on the test period: "
                    "(left) rank-power against uniform weights without refinement (H1); (right) refinement "
                    "against none with rank-power weights (H2). Differences in AP lift with 95% intervals; filled "
                    "points are significant after Holm adjustment."),
    "F_stability": ("F7_stability.png", 6.5, "Reproducibility. (a) Agreement of seed and block-bootstrap refits "
                    "with the fit on all development data. (b) Mean best Jaccard similarity of each cluster "
                    "across the 30 bootstrap refits; dotted lines mark the thresholds 0.5 and 0.75."),
    "F_drift": ("F8_drift.png", 6.5, "Monthly shares of the development-fitted profiles. Test and prospective "
                "transactions were assigned by the frozen model; prospective shares use the design weights. "
                "Vertical lines mark the start of the test period, the Runes launch and the start of the "
                "prospective sample."),
    "F_concentration": ("F9_curves.png", 6.5, "Enrichment (precision divided by base rate) against coverage "
                        "of the positives for selected annotations, averaged over the two cross-fitting "
                        "directions."),
    "F_elliptic": ("F10_elliptic.png", 6.5, "Elliptic test steps. (a) Enrichment against coverage of the test "
                   "illicit transactions. (b) Precision and recall, by time step, of the clusters that cover 25% "
                   "of the training illicit transactions; the dotted line marks the dark-market closure at "
                   "step 43."),
    "F_atypicality": ("F11_atypicality.png", 3.6, "ROC curves of atypicality scores against illicit status on "
                      "the Elliptic test steps. Curves below the diagonal mean that illicit transactions "
                      "receive lower scores."),
    "F_sensitivity": ("F12_sensitivity.png", 6.5, "AP lift on the test period for each sensitivity variant and "
                      "annotation (cell text). Colour shows the change relative to the reference variant in "
                      "the top row on a log2 scale; K is given for each variant."),
    "F_bench": ("F13_bench.png", 6.5, "Share of the attainable concentration reached on the test period by each clustering family; exploratory, added after the freeze. Cells give the attained share as a percentage of the ceiling 1/π; the base rate of each annotation is printed under its name. All families are fitted at K = 31 on the same twelve features, except the partial reproduction of Vlahavas et al., which uses the five features of that study."),
}


def strip_comments(md, notes, name):
    def repl(m):
        body = m.group(1).strip()
        if body.startswith("AUTHOR"):
            notes.append(f"- {name}: {body}")
        return ""
    return re.sub(r"<!--(.*?)-->", repl, md, flags=re.S)


def shift_headings(md):
    out = []
    for line in md.split("\n"):
        m = re.match(r"^(#{2,4}) (.*)$", line)
        out.append(("#" * (len(m.group(1)) - 1) + " " + m.group(2)) if m else line)
    return "\n".join(out)


def blocks_of(md):
    """Split into blocks separated by blank lines; list items are kept with their list."""
    parts = re.split(r"\n\s*\n", md)
    merged = []
    for p in parts:
        is_list = bool(re.match(r"^\s*(?:[*-]|\d+\.) ", p))
        if merged and is_list and merged[-1][1]:
            merged[-1] = (merged[-1][0] + "\n\n" + p, True)
        else:
            merged.append((p, is_list))
    return [p for p, _ in merged]


def table_block(label, key, fname, caption):
    path = TABLES / fname
    body = path.read_text(encoding="utf-8").strip() if path.exists() else \
        f"| missing |\n|---|\n| {fname} not built yet |"
    return f"Table: **Table {label}.** {caption}\n\n{body}"


def figure_block(label, key, fname, width, caption):
    src = FIGS / fname
    dst = HERE / "figs" / fname
    dst.parent.mkdir(exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return f"![**Figure {label}.** {caption}](figs/{fname}){{width={width}in}}"
    return f"**[Figure {label} missing: {fname}]** {caption}"


SUPP_HEADER = """# Supplementary Materials

**ZSH: Rank-Weighted Profiling of Bitcoin Transactions and a Pre-Specified Evaluation of What the
Profiles Capture**

Sagar D. Korde and Narendra M. Shekokar

This file holds the tables and figures that the article cites as Table S1, Table S2, ... and
Figure S1, Figure S2, ..., together with the full definitions of the measures summarised in
Section 5.3 of the article. Section, table and figure numbers without an S refer to the article.
"""


def write_supplementary(tnum, fnum, supp_t, supp_f, notes, label):
    """Assemble the Supplementary Materials file and build its .docx."""
    prose = ""
    sp = HERE / "draft_supplementary.md"
    if sp.exists():
        prose = strip_comments(sp.read_text(encoding="utf-8"), notes, sp.name)
        prose = re.sub(r"^# Draft[^\n]*\n", "", prose).strip()
    parts = [SUPP_HEADER.strip()]
    if prose:
        parts.append(prose)
    if supp_t:
        parts.append("## Supplementary tables")
        for k in supp_t:
            fname, cap = TAB.get(k) or APP_TAB[k]
            parts.append(table_block(tnum[k], k, fname, cap))
    if supp_f:
        parts.append("## Supplementary figures")
        for k in supp_f:
            parts.append(figure_block(fnum[k], k, *FIG[k]))
    parts.append("# References" + "\n\n" + "<!-- REFERENCES -->")
    text = re.sub(r"\{(T_[A-Za-z_]+|F_[A-Za-z_]+)\}", label, "\n\n".join(parts))
    out = HERE / "supplementary.md"
    out.write_text(text + "\n", encoding="utf-8")
    docx = OUT_DIR / ("ZSH_supplementary_v2_draft.docx" if DRAFT else "ZSH_supplementary_v2.docx")
    subprocess.run([sys.executable, str(HERE / "build_manuscript.py"), str(out),
                    str(HERE / "references.json"), str(docx)], check=True)
    print(f"supplementary: {len(supp_t)} tables, {len(supp_f)} figures -> {docx.name}")


def build_letters(label):
    """Write one self-contained response letter per reviewer, plus the combined file."""
    letters = HERE / "response_to_reviewers.md"
    if not letters.exists():
        print("response_to_reviewers.md not present; skipping the reply letters")
        return ""
    src = letters.read_text(encoding="utf-8")
    src = re.sub(r"\{(T_[A-Za-z_]+|F_[A-Za-z_]+)\}", label, src)
    (HERE / "response_resolved.md").write_text(src, encoding="utf-8")

    op_path = HERE / "letter_openings.md"
    openings = {}
    if op_path.exists():
        otext = re.sub(r"\{(T_[A-Za-z_]+|F_[A-Za-z_]+)\}", label, op_path.read_text(encoding="utf-8"))
        for m in re.finditer(r"^## Reviewer (\d+)\s*\n(.*?)(?=\n## |\Z)", otext, re.S | re.M):
            openings[m.group(1)] = m.group(2).strip()

    head = src[:src.index("## Summary of the main changes")]
    title_line = re.search(r"^\*\*Manuscript:\*\*.*$", head, re.M)
    title_line = title_line.group(0) if title_line else ""
    summary_start = src.index("## Summary of the main changes")
    first_rev = src.index("## Reviewer 1")
    summary = src[summary_start:first_rev].rstrip().rstrip("-").rstrip()

    parts = re.split(r"\n(?=## Reviewer \d+\s*\n)", src[first_rev:])
    out = []
    for part in parts:
        m = re.match(r"## Reviewer (\d+)", part)
        if not m:
            continue
        n = m.group(1)
        body = part[part.index("\n"):].strip()
        doc = [f"# Response to Reviewer {n}", "", title_line, ""]
        if n in openings:
            doc += [openings[n], ""]
        doc += [summary, "",
                "In the comment headings, table, figure, equation and line numbers refer to the first "
                "version; in our responses they refer to the revised manuscript unless marked \"old\".",
                "", "---", "", f"## Comments of Reviewer {n}", "", body, "",
                "We are grateful for the time these comments took, and we believe the article is a great "
                "deal more trustworthy for them."]
        md = HERE / f"response_reviewer_{n}.md"
        md.write_text("\n".join(doc) + "\n", encoding="utf-8")
        docx = OUT_DIR / (f"ZSH_response_reviewer_{n}_v2_draft.docx" if DRAFT
                          else f"ZSH_response_reviewer_{n}_v2.docx")
        subprocess.run(["pandoc", str(md), "-f", "markdown+tex_math_dollars", "-o", str(docx)], check=True)
        out.append((n, len("\n".join(doc).split())))
    print("letters: " + ", ".join(f"reviewer {n} ({w:,} words)" for n, w in out))
    return src


def main():
    notes = []
    texts = []
    for name in SECTIONS:
        p = HERE / name
        if not p.exists():
            continue
        md = p.read_text(encoding="utf-8")
        md = re.sub(r"^# Draft.*\n", "", md)
        md = strip_comments(md, notes, name)
        if name != "draft_front.md":
            md = shift_headings(md)
        texts.append((name, md.strip()))

    body = "\n\n".join(t for n, t in texts if n != "draft_appendix.md")
    appendix = "\n\n".join(t for n, t in texts if n == "draft_appendix.md")

    # number items by first mention; supplementary items get their own S series
    tnum, fnum = {}, {}
    supp_t, supp_f = [], []
    known_t, known_f = set(TAB) | set(APP_TAB), set(FIG)
    for m in re.finditer(r"\{(T_[A-Za-z_]+|F_[A-Za-z_]+)\}", body):
        k = m.group(1)
        if k in tnum or k in fnum:
            continue
        if k in SUPP:
            if k in known_t:
                supp_t.append(k)
                tnum[k] = f"S{len(supp_t)}"
            elif k in known_f:
                supp_f.append(k)
                fnum[k] = f"S{len(supp_f)}"
        elif k in TAB:
            tnum[k] = str(sum(1 for v in tnum.values() if not v.startswith("S")) + 1)
        elif k in FIG:
            fnum[k] = str(sum(1 for v in fnum.values() if not v.startswith("S")) + 1)
    for k in SUPP:                       # supplementary items never mentioned in the article
        if k in known_t and k not in tnum:
            supp_t.append(k)
            tnum[k] = f"S{len(supp_t)}"
        elif k in known_f and k not in fnum:
            supp_f.append(k)
            fnum[k] = f"S{len(supp_f)}"
    unused = [k for k in list(TAB) + list(FIG) + list(APP_TAB) if k not in tnum and k not in fnum]

    # place items after the block of first mention
    placed = set()
    out_blocks = []
    for blk in blocks_of(body):
        out_blocks.append(blk)
        for m in re.finditer(r"\{(T_[A-Za-z_]+|F_[A-Za-z_]+)\}", blk):
            k = m.group(1)
            if k in placed or k in SUPP or (k not in TAB and k not in FIG):
                continue
            placed.add(k)
            if k in TAB:
                out_blocks.append(table_block(tnum[k], k, *TAB[k]))
            else:
                out_blocks.append(figure_block(fnum[k], k, *FIG[k]))
    body = "\n\n".join(out_blocks)

    # list of supplementary items for the back matter
    def first_sentence(c):
        """First sentence of a caption, never cut mid-word."""
        c = re.sub(r"\s+", " ", c).strip()
        m = re.match(r"(.{0,260}?[.])(\s|$)", c)
        if m:
            return m.group(1).rstrip(".")
        cut = c[:260]
        if " " in cut:
            cut = cut[:cut.rindex(" ")]
        return cut.rstrip(" ,;:-") + "…"
    items = [f"Table {tnum[k]}: " + first_sentence((TAB.get(k) or APP_TAB[k])[1]) for k in supp_t]
    items += [f"Figure {fnum[k]}: " + first_sentence(FIG[k][2]) for k in supp_f]
    body = body.replace("{SUPP_LIST}", "; ".join(items))
    body += "\n\n# References\n\n<!-- REFERENCES -->\n"

    def label(m):
        k = m.group(1)
        return tnum.get(k) or fnum.get(k) or m.group(0)
    body = re.sub(r"\{(T_[A-Za-z_]+|F_[A-Za-z_]+)\}", label, body)
    # captions may reference other tables
    body = re.sub(r"\{(T_[A-Za-z_]+)\}", label, body)

    # leftover placeholders outside math
    nomath = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$|\{#eq:[^}]+\}|\{width=[^}]+\}|::: \{[^}]+\}|\\eqref\{[^}]+\}", "",
                    body, flags=re.S)
    left = sorted(set(re.findall(r"\{[^{}]{1,300}\}", nomath, flags=re.S)))
    out = HERE / "manuscript.md"
    out.write_text(body, encoding="utf-8")
    (HERE / "author_notes.md").write_text("# Items for the authors\n\n" + "\n".join(notes) + "\n", encoding="utf-8")
    print(f"tables: {tnum}\nfigures: {fnum}")
    if unused:
        print(f"never referenced: {unused}")
    AUTHOR_PLACEHOLDERS = {"{ZENODO_DOI}"}   # deliberately left for the authors
    blocking = [x for x in left if x not in AUTHOR_PLACEHOLDERS]
    if left:
        print(f"{len(left)} placeholders left: {left}")
    if blocking and not DRAFT:
        sys.exit(1)
    # response letters: one per reviewer, with the article's table and figure numbers
    letter = build_letters(label)
    left_l = sorted(set(re.findall(r"\{[^{}]{1,80}\}", re.sub(r"\$[^\$]+\$", "", letter))))
    if left_l:
        print(f"letter placeholders left: {left_l}")

    docx = OUT_DIR / ("ZSH_FinTech_MDPI_manuscript_v2_draft.docx" if DRAFT else "ZSH_FinTech_MDPI_manuscript_v2.docx")
    subprocess.run([sys.executable, str(HERE / "build_manuscript.py"), str(out), str(HERE / "references.json"),
                    str(docx)], check=True)
    write_supplementary(tnum, fnum, supp_t, supp_f, notes, label)


if __name__ == "__main__":
    main()
