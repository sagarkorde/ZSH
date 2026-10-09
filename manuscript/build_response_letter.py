"""Fill MDPI's 'Example for author to respond reviewer' template for round 5.

    python build_response_letter.py <mdpi_template.docx> <out.docx>

The letter is written into the template's own document.xml rather than imitated, so it
keeps MDPI's table, headings and styles exactly. The research-article table is filled;
the review-article table and the two 'For ... article' labels are removed. Revised
manuscript wording is set in red, as the template asks.

The template is MDPI's own and is not redistributed here; it comes with the review
request. The questions and ratings in section 2 are those of the review report for this
round, which differs from the generic template in asking whether the figures and tables
are clear and well presented rather than whether the cited references are relevant.

Every figure quoted in the letter is checked against results/ before it is sent.
"""
import re
import sys
import zipfile
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
RED = '<w:rPr><w:color w:val="FF0000"/></w:rPr>'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(text, rpr=""):
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(ppr, segments):
    """segments: list of (text, red?) or a bare string."""
    body = ""
    for seg in segments:
        t, red = seg if isinstance(seg, tuple) else (seg, False)
        if t:
            body += run(t, RED if red else "")
    return f"<w:p>{ppr}{body}</w:p>"


def cell_ppr(cell_xml):
    m = re.search(r"<w:pPr>.*?</w:pPr>", cell_xml, re.S)
    return m.group(0) if m else ""


def set_cell(row_xml, idx, paragraphs):
    """Replace the paragraphs of cell `idx`, keeping its tcPr and paragraph properties."""
    cells = list(re.finditer(r"<w:tc>(.*?)</w:tc>", row_xml, re.S))
    inner = cells[idx].group(1)
    tcpr = re.search(r"<w:tcPr>.*?</w:tcPr>", inner, re.S)
    ppr = cell_ppr(inner)
    new = (tcpr.group(0) if tcpr else "") + "".join(para(ppr, p) for p in paragraphs)
    return row_xml[:cells[idx].start(1)] + new + row_xml[cells[idx].end(1):]


# ---------------------------------------------------------------- the content
TITLE = "Response to Reviewer 1 Comments"

SUMMARY = [
    ["Thank you very much for taking the time to review this manuscript again, and for "
     "supporting publication. Both corrections were warranted and both have been made. The "
     "corresponding revisions are highlighted in the re-submitted files."],
    ["The two comments proved to carry different weight, and we would rather say so than "
     "level them. The first exposed a defect in our code, not a typographical slip: the "
     "Coinbase entry of 49.9% is what the script produced, and it was able to exceed the "
     "limit that Section 6.8 itself states because the cells were not equally populated in "
     "the fold on which they were scored. We have corrected the code, rerun the two "
     "analyses that used it, and verified rather than assumed that the principal benchmark "
     "results are unaffected. The second was an error of interpretation: we compared the "
     "wrong pair of columns in Table 15, and the comparison you ask for reverses the "
     "conclusion we drew from it."],
    ["The general evaluation marks the research design, the methods and the conclusions as "
     "“must be improved”, and the results and the figures and tables as "
     "“can be improved”. We take all five as consequences of the two corrections "
     "rather than as separate requests, and Section 2 below says for each one what was "
     "wrong and where it is now addressed. If any of them was meant more broadly, we would "
     "be glad to be told and will revise further."],
    ["As a consequence of the first comment one reported value in Table 14 changes "
     "substantially, together with four smaller entries in the same column and its median. "
     "No other reported value in the article changes. No analysis was added and no data "
     "were recollected. Section, table and page numbers below refer to the revised "
     "manuscript."],
]

# (question, rating, response). The question is set where the review report asks
# something different from the generic template; None keeps the template's wording.
# Ratings are those recorded in the review report.
EVAL = {
    4: (None, "Yes",
        "Thank you. No change was requested on this point and none was made."),
    5: ("Is the research design appropriate?", "Must be improved",
        "Addressed in Response 1. The fault was in how the supervised benchmark's score is "
        "cut into cells. The cells were formed on the pooled score, which does not divide "
        "each block-parity fold equally, so the “Equal cells” arm did not measure "
        "the quantity it was designed to measure and could exceed the limit the article "
        "states for it. The cells are now formed by rank within each fold, which is the "
        "construction the article describes, and every value in the column now respects "
        "that limit. The design of the analyses on which the principal claims rest is "
        "unchanged, and we verified rather than assumed that their results are unaffected."),
    6: ("Are the methods adequately described?", "Must be improved",
        "Addressed in Response 1. The description of how the score is cut into cells was "
        "incomplete: it did not state that the cells are formed within each block-parity "
        "fold, which is what makes the stated limit hold. The caption of Table 14 now "
        "states it (page 32) and the code has been corrected to implement it. The "
        "verification script now also asserts that limit, so a reported value that "
        "contradicts its own definition fails the build rather than reaching a reviewer."),
    7: ("Are the results clearly presented?", "Can be improved",
        "Addressed in Response 1. In the “Equal cells” column of Table 14, "
        "Coinbase changes from 49.9 to 0.9, P2PKH inputs from 96.7 to 96.8, P2WSH inputs "
        "from 43.6 to 43.7, Omni from 0.5 to 0.8, Exchange tag from 7.4 to 7.7 and the "
        "Median from 61.8 to 58.7 (page 32). The two figures quoted from that column in the "
        "text of Section 6.8 are updated to match, and coinbase is now the first example "
        "there because it is the clearest case of what the column measures. No other "
        "reported value in the article changes."),
    8: ("Are the conclusions supported by the results?", "Must be improved",
        "Addressed in Response 2. The conclusion drawn from Table 15 rested on the wrong "
        "comparison: Refit 24 against Frozen, which differ in fitting period as well as in "
        "feature set. Refit 12 against Refit 24 isolates the effect of the additional "
        "features and reverses it — the profiles do take up the extra information. "
        "Section 6.8 (page 34), Section 7.5 (page 37) and the abstract (page 1) now say so, "
        "and add that the profiles nevertheless reach about a sixth of what supervision "
        "extracts from the same features, so for exchange tags the representation and the "
        "objective both bind rather than either alone."),
    9: ("Are all figures and tables clear and well-presented?", "Can be improved",
        "Addressed in Responses 1 and 2. Table 14 carried the incorrect column and a "
        "caption that did not say how the cells are formed; both are corrected and the "
        "caption now states the per-fold construction (page 32). Table 15 is unchanged: we "
        "verified every entry against the stored result file, and that analysis uses cells "
        "of unconstrained size throughout, so it is untouched by the defect in Comment 1. "
        "What was wrong there was the discussion of the table, which has been rewritten. No "
        "other table or figure in the article is altered."),
}

COMMENT_1 = ("Comments 1: Please reconcile the Coinbase value of 49.9% in the "
             "“Equal cells” column of Table 14 with the stated construction of "
             "equally populated cells. Correct either the numerical entry or the "
             "methodological description, as appropriate, and confirm that the principal "
             "benchmark results are unaffected.")

RESPONSE_1 = [
    ["Response 1: Thank you for pointing this out. We agree with this comment, and it has "
     "proved to be a defect in our code rather than a typographical error. The caption "
     "described the correct construction; the code did not implement it. The entry has been "
     "corrected to 0.9% (Table 14, page 32)."],
    ["What went wrong. The “Equal cells” and “Benchmark” columns use a "
     "score computed by block parity: each half of the test period is scored by a "
     "gradient-boosted tree fitted on the other half. The two halves are therefore scored "
     "by two different models, and where the twelve features separate an annotation "
     "sharply the two models place their negatives at different values. For coinbase "
     "transactions the pooled score takes four values: 1,289,752 fold-0 negatives at "
     "5.3 × 10⁻⁸, 1,293,755 fold-1 negatives at 8.0 × 10⁻⁵, "
     "169 rows at zero and 854 rows at one. Cutting that pooled score into 31 cells of "
     "equal size divides the pooled rows equally but not each fold: the top cell held "
     "83,371 pooled rows, of which only 378 lay in fold 0, and 377 of those 378 were "
     "coinbase. Because the procedure ranks the cells on one half and scores them on the "
     "other, the direction that ranks on fold 1 and evaluates on fold 0 was measuring the "
     "precision of a cell of 378 rows that was 99.7% coinbase. That direction returns an "
     "average precision of 99.735% and the other returns 0.029%; their mean is the 49.9% in "
     "the table. A second defect compounded this without causing it: the quantile edges "
     "were de-duplicated, so wherever the score was tied the cut returned fewer than 31 "
     "cells. Cutting the pooled score into exactly 31 equal cells still yields 50.1%, so "
     "the mismatch between the two folds' score scales is the cause."],
    ["The correction. The cells are now formed by rank within each block-parity fold, so "
     "that every cell holds a thirty-first of the half it is scored on, with ties broken at "
     "random. This is the construction the caption describes, and under it the limit stated "
     "in Section 6.8 holds exactly. Coinbase now attains 0.9%, which is precisely the "
     "arithmetic maximum that 31 equally populated cells permit at a base rate of 0.03% "
     "(31/3451 = 0.898%). The corrected reading is therefore the opposite of a failure: the "
     "supervised score separates coinbase completely, and the cutting rule alone accounts "
     "for the whole of the apparent shortfall. Because this is the clearest demonstration "
     "of what that column measures, coinbase is now the first example in the paragraph, and "
     "the caption of Table 14 states the per-fold construction. Both scripts that used the "
     "faulty construction were corrected and rerun in full. In the “Equal cells” "
     "column, Coinbase changes from 49.9 to 0.9, P2PKH inputs from 96.7 to 96.8, P2WSH "
     "inputs from 43.6 to 43.7, Omni from 0.5 to 0.8, Exchange tag from 7.4 to 7.7 and the "
     "Median from 61.8 to 58.7. The remaining five entries in the column are unchanged."],
    ["The principal benchmark results are unaffected, and we checked this rather than "
     "assuming it. The column on which the principal claim rests, “A year "
     "earlier”, is fitted once on the development period and applied to the whole test "
     "period; it uses one model on one scale, so a cut on its score does divide each fold "
     "equally and the defect cannot arise there. Under the previous construction its "
     "coinbase entry was 0.8%, inside the limit, while the within-period arm returned "
     "49.9%; that contrast is what locates the defect in the within-period score alone. In "
     "the two free-cell columns the top-ranked cell is small and highly pure in both folds "
     "— for coinbase, 378 rows at 99.7% in one direction and 476 rows at 78.2% in the "
     "other — so those values carry no fold artefact. The rerun reproduced every "
     "supervised value in the other three columns exactly. The medians of the three "
     "unaffected columns, 16.5%, 96.4% and 89.8%, are unchanged, as is the comparison "
     "quoted in the abstract and the conclusions: 95.0% for the development-trained "
     "benchmark against 23.6% for the profiles on P2SH inputs. The shared-partition "
     "comparison later in Section 6.8 was rerun for the same reason; its per-annotation "
     "median is 74.1% before and after, so the sentence comparing the shared partition with "
     "the equal-cell variant (92.8% against 74.1%) is unchanged."],
    ["To prevent recurrence. Until now the verification script compared the article against "
     "the stored result files, which is why it passed: the article faithfully reported what "
     "the script had computed. It now also asserts a bound that the definition implies. For "
     "every arm whose cells are equally populated it fails if the attained share exceeds 31 "
     "times the base rate, the most that 31 equal cells can carry whatever the score says. "
     "Run against the previous result files it reports three violations: the Coinbase entry "
     "you identified, the same annotation in the shared-partition analysis, and one we had "
     "not noticed, Omni at 1.6% against a limit of 1.05%. Run against the corrected files "
     "it is silent."],
    ["The revised caption and paragraph read:"],
    [("“Equal cells: cells of equal size, which cannot isolate a rare annotation "
      "whatever the score; they are formed by rank within each block-parity fold, so that "
      "each cell holds a thirty-first of the half it is scored on and the bound in the text "
      "applies.” (Table 14 caption, page 32)", True)],
    [("“The difference is large exactly where it should be — 0.9% against 88.9% "
      "for coinbase transactions, 43.7% against 95.9% for P2WSH inputs and 7.7% against "
      "46.6% for exchange tags — and negligible for the common classes, where it "
      "changes nothing. Coinbase shows most plainly what the equal-cell column measures. "
      "Its 0.9% is the whole of what 31 cells of equal size permit at a base rate of 0.03%: "
      "the supervised score has in fact separated the annotation completely, and the cutting "
      "rule accounts for the entire apparent shortfall. Across the ten annotations the "
      "median is 58.7% against 96.4%.” (Section 6.8, page 32)", True)],
]

COMMENT_2 = ("Comments 2: In the interpretation of Table 15, assess the effect of additional "
             "features by comparing Refit 12 with Refit 24. The relevant changes are 3.0% to "
             "9.4% for exchange tags and 2.8% to 4.8% for other OP_RETURN transactions. "
             "Please align the accompanying discussion with these comparisons.")

RESPONSE_2 = [
    ["Response 2: Agree. The comparison you specify is the right one, and it reverses the "
     "conclusion we drew. We have accordingly rewritten the passage in Section 6.8 (page "
     "34) and the two statements elsewhere in the article that repeated the conclusion, in "
     "Section 7.5 (page 37) and the abstract (page 1)."],
    ["What was wrong. The sentence compared Refit 24 with Frozen — 9.4% against 9.8% "
     "for exchange tags, 4.8% against 5.9% for other OP_RETURN use — and read the "
     "near-equality as the additional features making no difference to the profiles. Frozen "
     "is the development model transferred to the held-out sample, so it differs from Refit "
     "24 in fitting period as well as in feature set and cannot isolate the effect of the "
     "features. Refit 12 and Refit 24 are fitted on the same 456,292 transactions by the "
     "same pipeline, with the same number of clusters, and differ only in the feature set. "
     "They are the pair that isolates it."],
    ["What that comparison says. The additional features reach the profiles, and decisively. "
     "Against Refit 12, the paired differences in AP lift from a block bootstrap stratified "
     "by month, using the design weights of the sampling scheme and Holm adjustment, are "
     "19.1 (95% CI 15.7 to 23.6) for exchange tags, whose attained share goes from 3.0% to "
     "9.4% of the ceiling; 1.15 (0.93 to 1.34) for other OP_RETURN use, from 2.8% to 4.8%; "
     "and 0.06 (0.05 to 0.07) for Runes, from 89.7% to 91.9%, which is already close to its "
     "ceiling. All three intervals exclude zero. For exchange tags the attained share more "
     "than trebles."],
    ["Why the earlier reading was wrong. Two effects of similar size cancel in the "
     "Frozen/Refit 24 comparison. Refitting on the held-out period alone costs 20.6 AP lift "
     "for exchange tags and 1.74 for other OP_RETURN use, measured against the transferred "
     "model, because one period gives the cluster ranking less to work with and the feature "
     "weights are re-estimated on it. The twelve address-level features then return 19.1 and "
     "1.15 of that. Refit 24 lands close to Frozen for a reason that has nothing to do with "
     "the features being useless, and we read a coincidence as a null result."],
    ["What the conclusion now is. The profiles do take up the extra information; what they "
     "do not do is approach what supervision extracts from it. Refit 24 reaches 9.4% of the "
     "ceiling for exchange tags where a supervised split of the same twenty-four features "
     "reaches 59.6%, about a sixth. For this annotation, therefore, the representation and "
     "the objective both bind, rather than either alone. This sharpens rather than weakens "
     "the statement in Section 7.5 that exchange tags are the documented exception to the "
     "article's general finding. The values in Table 15 are unchanged; we verified every "
     "entry against the stored result file. That analysis uses cells of unconstrained size "
     "throughout and is therefore untouched by the defect in Comment 1."],
    ["The revised passages read:"],
    [("“The features reach the profiles as well, and the comparison that shows it is "
      "Refit 12 against Refit 24, which are fitted on the same transactions by the same "
      "pipeline and differ only in the feature set. Exchange tags go from 3.0% to 9.4% of "
      "the ceiling and other OP_RETURN use from 2.8% to 4.8%; as paired differences in AP "
      "lift these are 19.1 (95% CI 15.7 to 23.6) and 1.15 (0.93 to 1.34), both well clear "
      "of zero. Runes rises from 89.7% to 91.9%, already close to its ceiling.” "
      "(Section 6.8, page 34)", True)],
    [("“Refit 24 nevertheless lands where the transferred model already was — 9.4% "
      "against 9.8% for exchange tags, 4.8% against 5.9% for other OP_RETURN use — and "
      "it would be a mistake to read that agreement as the features making no difference, "
      "which an earlier version of this section did. Two effects of similar size cancel in "
      "it. […] The profiles do take up the extra information. What they do not do is "
      "approach what supervision extracts from it: 9.4% against 59.6% for exchange tags, so "
      "the objective still accounts for most of the gap, and for this annotation the "
      "representation and the objective both bind.” (Section 6.8, page 34)", True)],
    [("“The profiles take up the extra information too — refitted on all "
      "twenty-four features they treble their attainment for exchange tags, from 3.0% to "
      "9.4% — but they end at a sixth of what supervision extracts from the same "
      "features, so for this annotation the representation and the objective both bind "
      "rather than either alone.” (Section 7.5, page 37)", True)],
    [("“… for exchange tags richer features raise what both supervision and the "
      "profiles extract, the profiles from 3.0% to 9.4% of the ceiling and supervision to "
      "59.6%.” (Abstract, page 1)", True)],
]

ENGLISH_POINT = ["Point 1: The review report records “The English is fine and does not "
                 "require any improvement”, and raises no separate point on the "
                 "language."]
ENGLISH_RESPONSE = [
    ["Response 1: Thank you. No change was required on this point. The passages rewritten "
     "for Comments 1 and 2 above were reread for clarity and concision; no other wording "
     "was changed."],
]

CLARIFY = [
    ["The repository. The two corrected scripts and the regenerated result files are in the "
     "public repository, together with the bound assertion described in Response 1, so the "
     "correction is visible in the commit history and not only in the output."],
    ["The data deposit is unaffected. The Zenodo record (Version 4.0.0, "
     "https://doi.org/10.5281/zenodo.23100775) contains the held-out sample, the raw API "
     "responses it was built from, the collection manifest and the checksum list. It holds "
     "no result tables, and every data file in it is byte-identical to the previous version, "
     "so the corrections above require no new version of the deposit and the Data "
     "Availability Statement is unchanged."],
    ["What we found by checking the rest. Prompted by the first comment we swept every "
     "numerical value in all of the stored result files, 1,326 concentration estimates, "
     "against the limits their own definitions imply. Three failed, all from the single "
     "defect described in Response 1, and one of the three we had not noticed. Two further "
     "flags proved benign: in a sensitivity variant the average precision is exactly "
     "1.000000, and the apparent 0.007% excess over the ceiling comes only from using the "
     "evaluation fold's base rate in the denominator of the lift and the pooled base rate in "
     "the ceiling."],
    ["Extent of the revision. One paragraph has been added to Section 6.8. Everything else "
     "in this round is a corrected value, a corrected sentence or an extended caption. We "
     "are grateful for both comments; the second in particular has left the article with a "
     "more interesting result than the one we had."],
]


def main():
    z = zipfile.ZipFile(SRC)
    d = z.read("word/document.xml").decode("utf8")
    body = re.search(r"<w:body>(.*)</w:body>", d, re.S)
    inner = body.group(1)

    tables = list(re.finditer(r"<w:tbl>.*?</w:tbl>", inner, re.S))
    assert len(tables) == 2, f"expected 2 tables, found {len(tables)}"
    tbl = tables[0].group(0)
    rows = list(re.finditer(r"<w:tr\b.*?</w:tr>", tbl, re.S))
    assert len(rows) == 20, f"expected 20 rows, found {len(rows)}"

    new_rows = {}
    new_rows[0] = set_cell(rows[0].group(0), 0, [[TITLE]])
    new_rows[2] = set_cell(rows[2].group(0), 0, SUMMARY)
    for ri, (question, rating, resp) in EVAL.items():
        r = rows[ri].group(0)
        if question is not None:
            r = set_cell(r, 0, [[question]])
        r = set_cell(r, 1, [[rating]])
        new_rows[ri] = set_cell(r, 2, [[resp]])
    new_rows[11] = set_cell(rows[11].group(0), 0, [[COMMENT_1]])
    new_rows[12] = set_cell(rows[12].group(0), 0, RESPONSE_1)
    new_rows[13] = set_cell(rows[13].group(0), 0, [[COMMENT_2]])
    new_rows[14] = set_cell(rows[14].group(0), 0, RESPONSE_2)
    new_rows[16] = set_cell(rows[16].group(0), 0, [ENGLISH_POINT])
    new_rows[17] = set_cell(rows[17].group(0), 0, ENGLISH_RESPONSE)
    new_rows[19] = set_cell(rows[19].group(0), 0, CLARIFY)

    out, pos = [], 0
    for ri, m in enumerate(rows):
        out.append(tbl[pos:m.start()])
        out.append(new_rows.get(ri, m.group(0)))
        pos = m.end()
    out.append(tbl[pos:])
    tbl_new = "".join(out)

    # keep only the research-article table; drop both 'For ... article' labels and the
    # review-article table, but keep the trailing sectPr (page setup)
    sectpr = re.search(r"<w:sectPr\b.*?</w:sectPr>", inner, re.S)
    inner_new = tbl_new + (sectpr.group(0) if sectpr else "")
    d_new = d[:body.start(1)] + inner_new + d[body.end(1):]

    from xml.etree import ElementTree as ET
    ET.fromstring(d_new.encode("utf8"))

    with zipfile.ZipFile(SRC) as zi, zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            zo.writestr(it, d_new.encode("utf8") if it.filename == "word/document.xml"
                        else zi.read(it.filename))
    print(f"wrote {DST}")


main()
