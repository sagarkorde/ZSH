# Response to Reviewer Comments

**Manuscript:** ZSH: Rank-Weighted Profiling of Bitcoin Transactions and a Pre-Specified Evaluation of What the Profiles Capture

**Authors:** Sagar D. Korde and Narendra M. Shekokar

---

## 1. Summary

Thank you for taking the time to review this manuscript again, and for your support for publication. Both corrections were warranted and both have been made. The corresponding revisions are highlighted in the re-submitted files.

The two comments proved to carry different weight, and we would rather say so than level them. The first exposed a defect in our code, not a typographical slip: the Coinbase entry of 49.9% is what the script produced, and the reason it could exceed the limit that Section 6.8 itself states is that the cells were not equally populated in the fold they were scored on. We have corrected the code, rerun the two analyses that used it, and verified — rather than assumed — that the principal benchmark results are unaffected. The second was an error of interpretation: we compared the wrong pair of columns in Table 15, and the comparison you ask for reverses the conclusion we drew from it. We have rewritten that passage and the two statements elsewhere in the article that repeated the conclusion.

As a consequence of the first comment one reported value in Table 14 changes substantially, together with four smaller entries in the same column and its median. No other reported value in the article changes. No analysis was added and no data were recollected.

Section, table and page numbers refer to the revised manuscript.

---

## 2. Questions for General Evaluation

| Reviewer's Evaluation | Response and Revisions |
|---|---|
| Does the introduction provide sufficient background and include all relevant references? | *[rating as recorded in the submission system]* — No change was requested and none was made. |
| Are all the cited references relevant to the research? | *[rating]* — No change was requested and none was made. |
| Is the research design appropriate? | *[rating]* — No change was requested and none was made. |
| Are the methods adequately described? | *[rating]* — The description of how the supervised score is cut into cells was incomplete: it did not say that the cells are formed within each block-parity fold. The caption of Table 14 now states it, and the code has been corrected to implement it. See Response 1. |
| Are the results clearly presented? | *[rating]* — The "Equal cells" column of Table 14 has been corrected. See Response 1. |
| Are the conclusions supported by the results? | *[rating]* — The conclusion drawn from Table 15 was not supported by the comparison that isolates the effect of the additional features. Section 6.8, Section 7.5 and the abstract have been corrected. See Response 2. |

---

## 3. Point-by-point response to Comments and Suggestions for Authors

### Comments 1

> Please reconcile the Coinbase value of 49.9% in the "Equal cells" column of Table 14 with the stated construction of equally populated cells. Correct either the numerical entry or the methodological description, as appropriate, and confirm that the principal benchmark results are unaffected.

### Response 1

Accepted. The caption described the right construction; the code did not implement it. The entry has been corrected to **0.9%**.

**What went wrong.** The "Equal cells" and "Benchmark" columns use a score computed by block parity: each half of the test period is scored by a gradient-boosted tree fitted on the other half. The two halves are therefore scored by two different models, and where the twelve features separate an annotation sharply, the two models place their negatives at different values. For coinbase transactions the pooled score takes four values: 1,289,752 fold-0 negatives at 5.3 × 10⁻⁸, 1,293,755 fold-1 negatives at 8.0 × 10⁻⁵, 169 rows at zero and 854 rows at one. Cutting that pooled score into 31 cells of equal size divides the pooled rows equally but not each fold. The top cell held 83,371 pooled rows, of which only 378 lay in fold 0 — and 377 of those 378 were coinbase. Because the procedure ranks the cells on one half and scores them on the other, the direction that ranks on fold 1 and evaluates on fold 0 was measuring the precision of a cell of 378 rows that was 99.7% coinbase. That direction returns an average precision of 99.735%; the other returns 0.029%; their mean is the 49.9% in the table.

A second defect compounded this without causing it. The quantile edges were de-duplicated, so wherever the score was tied the cut returned fewer than 31 cells: ten for coinbase, and fewer than 31 for seven of the eleven annotations. We mention it because we have fixed it, but it is not the explanation — cutting the pooled score into exactly 31 equal cells still yields 50.1%. The mismatch between the two folds' score scales is the cause.

**The correction.** The cells are now formed by rank within each block-parity fold, so that every cell holds a thirty-first of the half it is scored on, with ties broken at random. This is the construction the caption describes, and under it the limit stated in Section 6.8 holds exactly. Coinbase now attains 0.9%, which is precisely the arithmetic maximum that 31 equally populated cells permit at a base rate of 0.03% (31/3451 = 0.898%). The corrected reading is therefore the opposite of a failure: the supervised score separates coinbase completely, and the cutting rule alone accounts for the whole of the apparent shortfall. Because this is the clearest demonstration of what that column measures, coinbase is now the first example in the paragraph (page 32), and the caption of Table 14 states the per-fold construction.

Both scripts that used the faulty construction — the supervised benchmark of Section 6.8 and the per-annotation arm of the shared-partition comparison — were corrected and rerun in full. The changes in the "Equal cells" column of Table 14 are:

| Annotation | Previously | Now |
|---|---:|---:|
| Coinbase | 49.9 | 0.9 |
| P2PKH inputs | 96.7 | 96.8 |
| P2WSH inputs | 43.6 | 43.7 |
| Omni | 0.5 | 0.8 |
| Exchange tag | 7.4 | 7.7 |
| Median | 61.8 | 58.7 |

The remaining five entries in the column are unchanged. The two figures quoted in the text of Section 6.8 are updated accordingly, from 43.6% to 43.7% for P2WSH inputs and from 7.4% to 7.7% for exchange tags.

**The principal benchmark results are unaffected, and we checked this rather than assuming it.**

- The column on which the principal claim rests, "A year earlier", is fitted once on the development period and applied to the whole test period. It uses one model on one scale, so a cut on its score does divide each fold equally and the defect cannot arise there. Under the previous construction its coinbase entry was 0.8%, inside the limit, while the within-period arm returned 49.9%; that contrast is what locates the defect in the within-period score alone. Under the corrected construction both arms give 0.9%.
- In the two free-cell columns the top-ranked cell is small and highly pure in *both* folds — for coinbase, 378 rows at 99.7% in one direction and 476 rows at 78.2% in the other — so those values carry no fold artefact. They are measurements of the features, as intended.
- The rerun reproduced every supervised value in the other three columns exactly. (The profiles column moved by less than 0.001 percentage points, far below the precision reported: the stored profile labels had been regenerated by a later experiment between the original run and this one.)
- The medians of the three unaffected columns — 16.5% for the profiles, 96.4% for the within-period benchmark and 89.8% for the development-trained benchmark — are unchanged, as is the comparison quoted in the abstract and the conclusions: 95.0% for the development-trained benchmark against 23.6% for the profiles on P2SH inputs.
- The shared-partition comparison later in Section 6.8 was rerun for the same reason. Its per-annotation median is 74.1% before and after the correction, so the sentence comparing the shared partition with the equal-cell variant (92.8% against 74.1%) is unchanged.

**To prevent recurrence.** Until now the verification script compared the article against the stored result files, which is why it passed: the article faithfully reported what the script had computed. It now also asserts a bound that the definition implies. For every arm whose cells are equally populated it fails if the attained share exceeds 31 times the base rate, the most that 31 equal cells can carry whatever the score says. Run against the previous result files it reports three violations: the Coinbase entry you identified, the same annotation in the shared-partition analysis, and one we had not noticed — Omni at 1.6% against a limit of 1.05%. Run against the corrected files it is silent. A reported value that contradicts its own definition now fails the build rather than reaching a reviewer.

### Comments 2

> In the interpretation of Table 15, assess the effect of additional features by comparing Refit 12 with Refit 24. The relevant changes are 3.0% to 9.4% for exchange tags and 2.8% to 4.8% for other OP_RETURN transactions. Please align the accompanying discussion with these comparisons.

### Response 2

Agreed. The comparison you specify is the right one, and it reverses the conclusion we drew.

**What was wrong.** The sentence compared Refit 24 with **Frozen** — 9.4% against 9.8% for exchange tags, 4.8% against 5.9% for other OP_RETURN use — and read the near-equality as the additional features making no difference to the profiles. Frozen is the development model transferred to the held-out sample, so it differs from Refit 24 in fitting period as well as in feature set and cannot isolate the effect of the features. Refit 12 and Refit 24 are fitted on the same 456,292 transactions by the same pipeline, with the same number of clusters, and differ only in the feature set. They are the pair that isolates it.

**What that comparison says.** The additional features reach the profiles, and decisively. Against Refit 12, the paired differences in AP lift from a block bootstrap stratified by month, using the design weights of the sampling scheme and Holm adjustment, are:

| Annotation | Refit 12 | Refit 24 | Paired difference in AP lift (95% CI) |
|---|---:|---:|---|
| Exchange tag | 3.0% | 9.4% | 19.1 (15.7 to 23.6) |
| Other OP_RETURN | 2.8% | 4.8% | 1.15 (0.93 to 1.34) |
| Runes | 89.7% | 91.9% | 0.06 (0.05 to 0.07) |

For exchange tags the attained share more than trebles. All three intervals exclude zero.

**Why the earlier reading was wrong.** Two effects of similar size cancel in the Frozen/Refit 24 comparison. Refitting on the held-out period alone *costs* 20.6 AP lift for exchange tags and 1.74 for other OP_RETURN use, measured against the transferred model, because one period gives the cluster ranking less to work with and the feature weights are re-estimated on it. The twelve address-level features then return 19.1 and 1.15 of that. Refit 24 lands close to Frozen for a reason that has nothing to do with the features being useless, and we read a coincidence as a null result.

**What the conclusion now is.** The profiles do take up the extra information; what they do not do is approach what supervision extracts from it. Refit 24 reaches 9.4% of the ceiling for exchange tags where a supervised split of the same twenty-four features reaches 59.6% — about a sixth. For this annotation, therefore, the representation and the objective both bind, rather than either alone. This sharpens rather than weakens the statement in Section 7.5 that exchange tags are the documented exception to the article's general finding.

**Where the text changed.**

- **Section 6.8, page 34.** The sentence has been replaced by the Refit 12 against Refit 24 comparison with the paired differences above, and a new paragraph sets out the two offsetting effects and states plainly that an earlier version of the section misread their cancellation.
- **Section 7.5, page 37.** "Refitting the profiles on all twenty-four features moves them barely at all, so the extra information does not reach them through the clustering objective either" now reads that the profiles do take up the extra information, trebling their attainment for exchange tags from 3.0% to 9.4%, and still end at a sixth of what supervision extracts, so the representation and the objective both bind.
- **Abstract, page 1.** "for exchange tags richer features raise what supervision extracts, not what the profiles recover" now reads "for exchange tags richer features raise what both supervision and the profiles extract, the profiles from 3.0% to 9.4% of the ceiling and supervision to 59.6%".

**The values in Table 15 are unchanged.** We verified every entry against the stored result file. That analysis uses cells of unconstrained size throughout and is therefore untouched by the defect in Comment 1.

---

## 4. Response to Comments on the Quality of English Language

The report raised no point on the quality of the English. The passages rewritten for the two comments above were reread for clarity and concision; no other wording was changed.

---

## 5. Additional clarifications

**The repository.** The two corrected scripts and the regenerated result files are in the public repository, together with the added bound assertion described in Response 1. The correction is visible in the commit history rather than only in the output.

**The data deposit is unaffected.** The Zenodo record (Version 4.0.0, <https://doi.org/10.5281/zenodo.23100775>) contains the held-out sample, the raw API responses it was built from, the collection manifest and the checksum list. It holds no result tables, and every data file in it is byte-identical to the previous version, so the corrections above require no new version of the deposit and the Data Availability Statement is unchanged.

**What we found by checking the rest.** Prompted by the first comment we swept every numerical value in all of the stored result files — 1,326 concentration estimates — against the limits their own definitions imply. Three failed, all from the single defect described in Response 1; one of the three, Omni in the shared-partition analysis, we had not noticed. Two further flags proved to be benign: in a sensitivity variant the average precision is exactly 1.000000 and the apparent 0.007% excess over the ceiling comes only from using the evaluation fold's base rate in the denominator of the lift and the pooled base rate in the ceiling. We report the sweep because it is the reason we can state that nothing else in the article is affected.

**Extent of the revision.** One paragraph has been added to Section 6.8. Everything else in this round is a corrected value, a corrected sentence or an extended caption. We are grateful for both comments; the second in particular has left the article with a more interesting result than the one we had.
