# Response to Reviewer 1 — fourth revision

**Manuscript:** ZSH: Rank-Weighted Profiling of Bitcoin Transactions and a Pre-Specified Evaluation of What the Profiles Capture

We accept all five comments. The reviewer's summary is accurate: the previous round corrected the claims in the main text and left them standing elsewhere, which is a worse state than before, because a reader of the supplement alone would have found a claim the article had withdrawn. We have now treated the submission as one package rather than as a manuscript with attachments.

Three of the five comments concern corrections we announced in the last reply and did not carry through. One (Comment 3) asks us to restrain conclusions that the evidence does not reach. One (Comment 5) asks us to verify the archive rather than assert it; we did, and it found a defect in the deposit that we would not otherwise have known about.

No result changed. Every reported number is unaltered. What changed is the scope claimed for several of them, the terminology, and three counts that were wrong.

Section numbers refer to the revised manuscript; every change is highlighted in the marked copy, and the highlighting of the previous submission has been removed so that only this round's changes are marked.

---

## Comment 1 — The supplementary materials retain the withdrawn upper-bound claim, and Section 7.5 still infers that weighting is close to exhausted.

Accepted on both counts.

**The supplement.** Section S1 of the Supplementary Materials carried the heading "An upper bound for the weighting" and the sentence that the oracle "bounds what any weighting of these twelve features could achieve for that annotation within this clustering scheme". That is the claim Section 6.6 withdrew. The heading now reads "An oracle-informed weighting", and the sentence has been replaced by its negation: the experiment compares particular weighting rules and does not bound the family they are drawn from, because a rule derived from one annotation can concentrate another better than that annotation's own oracle does, so no ordering of these rules establishes a maximum. The paragraph states that we described the experiment as an upper bound in earlier versions and withdraws that description, so that a reader who sees only the supplement is not left with it.

A second use of the phrase, in the paragraph on the shared partition of Section S1, read "It remains an upper bound rather than a method". The sense there was different — that partition needs labels for every annotation and so is not available in practice — but the phrase is the one under dispute, and it now reads "It is a benchmark rather than a method".

**Table S2.** The row is relabelled "Oracle-informed weighting". We also corrected the docstring of the function that generates that table, which still described the analysis as an upper bound; the label was being regenerated from a source that had not been changed, which is why it survived the last round.

**Section 7.5.** The sentence "Within this scheme feature weighting is close to exhausted" is removed. The paragraph now states what was tested and what follows: for the three rarer of the four annotations tested, no rule we tried — the two oracle variants, the unsupervised ranking and the rules derived from the other annotations — lifted attainment past a quarter of the ceiling, and the effect of reweighting was erratic rather than ordered. It then states what does not follow: that this does not exclude that some other weighting of these twelve features would do better, and that nothing in these experiments bounds what weighting in general could achieve. The same paragraph previously said the profiles lack "not better weights"; it now says "not better weights among those we tried".

**To prevent recurrence.** The verification script that checks the numerical claims now also fails if the strings "upper bound for the weighting" or "Oracle-weight upper bound" occur anywhere in the manuscript *or* the supplementary file. The two files are concatenated before the check, so a claim withdrawn in one and kept in the other now fails the build rather than reaching a reviewer.

---

## Comment 2 — The terminology for temporal validation remains inconsistent, and the Data Availability Statement should be reconciled with the frozen-reanalysis distinction.

Accepted. The word was used in 82 places: 66 in the section drafts, 13 in the script that generates the manuscript tables, and 3 in the script that generates the figures. All 82 are replaced. The sample is called the **held-out temporal sample** throughout the article, its tables, its figures and the supplementary file, and the word "prospective" now appears in exactly two places: the sentence in Section 3.2 and the sentence in Section 5.1 that withdraw it.

Three details are worth reporting, because they explain why the previous round missed them.

First, two of the stale labels were not in prose. Tables 7 and S10 print a period label read straight from a frozen result file, which still contains the old string. Rewriting frozen result files is not acceptable, so the table builder now maps the label on display and the result files are untouched.

Second, and more seriously, **the caption of Table S2 still asserted that the 2024–2026 sample "did not exist at the freeze"** — the precise claim withdrawn in the previous round, in the one place neither of us checked. It is removed. The caption now explains the status category as evidence collected after the freeze from seeds fixed in the plan, whose blocks were nevertheless mined before it.

Third, Section 3.2 now says explicitly why the word was dropped: that it invites the stronger reading that the transactions did not exist when the plan was written, which is not the case. We prefer to record the withdrawal rather than to make the substitution silently.

**The Data Availability Statement** now carries the distinction rather than leaving it to Section 5.1. It states that the 2022–2024 Bitcoin corpus and the Elliptic data set had both been examined in an earlier version of this work, so every analysis of them is a frozen reanalysis and not a blind confirmatory test, and that the 2024–2026 sample deposited at Zenodo was collected after the freeze and was unexamined when the plan was written, which makes it held out in collection rather than in time. The README of the deposit itself has been rewritten in the same terms, and the record's title and description no longer use the withdrawn word.

The verification script now fails on any occurrence of the word outside the two sentences that withdraw it.

---

## Comment 3 — The supervised benchmark requires a more restrained interpretation, and the seven-family results do not generalise beyond the methods, settings and data tested.

Accepted. We had stated the leakage caveat but had gone on quoting the within-period figure as the headline, which is the inconsistency the reviewer identifies.

**Separating the two benchmarks.** Section 6.8 now opens by saying that two versions are reported and which one carries the conclusion: the benchmark fitted on the development period a year before the evaluation data, which uses no test-period label, is the one the section's conclusion rests on; the within-period benchmark is an exploratory comparison and is optimistic. The headline median is changed accordingly, in Section 6.8, Section 7.5 and Section 8, from 96.4% (within period) to **89.8%** (development-trained), against 16.5% for the profiles. The within-period figure is retained — removing it would hide a result — but it is labelled optimistic and exploratory at each of the three places it is quoted, including the pricing of the shared partition (92.8% against 96.4%), where both figures are within-period and the comparison now says so.

On the reviewer's point about separating the stages: a design that separates model fitting, cell construction, ranking and final evaluation completely would need a third fold, which we did not run. Section 6.8 now says that, rather than implying that the two-fold design achieves the separation. The development-trained column separates the stages in time instead, which is why the principal claim rests on it.

We also removed a contradiction we found while making this change. Section 6.8 said "Both columns say the same thing" immediately before the paragraph showing two annotations where they emphatically do not (other OP_RETURN use falls from 97.0% to 4.8%, Omni from 52.7% to 2.4%). The sentence now says that the two columns agree for the script classes and coinbase transactions and do not for the two annotations named next.

Because the development-trained median is now a headline number, it is a **checked** number: the verification script recomputes it from the saved result file, which it previously did only for the within-period median.

**The seven families.** The two over-general sentences are replaced. "These annotations are too rare for 31 clusters of these features to isolate, whatever builds the clusters" now reads that for every one of the seven families tried here, 31 clusters of these twelve features do not isolate annotations this rare, followed by the limitation: seven families at one K on one feature set cannot show that no clustering could; what they show is that the failure is not peculiar to ZSH. The closing paragraph of Section 6.7 no longer says that three results "belong to the task and not to the method"; it says they are shared by all seven families tested here, names the scope — seven families, K = 31, these twelve features (five for the Vlahavas reproduction), this corpus and these annotations — and states that this does not establish that they are unavoidable properties of transaction profiling in general, and that a different feature set, a different K, or an objective other than variance minimisation is untested here. The corresponding sentences in the introduction (contribution 5) and the conclusion are restricted in the same way.

---

## Comment 4 — The analysis inventory and cross-references require correction.

All three items confirmed and corrected, and the consistency check the reviewer asks for was run over the whole package.

**The counts.** Table S2 contains 25 analyses, of which 15 are marked exploratory. Sections 5.1 and 7.8 said ten and fourteen of twenty-four. They now read ten specified in the plan and fifteen added afterwards, of twenty-five. The verification script now reads Table S2, counts its rows and its exploratory rows, and fails if the words used in the text do not match; a future addition to the table cannot leave the prose behind.

**The hypothesis label.** H5 in Section 5.1 concerns the ranking of Elliptic clusters by illicit rate. The claim tested for the concentration of non-input annotations is the RQ3 claim, which carries no H number. The Table S2 row is relabelled "Concentration of non-input annotations (RQ3)", and the row "Elliptic with a temporal split" now carries "(H5)", which it should have carried from the start.

**The Table 9 caption.** The clause "reached with the number of top-ranked profiles given in the next column" is removed. The caption now names both columns that were dropped and says where each quantity can be found: the maximum lift is 1/π and is read from the base rate, which the table still gives, and the number of top-ranked profiles reaching 25% coverage is reported in the text where it is discussed.

**The wider consistency check.** We wrote and ran an audit over the assembled manuscript and supplementary file together. It confirms that all 29 tables and 13 figures are both defined and cited, that every "Section N.M" reference resolves to an existing heading, that every numeric value in the abstract occurs in the body, and that none of the withdrawn phrases survives anywhere. It found one further error of exactly the kind the reviewer describes: the back matter stated that "the 300 numerical claims" in the article and supplement are checked by the repository script, where the script in fact checks 332. The back matter now gives the correct figure, and the script fails if that sentence and its own count ever disagree again.

---

## Comment 5 — The corrected data archive requires verification.

The reviewer is right that the response letter does not establish the correspondence. We therefore verified it rather than restating it, and the verification found a defect.

**What was checked.** Record 10.5281/zenodo.23037947 is of type *dataset*, licensed CC BY 4.0, open access, 32 files, 548.8 MB. We retrieved its file manifest from the Zenodo API and compared every entry with the local archive from which the deposit was made: **all 32 files match on size and on MD5**, so every deposited file is byte-identical to the local copy. We then confirmed the substantive contents by reading the files: 456,292 rows in the sample file; 24,526 raw responses counted across the 25 parts; 23 monthly strata from 2024-10 to 2026-08; 18,400 blocks with a page. The SHA-256 of the deposited sample file is `dff579fba854e86328297a5d740d736f69b7e1b9c30de8358396e617a3288733`, which equals the value `d4_sha256` written into `future_manifest.json` by the collector when the sample was closed in September 2026. That chain — manifest value, deposited file, local file — is what establishes that the archived sample is the one the reported design-weighted estimates were computed from.

**The defect.** Verifying the deposited `SHA256SUMS` line by line, 30 of its 31 entries matched the deposited files and one did not: the entry for `README.md` had been written from an earlier draft of that file, which was then edited before upload. No data file was affected. A reader following the reviewer's instruction would have met a failing checksum, and we are grateful that the instruction was given. `SHA256SUMS` has been regenerated and re-verified end to end against every file it lists.

**A second finding, which affects the manuscript rather than the deposit.** The copy of the sample at the path the analysis scripts read hashes to a different value from the deposited copy. We compared the two as data rather than as bytes: 456,292 rows, 42 columns, identical dtypes, and the two data frames are equal. The difference is the Parquet writer: the two files were written by different Arrow builds, which produce different bytes for identical data. No reported number is affected. We report it because the Data Availability Statement invites a reader to hash the file, and a reader who *regenerates* the sample rather than downloading it would get a different checksum and could reasonably conclude that something was wrong. The statement, and the README of the deposit, now say that a Parquet file written by a different Arrow build holds identical data in different bytes, and that a regenerated file should be compared on content.

**What is deposited now.** A new version of the record carries the corrected `SHA256SUMS`, a README rewritten in the terminology of Comment 2 with explicit verification instructions, and metadata from which the withdrawn word has been removed. Every data file in the new version is byte-identical to the previous one; nothing in the data changed. The Data Availability Statement now cites the **concept DOI, 10.5281/zenodo.23037946**, which always resolves to the current version, rather than a version DOI that a later correction would leave behind. The version in which the reported analyses were run remains citable and unchanged.

---

## Summary of the safeguards added

Four of the five comments in this round, and two of the three in the last, were failures of internal consistency rather than of analysis: a claim withdrawn in one file and kept in another, a count that no longer matched its table, a caption naming a deleted column, a checksum list written before its files. The verification script now checks for all four classes, over the manuscript and the supplementary file together:

* withdrawn phrasings — the oracle upper bound, the prospective sample, the chronology of the freeze, the deleted column, "close to exhausted" — fail the build wherever they occur;
* the analysis counts stated in words are read from Table S2 and must agree with it;
* the number of checked claims stated in the back matter must equal the number the script actually checks;
* the development-trained median, now a headline number, is recomputed rather than quoted.

All 335 checks pass: the 332 numerical claims stated in the back matter, plus the three
consistency checks described above. The audit of references, captions, abstract values and withdrawn phrasings reports no remaining problems.

We are aware that this is the fourth round and that most of what the reviewer found was within our power to find first. The practice of opening the archive, checking the timestamps and reading the supplement against the article has improved this paper in every round, and we are grateful for it.
