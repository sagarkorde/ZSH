# Response to Reviewer 1 — third revision

**Manuscript:** ZSH: Rank-Weighted Profiling of Bitcoin Transactions and a Pre-Specified Evaluation of What the Profiles Capture

We thank the reviewer for checking the repository tags and opening the archive rather than taking our word for either. Both of the errors found that way would otherwise have reached print.

All three problems are accepted without qualification. Two of them were errors on our part, not differences of interpretation, and one of those was also stated in our previous reply to this reviewer. We set out below what was wrong, what the manuscript says now, and how each correction can be checked. Section numbers refer to the revised manuscript; the changes are highlighted in the marked copy.

No result changed. These are corrections to claims about the evidence, not to the evidence itself; every reported number is unaltered, and all 331 numerical claims in the article and its supplementary file remain checked against the saved result files by `experiments/verify_manuscript_numbers.py`.

---

**Comment 1 — The account of prospective validation is chronologically inconsistent. The v2-plan and v2-frozen tags are dated 16 September 2026, whereas the sample covers October 2024 to August 2026, so the statements in Sections 5.1 and 7.8 that these blocks had not been mined at the freeze cannot be correct.**

The reviewer is right, and the dates are as stated: the two tags are timestamped 16 September 2026 (17 September in the authors' local time), while the sampled blocks run from height 863,566 in October 2024 to 964,957 in August 2026. Every one of them was mined before the freeze. The sentences in Sections 5.1 and 7.8 were false, and the same claim appeared in our previous reply to this reviewer, where we withdraw it as well.

What is true is narrower, and is all we now assert: the sample was collected after the freeze, the block heights and page offsets were drawn from seeds fixed in the analysis plan itself, and none of the sample had been collected or examined when the plan was written. That makes it held-out data and a test of temporal generalisation — a model fitted on 2022–2023 applied one to two years later — but not evidence about observations that did not exist at the freeze.

Four changes follow:

* Section 3.2 defines the term where the sample is first introduced, stating that its blocks were mined between October 2024 and August 2026 and therefore existed before the freeze of 16 September 2026, and that we use the word "prospective" only for the period the sample covers.
* Section 5.1 renames the evidence category from "prospective evidence" to "held-out temporal evidence" and carries the same statement of what the freeze does and does not establish.
* Section 7.8 no longer claims genuinely prospective status; it says the sample tests whether a model fitted on 2022–2023 still describes later activity, "not anything that was unobservable at the freeze".
* Table S2 relabels the affected rows from "prospective on 2024-26" to "held out on 2024-26".

We take the reviewer's point that this matters more because of the prior exposure acknowledged elsewhere. Section 7.8 now states both limitations together: the design was informed by an earlier version of this work on the same corpus, and the one body of evidence we had not seen when the plan was written is held out in collection rather than in time.

---

**Comment 2 — The Data Availability Statement directs readers to Zenodo record 22933176, which contains the manuscript PDF only, while the reply letter gives the dataset DOI as the unresolved placeholder {ZENODO_DOI}.**

Accepted, and this was the more serious of the two errors: the statement asserted an archive that did not exist. Record 22933176 is of type *journal article* and holds the manuscript; it should never have been cited as the data archive, and the placeholder should never have been submitted unresolved.

The data are now deposited as a record of type **dataset** at <https://doi.org/10.5281/zenodo.23037947>, published under CC BY 4.0 with open access to all files. It contains 32 files totalling 549 MB:

* `d4_future.parquet` — the 456,292-transaction sample with its design weights (55.4 MB);
* `d4_raw_api_responses_partNNof25.jsonl.gz` — all 24,526 raw Esplora responses exactly as received, in 25 parts of 1,000 responses; each part is a valid gzipped JSONL file on its own;
* `d5_sample_d1rows.parquet` and `d5_api_records.parquet` — the 2,000-transaction API-equivalence check;
* `future_manifest.json` — the per-month collection record, with heights, blocks sampled and pages obtained;
* `README.md` — file list, the two-stage sampling design, the weighting rule and terms of use;
* `SHA256SUMS` — checksums of every file in the record.

The Data Availability Statement now names the deposited checksum of the sample file, `dff579fb…8733`, which is the value the collector recorded when the sample was closed in September 2026 and the value listed in both `future_manifest.json` and `SHA256SUMS`. A reader can therefore download the file, hash it, and confirm that the archived sample is the one the reported weighted estimates were computed from. The manuscript no longer cites any Zenodo record of itself.

To prevent a recurrence, the build now refuses to produce a final manuscript while any placeholder remains; the previous exemption that allowed `{ZENODO_DOI}` through is removed, with the reason recorded in the source.

---

**Comment 3 — The "oracle" weighting experiment is described as an upper bound, but Table 11 contradicts this: for P2PKH inputs the annotation-specific oracle gives an AP lift of 13.4 while weighting derived from another annotation gives 22.7.**

Accepted. The experiment compares particular weighting rules and does not bound the family they are drawn from, and the reviewer's counter-example is taken from our own results: on P2PKH inputs the rule derived from P2WSH reaches 22.7 and the rule derived from P2SH 22.0, against 13.4 for the rule built from P2PKH itself and 20.4 for the unsupervised ranking. A weighting informed by the answer can be worse than one informed by a different annotation, so no ordering of these rules establishes a maximum.

Section 6.6 now states this directly rather than in passing. The paragraph says that the experiment is not an upper bound and that we no longer describe it as one, reports the cross-annotation comparison above as a finding in its own right — the effect of reweighting is erratic — and retains only the claim the table does support: that across the four annotations no rule, the oracle included, lifts attainment past a quarter of the ceiling for the three rarer ones. The caption of Table 11 and the description in Section 5.3 are relabelled from "an upper bound for the weighting" to "an oracle-informed weighting".

The conclusion that weighting is not where the headroom lies is retained, but it no longer rests on this experiment. It rests on two others: six further clustering families reach the same low shares of the ceiling for the least frequent annotations (Section 6.7), and a supervised split of the same twelve features attains a median of 96.4% of the ceiling against 16.5% for the profiles, with 95.0% against 23.6% on P2SH inputs using a model fitted a year earlier that saw no test-period label (Section 6.8). Those results bound what the features permit; the oracle experiment never did.

---

**Comment 4 — Table 9 remains difficult to read because headings and values break across lines.**

Corrected. Table 9 is reduced from twelve columns to ten. The two removed columns were redundant: "Maximum lift" is 1/π and is read directly from the base rate, which the table still gives, and the count of top-ranked profiles reaching 25% coverage is reported in the text where it is discussed. The caption records both removals and why.

---

We are aware that this is the third round and that two of these problems were of our own making. The reviewer's practice of checking the archive and the timestamps has improved the paper each time, and we are grateful for it.
