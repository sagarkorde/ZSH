# Reply to the Academic Editor — fourth revision

Dear Editor,

We submit the fourth revision of manuscript fintech-23826, together with the revised
supplementary file and a point-by-point reply to Reviewer 1.

Reviewer 1 returned five points and we accept all five. Four of them identify the same
failure: the previous round corrected a claim in the main text and left it standing in the
supplementary file, in a table caption, or in a count. The fifth asks us to verify the
data deposit rather than assert its contents, which we did; the verification found one
defective checksum entry in the deposit, now corrected in a new version of the record.

No result changed in this revision. Every reported number is unaltered. What changed is
the scope claimed for several of them, the terminology used for the 2024–2026 sample, and
four counts that were wrong — three identified by the reviewer and one by our own sweep.

To make this round the last of its kind, the verification script in the repository now
checks the whole submission rather than the article alone. It fails the build if a
withdrawn phrase appears in either file, if the analysis counts stated in the text
disagree with the table they describe, or if the number of checked claims stated in the
back matter differs from the number the script actually checks. The reasons for each
check are recorded in the source beside it.

Changes are highlighted in the revised manuscript. The highlighting of the previous
submission has been removed, so the marked passages are the changes of this round only.

We are grateful to Reviewer 1 for four rounds of close reading, and to you for the time
the journal has given this submission.

Yours sincerely,

Sagar D. Korde, on behalf of both authors

---

# Note for Reviewers 2 and 3 — fourth revision

Reviewers 2 and 3 raised no new points in this round, and we thank them for their earlier
comments, which the second and third revisions addressed.

For completeness, the changes made in this revision are these. The oracle-informed weighting
experiment is no longer described as an upper bound anywhere in the submission, including
the supplementary file and Table S2, and Section 7.5 no longer infers from it that feature
weighting is close to exhausted. The 2024–2026 sample is called the held-out temporal
sample throughout, in place of "prospective", which was used in 82 places including table
headings and figure legends. The supervised benchmark of Section 6.8 is now reported in
two clearly separated versions, and the principal claim rests on the one fitted a year
before the evaluation data, whose median is 89.8% against 16.5% for the profiles; the
within-period figure of 96.4% is retained and labelled optimistic and exploratory. The
conclusions drawn from the seven-family comparison are restricted to the methods,
settings and data tested. Four counts were corrected, and the data deposit was verified
file by file against the local archive.

No reported number changed. The 333 numerical claims in the article and its supplementary
file remain checked against the saved result files by a script in the repository, which runs
336 checks in all.
