# Deviations from the analysis plan

Every change made after tag `v2-frozen` to data, code, configuration or
analysis is recorded here.

| Date | Commit | Plan section | Change | Reason | Effect on results |
|---|---|---|---|---|---|
| 2026-09-17 | (next commit) | §6 environment | Added `tabulate==0.10.0` to the pinned environment; no other package changed (verified with `git diff env/requirements.lock`). | Needed by pandas to write Markdown tables for the article. | None: presentation only. |
| 2026-09-17 | (next commit) | §6 | Added `experiments/make_tables_figures.py` and `zsh/plotstyle.py` after the freeze. | Presentation code; reads saved results only. | None on any computed quantity. |
| 2026-09-17 | (next commit) | §2.2 collection mechanics | Block hashes and transaction counts are read from mempool.space `/api/v1/blocks/<height>` (15 headers per request) instead of `/block-height/<h>` plus `/block/<hash>` per block. The sampled heights, page offsets, page requests and parsing are unchanged. | At the observed ~1 request/s (blockstream.info rate-limited) the per-block requests would have taken ~15 h; the bulk endpoint cuts ~31,000 requests to ~6,700. Checked on block 863,600: identical hash and tx_count from both endpoints. | None on the collected data (same blocks, same pages). |
| 2026-09-17 | (next commit) | E8, §7 item 5 | `GMMU` now records a failure when none of the pre-declared `reg_covar` values (1e-4, 1e-3, 1e-2) fits, and E8 reports the Gaussian mixture as "failed to fit" instead of stopping. E8 was rerun. | On the full Elliptic training data (165 features) the diagonal mixture raised a covariance error at all three values, which ended the first E8 run before any result was written. | The Gaussian-mixture baseline is absent from the Elliptic results; nothing else changed. |
