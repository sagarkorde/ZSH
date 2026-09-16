# Deviations from the analysis plan

Every change made after tag `v2-frozen` to data, code, configuration or
analysis is recorded here.

| Date | Commit | Plan section | Change | Reason | Effect on results |
|---|---|---|---|---|---|
| 2026-09-17 | (next commit) | §6 environment | Added `tabulate==0.10.0` to the pinned environment; no other package changed (verified with `git diff env/requirements.lock`). | Needed by pandas to write Markdown tables for the article. | None: presentation only. |
| 2026-09-17 | (next commit) | §6 | Added `experiments/make_tables_figures.py` and `zsh/plotstyle.py` after the freeze. | Presentation code; reads saved results only. | None on any computed quantity. |
