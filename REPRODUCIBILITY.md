# Reproducibility Checklist

## Minimum Contents for Public Release

- all Python pipeline scripts
- `requirements.txt`
- a short dataset description
- one command sequence for reproducing the full pipeline
- one command sequence for regenerating the paper tables and figures
- artifact map linking claims to files
- license and citation metadata

## Recommended Additions Before Submission

- add a small redacted sample dataset if the full dataset is restricted
- include a checksum for `Dataset.parquet`
- include a commit hash in the paper before final submission
- tag the release used for submission, for example `v1.0-paper-submission`
- add a release note documenting the exact manuscript version supported by that tag

## Suggested Manuscript Statement

`Code and artifact package available at: https://github.com/your-account/zsh-blockchain-profiling.`

Replace the placeholder with the real public URL before submission.
