# IEEE Access Submission Checklist

This checklist is tailored to the current ZSH manuscript and aligned with the official IEEE Access author guidance.

Official sources:

- [Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/)
- [Submission Guidelines](https://ieeeaccess.ieee.org/guide-for-authors/submission-guidelines/)
- [Reproducibility Guidance](https://ieeeaccess.ieee.org/authors/reproducibility/)

## A. Manuscript Compliance

- Use the official IEEE Access template in Word or LaTeX.
- Submit both the source file and the PDF file.
- Ensure the source file and PDF contain the same title, authors, affiliations, and content.
- Keep the manuscript in double-column, single-spaced IEEE Access format.
- Confirm all author names appear correctly in both the source file and PDF.
- Add short biographies for **all** authors below the references.
- If author photos are required by the final template workflow, prepare them in advance.

## B. Scientific Readiness

- Do not claim that `ZSH` is universally better than `KMeans`.
- Keep the main claim conditional:
  - `KMeans++ Elkan` is stronger on intrinsic Euclidean geometry.
  - `ZSH` is stronger for semantic blockchain profiling and anomaly-aware interpretability.
- Keep the contextual comparison framed as `task-alignment evidence`, not external ground truth.
- Keep the limitations section intact.
- Make sure all conclusions are directly supported by the reported data.

## C. Language and Presentation

- Run one final grammar and style pass.
- Remove any placeholder text left in title page, acknowledgments, repository statement, or biographies.
- Keep dense UMAP-style figures out of the main manuscript if they reduce readability.
- Keep the main paper focused on:
  - `Fig. 3`
  - `Fig. 9`
  - `Fig. 10`
  - `Fig. 11`
  - `Fig. 12`
- Move `fig1`, `fig2`, and `fig5` to supplementary material or repository documentation if layout becomes crowded.

## D. Reproducibility and Integrity

- Replace the GitHub placeholder URL with the real public repository link.
- Ensure the public repository includes:
  - pipeline scripts
  - `requirements.txt`
  - artifact map
  - reproducibility instructions
  - license
  - citation metadata
- If the full dataset cannot be released, clearly explain that in the repository and provide schema or sample data guidance.
- Add a code/data availability statement in the manuscript.
- Keep the AI-use disclosure in the acknowledgment if AI-assisted writing was used.

## E. Ethics and Originality

- Confirm the manuscript is not under review elsewhere.
- Confirm all references are properly cited.
- Confirm the manuscript is original and not duplicate publication.
- If any part overlaps with a thesis, dissertation, preprint, or conference version, disclose it in:
  - the manuscript
  - the cover letter
  - the submission system

## F. Final Build Checks

- Generate the final PDF from the revised source.
- Manually inspect every page for:
  - overlapping text
  - cut-off equations
  - unreadable captions
  - squeezed tables
  - missing figures
  - broken references
- Verify that wide figures are placed as full-width where needed.
- Verify that all tables are legible in two-column format.
- Confirm the final PDF page count is reasonable for IEEE Access.

## G. Current Status for This Project

- Manuscript framing: improved
- Reproducibility package: prepared locally
- GitHub public link: still pending
- Final compile verification: still pending
- Author details and biographies: still pending
- Submission readiness: **not yet final, but close after these last steps**
