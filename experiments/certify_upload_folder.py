"""Certify the SUBMIT_V5 folder: check the files that will be uploaded, not their sources."""
import hashlib
import pathlib
import re
import zipfile

D = pathlib.Path("C:/Users/sagar/Desktop/ZHS_23826/Fintech MDPI/Fintech MDPI Revised/SUBMIT_V5")
SRC = pathlib.Path("C:/Users/sagar/Desktop/ZHS_23826/ZSH_repo_clean/manuscript")
problems = []


def text(p):
    with zipfile.ZipFile(p) as z:
        return re.sub(r"<[^>]+>", "", z.read("word/document.xml").decode("utf-8"))


def highlights(p):
    with zipfile.ZipFile(p) as z:
        return len(re.findall(r'w:highlight w:val="yellow"', z.read("word/document.xml").decode("utf-8")))


files = sorted(D.glob("*.docx"))
expected = {"ZSH_FinTech_MDPI_manuscript_V5.docx", "ZSH_supplementary_V5.docx",
            "ZSH_response_reviewer_1_v5.docx", "ZSH_cover_letter_v5.docx",
            "ZSH_editor_reply_v5.docx"}
got = {p.name for p in files}
if got != expected:
    problems.append(f"folder contents: missing {sorted(expected - got)}, extra {sorted(got - expected)}")
for p in D.iterdir():
    if p.suffix.lower() != ".docx":
        problems.append(f"non-.docx file in the upload folder: {p.name}")
    if re.search(r"_V4|_v4|_v2\b|_draft", p.stem):
        problems.append(f"stale build in the upload folder: {p.name}")

man = text(D / "ZSH_FinTech_MDPI_manuscript_V5.docx")
sup = text(D / "ZSH_supplementary_V5.docx")
rev = text(D / "ZSH_response_reviewer_1_v5.docx")
cov = text(D / "ZSH_cover_letter_v5.docx")
edi = text(D / "ZSH_editor_reply_v5.docx")

# ---- manuscript
for k in ("Data Availability Statement", "10.5281/zenodo.23037946", "The 333 numerical claims",
          "dff579fba854e86328297a5d740d736f69b7e1b9c30de8358396e617a3288733",
          "Use of Generative AI", "Conflicts of Interest", "held-out temporal"):
    if k not in man:
        problems.append(f"manuscript: '{k}' missing")
if "zenodo.23037947" in man or "zenodo.22933176" in man:
    problems.append("manuscript: cites a superseded archive DOI")
nums = sorted({int(n) for n in re.findall(r"Table (\d+)\.", man)})
if nums != list(range(1, 16)):
    problems.append(f"manuscript: table numbers {nums}")
fnums = sorted({int(n) for n in re.findall(r"Figure (\d+)\.", man)})
if fnums != list(range(1, 8)):
    problems.append(f"manuscript: figure numbers {fnums}")

# ---- supplement
snums = sorted({int(n) for n in re.findall(r"Table S(\d+)\.", sup)})
if snums != list(range(1, 15)):
    problems.append(f"supplement: table numbers {snums}")
sfig = sorted({int(n) for n in re.findall(r"Figure S(\d+)\.", sup)})
if sfig != list(range(1, 7)):
    problems.append(f"supplement: figure numbers {sfig}")

# ---- withdrawn wording, over both
pkg = man + "\n" + sup
for pat, why in [(r"\bprosp(?!ective anywhere|ective\.)", "withdrawn term or abbreviation"),
                 (r"upper bound for the weighting|Oracle-weight upper bound", "withdrawn claim"),
                 (r"close to exhausted", "over-general claim"),
                 (r"properties of the task|belong to the task and not", "over-general scope"),
                 (r"whatever builds the clusters", "over-general scope"),
                 (r"given in the next column", "deleted column"),
                 (r"did not exist at the freeze|had not been mined", "withdrawn chronology"),
                 (r"Both columns say the same thing", "self-contradiction"),
                 (r"[\u4e00-\u9fff]", "CJK character"),
                 (r"\{[A-Z_]{3,}\}", "unresolved placeholder")]:
    for m in re.finditer(pat, pkg, re.I):
        problems.append(f"{why}: ...{re.sub(r'  +', ' ', pkg[max(0, m.start()-45):m.end()+45])}...")
n_pros = len(re.findall(r"\bprospectiv", pkg, re.I))
if n_pros != 2:
    problems.append(f"the word appears {n_pros} times, expected 2 (the withdrawal sentences)")

# ---- letters
if "104" not in rev or "Prosp." not in rev or "336 checks" not in rev:
    problems.append("reviewer letter: the corrected count or the abbreviation note is missing")
for bad in ("82 places", "86 places", "All 82", "All 86", "332 numerical", "335 checks"):
    for lab, t in (("reviewer letter", rev), ("cover letter", cov)):
        if bad in t:
            problems.append(f"{lab}: stale count '{bad}'")
for lab, t in (("reviewer letter", rev), ("cover letter", cov)):
    if "10.5281/zenodo.23100775" not in t:
        problems.append(f"{lab}: published dataset DOI missing")
if "Note for Reviewers 2 and 3" not in edi:
    problems.append("editor reply: the note for Reviewers 2 and 3 is missing")

# ---- freshness: every upload must be newer than every source
newest_src = max(p.stat().st_mtime for p in SRC.glob("draft_*.md"))
for p in files:
    if p.stat().st_mtime < newest_src:
        problems.append(f"{p.name} is older than the newest source draft")

print(f"{'file':<38}{'KB':>9}{'pages/marks':>14}  sha256")
lines = []
for p in files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    hl = highlights(p)
    print(f"{p.name:<38}{p.stat().st_size/1024:>9.0f}{('%d marks' % hl) if hl else '-':>14}  {h[:16]}…")
    lines.append(f"{h}  {p.name}")
(D.parent / "SUBMIT_V5_SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"\nhighlighted runs: manuscript {highlights(D / 'ZSH_FinTech_MDPI_manuscript_V5.docx')}, "
      f"supplement {highlights(D / 'ZSH_supplementary_V5.docx')}")
print()
if problems:
    print(f"{len(problems)} problem(s):")
    for x in problems:
        print("  -", x)
else:
    print("CERTIFIED: no problems found in the upload folder")
