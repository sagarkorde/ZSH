"""Final consistency pass over the assembled package, including every item the
external review raised. Reads manuscript.md / supplementary.md and, where the point
is about rendering, the built .docx files.
"""
import re
import pathlib
import zipfile
import collections

MS = pathlib.Path("C:/Users/sagar/Desktop/ZHS_23826/ZSH_repo_clean/manuscript")
SUB = pathlib.Path("C:/Users/sagar/Desktop/ZHS_23826/Fintech MDPI/Fintech MDPI Revised")
man = (MS / "manuscript.md").read_text(encoding="utf-8")
sup = (MS / "supplementary.md").read_text(encoding="utf-8")
both = man + "\n" + sup
problems = []


def docx(name):
    with zipfile.ZipFile(SUB / name) as z:
        return z.read("word/document.xml").decode("utf-8")


# ---------- source-level checks ----------
heads = set()
for m in re.finditer(r"^#{1,4}\s+(\d+(?:\.\d+)?)\.", man, flags=re.M):
    heads.add(m.group(1))
    heads.add(m.group(1).split(".")[0])
for r, n in collections.Counter(re.findall(r"Sections?\s+(\d+(?:\.\d+)?)", both)).items():
    if r not in heads:
        problems.append(f"section reference {r} has no heading ({n}x)")

dt = set(re.findall(r"\*\*Table (S?\d+)\.\*\*", both))
df = set(re.findall(r"\*\*Figure (S?\d+)\.\*\*", both))
ct = set(re.findall(r"Tables? (S?\d+)", both)) | {
    x for p in re.findall(r"Tables (S?\d+) and (S?\d+)", both) for x in p}
cf = set(re.findall(r"Figures? (S?\d+)", both))
for t in sorted(ct - dt):
    problems.append(f"Table {t} cited but not defined")
for f in sorted(cf - df):
    problems.append(f"Figure {f} cited but not defined")
for t in sorted(dt - ct):
    problems.append(f"Table {t} defined but never cited")
for f in sorted(df - cf):
    problems.append(f"Figure {f} defined but never cited")

for pat, why in [
    (r"\bprosp(?!ective anywhere,|ective\.==)", "withdrawn term or its abbreviation"),
    (r"upper bound for the weighting|Oracle-weight upper bound", "withdrawn claim"),
    (r"given in the next column", "deleted column"),
    (r"close to exhausted", "over-general claim"),
    (r"did not exist at the freeze|had not been mined", "withdrawn chronology"),
    (r"properties of the task|belong to the task and not|whatever builds the clusters",
     "over-general scope claim"),
    (r"Both columns say the same thing", "self-contradiction"),
    (r"\{(?:T_|F_|SUPP|ZENODO|DOI)[A-Za-z_]*\}", "unresolved placeholder"),
    (r"zenodo\.22933176|zenodo\.22945019|zenodo\.23037947", "superseded archive DOI in the paper"),
    (r"[\u4e00-\u9fff]", "CJK character"),
]:
    for m in re.finditer(pat, both, flags=re.I):
        problems.append(f"{why}: ...{both[max(0, m.start() - 50):m.end() + 50]!r}...")

for name, txt in (("manuscript", man), ("supplementary", sup)):
    if txt.count("==") % 2:
        problems.append(f"{name}: odd number of == marks")

# ---------- rendering checks, in the built .docx ----------
for fname, label in (("ZSH_FinTech_MDPI_manuscript_V5.docx", "manuscript"),
                     ("ZSH_supplementary_V5.docx", "supplement")):
    xml = docx(fname)
    plain = re.sub(r"<[^>]+>", "", xml)
    pre = "S" if label == "supplement" else ""
    nums = [int(x) for x in re.findall(rf"Table {pre}(\d+)\.", plain)]
    seen = sorted(set(nums))
    gaps = [n for n in range(1, max(seen) + 1) if n not in seen] if seen else []
    print(f"{label}: tables {pre}{seen[0]}-{pre}{seen[-1]}, gaps: {gaps or 'none'}")
    if gaps:
        problems.append(f"{label}: table number gap(s) {gaps}")
    if re.search(r"[\u4e00-\u9fff]", plain):
        problems.append(f"{label}: CJK character in the rendered document")
    for m in re.finditer(r"\bprosp", plain, flags=re.I):
        if "prospective anywhere" in plain[m.start():m.start() + 40] or \
           "as prospective" in plain[max(0, m.start() - 20):m.end() + 20]:
            continue
        problems.append(f"{label}: 'prosp' in the rendered document: "
                        f"...{re.sub(r'  +', ' ', plain[max(0, m.start() - 55):m.end() + 45])}...")
    # one caption per number; a caption may run to several sentences, so count the
    # distinct numbers that carry a caption rather than counting sentence starts
    captioned = {int(n) for n in re.findall(rf"Table {pre}(\d+)\.\s*[A-Z]", plain)}
    if captioned != set(seen):
        problems.append(f"{label}: numbered {sorted(set(seen))} but captioned {sorted(captioned)}")

mxml = re.sub(r"<[^>]+>", "", docx("ZSH_FinTech_MDPI_manuscript_V5.docx"))
for k in ("Data Availability Statement", "zenodo.23037946", "Funding:",
          "Institutional Review Board Statement", "Informed Consent Statement",
          "Conflicts of Interest", "Use of Generative AI"):
    if k not in mxml:
        problems.append(f"back matter: '{k}' missing from the rendered manuscript")

print()
if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
else:
    print("no problems found")
