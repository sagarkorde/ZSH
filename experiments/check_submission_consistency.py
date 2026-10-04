"""Consistency of the whole submission package, not just its numbers.

Four rounds of review found errors of one kind: a claim withdrawn in one file and kept in
another, a count that no longer matched its table, a caption naming a deleted column, a
checksum list written before its files. None was a numerical claim, so none was caught by
verify_manuscript_numbers.py. This script checks the rest:

* every "Section N.M" reference resolves to a heading;
* every table and figure is both defined and cited, and numbered without gaps;
* every value in the abstract reappears in the body;
* withdrawn phrasings appear nowhere in either file;
* the built .docx and the exported .pdf agree with the source on numbering and captions.

Run after assemble.py and after exporting the PDFs.
"""
import collections
import hashlib
import pathlib
import re
import zipfile

MS = pathlib.Path(__file__).resolve().parent.parent / "manuscript"
SUB = pathlib.Path("C:/Users/sagar/Desktop/ZHS_23826/Fintech MDPI/Fintech MDPI Revised")
man = (MS / "manuscript.md").read_text(encoding="utf-8")
sup = (MS / "supplementary.md").read_text(encoding="utf-8")
both = man + "\n" + sup
problems = []


def docx(name):
    with zipfile.ZipFile(SUB / name) as z:
        return z.read("word/document.xml").decode("utf-8")


# ---------- section cross-references ----------
heads = set()
for m in re.finditer(r"^#{1,4}\s+(\d+(?:\.\d+)?)\.", man, flags=re.M):
    heads.add(m.group(1))
    heads.add(m.group(1).split(".")[0])
for ref, n in collections.Counter(re.findall(r"Sections?\s+(\d+(?:\.\d+)?)", both)).items():
    if ref not in heads:
        problems.append(f"section reference {ref} has no heading ({n}x)")

# ---------- tables and figures: cited <-> defined ----------
defined_t = set(re.findall(r"\*\*Table (S?\d+)\.\*\*", both))
defined_f = set(re.findall(r"\*\*Figure (S?\d+)\.\*\*", both))
cited_t = set(re.findall(r"Tables? (S?\d+)", both))
cited_t |= {x for pair in re.findall(r"Tables (S?\d+) and (S?\d+)", both) for x in pair}
cited_f = set(re.findall(r"Figures? (S?\d+)", both))
for t in sorted(cited_t - defined_t):
    problems.append(f"Table {t} cited but not defined")
for f in sorted(cited_f - defined_f):
    problems.append(f"Figure {f} cited but not defined")
for t in sorted(defined_t - cited_t):
    problems.append(f"Table {t} defined but never cited")
for f in sorted(defined_f - cited_f):
    problems.append(f"Figure {f} defined but never cited")
print(f"tables: {len(defined_t)} defined, {len(cited_t)} cited; "
      f"figures: {len(defined_f)} defined, {len(cited_f)} cited")

# ---------- the abstract's numbers must reappear in the body ----------
abs_m = re.search(r"\*\*Abstract:\*\*(.+?):::", man, flags=re.S)
if abs_m:
    body = man[abs_m.end():]
    # 11% is a rounded bound on the five rarest annotations (max 10.4%, given in the body)
    skip = {"12", "31", "2022", "2023", "2024", "2026", "2.6", "3.3", "11", "11%"}
    for tok in sorted(set(re.findall(r"\d[\d,]*\.?\d*%?", abs_m.group(1)))):
        if len(tok) > 1 and tok not in skip and tok not in body:
            problems.append(f"abstract value {tok!r} does not appear in the body")

# ---------- the analysis inventory ----------
cap = sup.find("**Table S2.**")
data = []
for line in reversed(sup[:cap].rstrip().splitlines()):
    if not line.startswith("|"):
        break
    if set(line) <= set("|:- ") or ("Analysis" in line and "Status" in line):
        continue
    data.append(line)
if data:
    n_all = len(data)
    n_exp = sum(1 for line in data if "Exploratory" in line)
    print(f"Table S2: {n_all} analyses, {n_exp} exploratory, {n_all - n_exp} pre-specified")
    words = {24: "twenty-four", 25: "twenty-five", 26: "twenty-six"}
    if words.get(n_all, "?") not in both.lower():
        problems.append(f"Table S2 has {n_all} rows; the text does not say so in words")

# ---------- phrases that must appear nowhere in the package ----------
FORBIDDEN = [
    (r"\bprosp(?!ective anywhere,|ective\.==)", "withdrawn term or its abbreviation"),
    (r"upper bound for the weighting|Oracle-weight upper bound|oracle-weight bound"
     r"|bounds what any weighting", "withdrawn claim"),
    (r"given in the next column", "reference to a deleted column"),
    (r"close to exhausted", "over-general claim"),
    (r"did not exist at the freeze|had not been mined", "withdrawn chronology"),
    (r"properties of the task|belong to the task and not|whatever builds the clusters",
     "over-general scope claim"),
    (r"Both columns say the same thing", "self-contradiction"),
    (r"\{(?:T_|F_|SUPP|ZENODO|DOI)[A-Za-z_]*\}", "unresolved placeholder"),
    (r"zenodo\.22933176|zenodo\.22945019|zenodo\.23037947", "superseded archive DOI"),
    (r"[\u4e00-\u9fff]", "CJK character"),
]
for pat, why in FORBIDDEN:
    for m in re.finditer(pat, both, flags=re.I):
        problems.append(f"{why}: ...{re.sub(r'  +', ' ', both[max(0, m.start() - 50):m.end() + 50])}...")

for name, txt in (("manuscript", man), ("supplementary", sup)):
    if txt.count("==") % 2:
        problems.append(f"{name}: odd number of == marks")

# ---------- the built .docx ----------
for fname, label, pre, ntab, nfig in (
        ("ZSH_FinTech_MDPI_manuscript_V5.docx", "manuscript", "", 15, 7),
        ("ZSH_supplementary_V5.docx", "supplement", "S", 14, 6)):
    if not (SUB / fname).exists():
        problems.append(f"{fname} not found")
        continue
    plain = re.sub(r"<[^>]+>", "", docx(fname))
    nums = sorted({int(x) for x in re.findall(r"Table " + pre + r"(\d+)\.", plain)})
    if nums != list(range(1, ntab + 1)):
        problems.append(f"{label}: table numbers {nums}, expected 1-{ntab}")
    figs = sorted({int(x) for x in re.findall(r"Figure " + pre + r"(\d+)\.", plain)})
    if figs != list(range(1, nfig + 1)):
        problems.append(f"{label}: figure numbers {figs}, expected 1-{nfig}")
    # one caption per number; a caption may run to several sentences, so count the
    # distinct numbers that carry a caption rather than counting sentence starts
    captioned = {int(x) for x in re.findall(r"Table " + pre + r"(\d+)\.\s*[A-Z]", plain)}
    if captioned != set(nums):
        problems.append(f"{label}: numbered {nums} but captioned {sorted(captioned)}")
    if re.search(r"[\u4e00-\u9fff]", plain):
        problems.append(f"{label}: CJK character in the rendered document")
    # pandoc's mark extension does not nest: a == span opened inside another leaves a
    # literal "==" in the output. Even parity in the source does not catch that, so test
    # the rendered document. This shipped once in the Data Availability Statement.
    if "==" in plain:
        problems.append(f"{label}: {plain.count('==')} literal highlight marker(s) "
                        f"rendered as text")
    print(f"{label}: tables {pre}1-{pre}{ntab}, figures {pre}1-{pre}{nfig}, captions complete")

# ---------- the exported .pdf text layer ----------
# Claims that the supplement's tables are out of order, or that a caption reads a CJK
# glyph instead of "Figure", have been made about the PDF. Check the PDF itself rather
# than inferring from the .docx. Skipped when pypdf is not installed.
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed: PDF text-layer check skipped")
else:
    for name, pre, ntab, nfig in (("ZSH_supplementary_V5", "S", 14, 6),
                                  ("ZSH_FinTech_MDPI_manuscript_V5", "", 15, 7)):
        pdf = SUB / "SUBMIT_V5_pdf_copies" / (name + ".pdf")
        if not pdf.exists():
            pdf = SUB / (name + ".pdf")
        if not pdf.exists():
            problems.append(name + ".pdf not found")
            continue
        t = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages)
        # A caption may cite another table and end that sentence before a capital
        # ("... as in Table 7. Constrained: ..."), which looks like a second caption.
        # Compare the order in which each number FIRST appears.
        seq = [m.group(1) for m in re.finditer(r"Table (" + pre + r"\d+)\.\s*[A-Z]", t)]
        caps = list(dict.fromkeys(seq))
        want = [pre + str(i) for i in range(1, ntab + 1)]
        if caps != want:
            problems.append(name + f".pdf: table captions out of order: {caps}")
        figs = sorted({int(x) for x in re.findall(r"Figure " + pre + r"(\d+)\.", t)})
        if figs != list(range(1, nfig + 1)):
            problems.append(name + f".pdf: figure captions {figs}, expected 1-{nfig}")
        if re.search(r"[\u4e00-\u9fff]", t):
            problems.append(name + ".pdf: CJK character in the text layer")
        # a caption separated from its table by a page break reads as a caption over
        # the wrong table; the build sets keepNext and repeats header rows to stop that
        pages = [pg.extract_text() or "" for pg in PdfReader(str(pdf)).pages]
        for pno, ptxt in enumerate(pages, 1):
            for cm in re.finditer(r"Table (" + pre + r"\d+)\.\s*[A-Z]", ptxt):
                tail = ptxt[cm.end():]
                if len(re.sub(r"\s+", "", re.sub(r"\d{1,4}", " ", tail))) < 40:
                    problems.append(
                        name + f".pdf p{pno}: caption for Table {cm.group(1)} ends the page")
        print(f"{name}.pdf: {len(caps)} captions in order, figures {figs}, no CJK")

# ---------- an image and its caption must land on the same page ----------
# A figure caption follows its image, so the image carries keepNext. When that was set
# on the caption instead, each caption was bound to the NEXT figure and landed on the
# following page above a different image, which reads as a mismatched caption.
try:
    from pypdf import PdfReader as _R
except ImportError:
    pass
else:
    for _name, _pre in (("ZSH_supplementary_V5", "S"), ("ZSH_FinTech_MDPI_manuscript_V5", "")):
        _pdf = SUB / "SUBMIT_V5_pdf_copies" / (_name + ".pdf")
        if not _pdf.exists():
            _pdf = SUB / (_name + ".pdf")
        if not _pdf.exists():
            continue
        _i = _c = 0
        for _n, _pg in enumerate(_R(str(_pdf)).pages, 1):
            _caps = re.findall(r"Figure " + _pre + r"\d+\.", _pg.extract_text() or "")
            _imgs = len(_pg.images) - (1 if _n == 1 else 0)   # page 1 carries the logo
            _i += max(_imgs, 0)
            _c += len(_caps)
            if _i != _c:
                problems.append(f"{_name}.pdf: after p{_n}, {_i} image(s) but {_c} caption(s) "
                                f"- a caption is separated from its image by a page break")
                break
        else:
            print(f"{_name}.pdf: every image and its caption share a page")

# ---------- every figure caption sits under the image it describes ----------
# Checked by hashing each embedded image against the file the caption names, rather
# than by reading the rendered page, where an extractor can pair them wrongly.
FIG_FOR_CAPTION = {
    "ZSH_supplementary_V5.docx": ["F3_weights.png", "F4_profiles.png", "F5_methods.png",
                                  "F6_factorial.png", "F8_drift.png", "F11_atypicality.png"],
    "ZSH_FinTech_MDPI_manuscript_V5.docx": ["F2_design.png", "F1_pipeline.png", "F7_stability.png",
                                            "F9_curves.png", "F10_elliptic.png",
                                            "F12_sensitivity.png", "F13_bench.png"],
}
FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
by_hash = {}
for _p in FIGDIR.glob("*.png"):
    by_hash[hashlib.sha256(_p.read_bytes()).hexdigest()] = _p.name
for fname, expected in FIG_FOR_CAPTION.items():
    if not (SUB / fname).exists():
        continue
    with zipfile.ZipFile(SUB / fname) as z:
        x = z.read("word/document.xml").decode("utf-8")
        rel = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"',
                              z.read("word/_rels/document.xml.rels").decode("utf-8")))
        got = []
        for m in re.finditer(r'r:embed="([^"]+)"', x):
            tgt = rel.get(m.group(1), "")
            if not tgt.startswith("media/"):
                continue
            got.append(by_hash.get(hashlib.sha256(z.read("word/" + tgt)).hexdigest(), "unknown"))
    if got != expected:
        problems.append(f"{fname}: figures in document order are {got}, expected {expected}")
    else:
        print(f"{fname}: {len(got)} figures, each under its own caption")

# ---------- back matter ----------
mplain = re.sub(r"<[^>]+>", "", docx("ZSH_FinTech_MDPI_manuscript_V5.docx")) \
    if (SUB / "ZSH_FinTech_MDPI_manuscript_V5.docx").exists() else man
for k in ("Data Availability Statement", "zenodo.23037946", "Funding:",
          "Institutional Review Board Statement", "Informed Consent Statement",
          "Conflicts of Interest", "Use of Generative AI",
          "dff579fba854e86328297a5d740d736f69b7e1b9c30de8358396e617a3288733"):
    if k not in mplain:
        problems.append(f"back matter: '{k}' missing from the rendered manuscript")

print()
if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
else:
    print("no problems found")
