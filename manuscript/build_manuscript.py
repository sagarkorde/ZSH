"""Build the MDPI-formatted manuscript (.docx) from Markdown.

Steps
1. Resolve citation keys [@Key] to numbers in order of first appearance and
   write the reference list from references.json in the same order.
2. Convert with pandoc, using the submitted MDPI manuscript as reference document.
3. Post-process word/document.xml: map pandoc styles to MDPI styles, format
   tables (three-line), equations with numbers, captions, and references.

Usage: python build_manuscript.py manuscript.md references.json out.docx
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# MDPI's own template, used as a pandoc reference document. It is not
# redistributed here; set ZSH_MDPI_TEMPLATE to your copy.
TEMPLATE = Path(os.environ.get("ZSH_MDPI_TEMPLATE",
                               HERE.parent / "ZSH_FinTech_MDPI_manuscript.docx"))

STYLE_MAP = {
    "Heading1": "MDPI21heading1", "Heading2": "MDPI22heading2", "Heading3": "MDPI23heading3",
    "FirstParagraph": "MDPI31text", "BodyText": "MDPI31text", "Compact": "MDPI38bullet",
    "TableCaption": "MDPI41tablecaption", "ImageCaption": "MDPI51figurecaption",
    "CaptionedFigure": "MDPI52figure", "Figure": "MDPI52figure",
}


def resolve_citations(md, refs):
    order = []

    def number(key):
        if key not in refs:
            raise KeyError(f"unknown reference key: {key}")
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    def repl(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = sorted(number(k) for k in keys)
        # compress consecutive runs: 1,2,3 -> 1–3
        parts, start, prev = [], nums[0], nums[0]
        for n in nums[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n
                continue
            parts.append(f"{start}" if start == prev else (f"{start},{prev}" if prev == start + 1 else f"{start}–{prev}"))
            if n is not None:
                start = prev = n
        return "[" + ",".join(parts) + "]"

    md = re.sub(r"\[(@[A-Za-z0-9_\-]+(?:\s*;\s*@[A-Za-z0-9_\-]+)*)\]", repl, md)
    return md, order


DOI_URL = re.compile(r"\s*https?://(?:dx\.)?doi\.org/(\S+?)\.?\s*$")


def crossref_tag(entry):
    """MDPI style: a trailing DOI is printed as a [CrossRef] link rather than a bare URL."""
    m = DOI_URL.search(entry.rstrip())
    if not m:
        return entry
    head = DOI_URL.sub("", entry.rstrip()).rstrip()
    return head + " [" + r"\[CrossRef\]" + "](<https://doi.org/" + m.group(1) + ">)"


def reference_block(order, refs):
    lines = ["", '::: {custom-style="MDPI_8.1_references"}']
    for k in order:
        lines.append(crossref_tag(refs[k]))
        lines.append("")
    lines.append(":::")
    return "\n".join(lines)

def number_equations(md):
    """Lines of the form  $$ ... $$ {#eq:label}  become numbered equations; \\eqref{label} -> (n)."""
    labels = {}

    def repl(m):
        labels[m.group(2)] = len(labels) + 1
        return f"$${m.group(1)}$$EQNUM{labels[m.group(2)]}EQNUM"

    md = re.sub(r"\$\$(.+?)\$\$\s*\{#eq:([A-Za-z0-9_]+)\}", repl, md, flags=re.S)
    md = re.sub(r"\\eqref\{([A-Za-z0-9_]+)\}", lambda m: f"({labels[m.group(1)]})", md)
    return md


def postprocess(xml):
    for a, b in STYLE_MAP.items():
        xml = xml.replace(f'<w:pStyle w:val="{a}" />', f'<w:pStyle w:val="{b}" />')
        xml = xml.replace(f'<w:pStyle w:val="{a}"/>', f'<w:pStyle w:val="{b}"/>')
    # paragraphs without a style become MDPI body text
    xml = re.sub(r"<w:p>(?!<w:pPr>)", '<w:p><w:pPr><w:pStyle w:val="MDPI31text"/></w:pPr>', xml)
    # list items: pandoc's numbering, MDPI list indentation (as in the template's list definitions)
    xml = re.sub(r"(<w:numPr>(?:(?!</w:numPr>).)*</w:numPr>)(?!<w:ind)",
                 r'\1<w:ind w:left="3033" w:hanging="425"/>', xml, flags=re.S)
    # table cells: body style, no indent
    def fix_table(m):
        t = m.group(0)
        t = re.sub(r'<w:pStyle w:val="[^"]+" ?/>', '<w:pStyle w:val="MDPI42tablebody"/>', t)
        t = re.sub(r"<w:p>(?!<w:pPr>)", '<w:p><w:pPr><w:pStyle w:val="MDPI42tablebody"/></w:pPr>', t)
        # three-line table: top and bottom borders on the table, header row bottom border
        t = re.sub(r"<w:tblStyle [^>]*/>", "", t)
        borders = ('<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
                   '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/></w:tblBorders>')
        t = re.sub(r"(<w:tblPr>)", r"\1" + borders, t, count=1)
        t = re.sub(r'<w:tblW [^>]*/>', '<w:tblW w:w="5000" w:type="pct"/><w:jc w:val="center"/>', t, count=1)
        # header row: bottom border on first row cells
        first_row_end = t.find("</w:tr>")
        if first_row_end > 0:
            head = t[:first_row_end]
            head = re.sub(r"<w:tcPr>", '<w:tcPr><w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tcBorders>', head)
            head = re.sub(r"<w:tc>(?!<w:tcPr>)", '<w:tc><w:tcPr><w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tcBorders></w:tcPr>', head)
            t = head + t[first_row_end:]
        return t
    xml = re.sub(r"<w:tbl>.*?</w:tbl>", fix_table, xml, flags=re.S)
    # numbered equations: paragraph containing m:oMathPara followed by EQNUMnEQNUM
    def fix_eq(m):
        body, n = m.group(1), m.group(2)
        body = body.replace("<m:oMathPara>", "").replace("</m:oMathPara>", "")
        body = re.sub(r"<m:oMathParaPr>.*?</m:oMathParaPr>", "", body, flags=re.S)
        return ('<w:p><w:pPr><w:pStyle w:val="MDPI39equation"/><w:tabs><w:tab w:val="center" w:pos="4680"/>'
                '<w:tab w:val="right" w:pos="9360"/></w:tabs></w:pPr><w:r><w:tab/></w:r>' + body +
                f'<w:r><w:tab/><w:t xml:space="preserve">({n})</w:t></w:r></w:p>')
    xml = re.sub(r'<w:p>(?:<w:pPr>(?:(?!</w:pPr>).)*</w:pPr>)?((?:(?!</w:p>).)*?<m:oMathPara>.*?</m:oMathPara>)'
                 r'(?:(?!</w:p>).)*?</w:p>\s*<w:p>(?:<w:pPr>(?:(?!</w:pPr>).)*</w:pPr>)?<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?'
                 r'<w:t[^>]*>EQNUM(\d+)EQNUM</w:t></w:r></w:p>', fix_eq, xml, flags=re.S)
    xml = re.sub(r"EQNUM(\d+)EQNUM", r"(\1)", xml)
    return xml


def main():
    md_path, refs_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    refs = json.load(open(refs_path, encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")
    md = number_equations(md)
    md, order = resolve_citations(md, refs)
    md = md.replace("<!-- REFERENCES -->", reference_block(order, refs))
    tmp = Path(tempfile.mkdtemp())
    (tmp / "in.md").write_text(md, encoding="utf-8")
    raw = tmp / "raw.docx"
    subprocess.run(["pandoc", str(tmp / "in.md"), "-f", "markdown+pipe_tables+tex_math_dollars",
                    "-o", str(raw), f"--reference-doc={TEMPLATE}", f"--resource-path={md_path.parent}"],
                   check=True)
    with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = postprocess(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    shutil.rmtree(tmp)
    print(f"wrote {out}; {len(order)} references cited")


if __name__ == "__main__":
    main()
