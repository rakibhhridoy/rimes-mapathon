"""
Write the similarity-check copy of the SoftwareX manuscript as a Word file.

The copy holds the abstract and the body text only: no title, authors,
section headings, tables, images, declarations or references. Figure
captions stay as plain paragraphs. Citations and numbers read exactly as in
the PDF, because the text is typeset by LaTeX from the manuscript itself:
a stripped copy is compiled with the same macros and bibliography on a page
wide enough that every paragraph sets as one line, with hyphenation off, and
its text is read back.

    python docs/softwarex/build_turnitin.py   # -> docs/softwarex/softwarex_turnitin.docx
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "softwarex.tex"
OUT = HERE / "softwarex_turnitin.docx"
# Marks where the reference list starts, so it can be cut from the text.
REFS = "TURNITINREFERENCESSTART"


def stripped_source() -> str:
    tex = SOURCE.read_text()
    preamble = tex[:tex.index("\\begin{document}")]
    abstract = tex[tex.index("\\begin{abstract}") + len("\\begin{abstract}"):tex.index("\\end{abstract}")]
    body = tex[tex.index("\\section{Motivation"):tex.index("%% ── Declarations")]
    # cross-references take the numbers they have in the manuscript's PDF,
    # read from its .aux, since the headings and tables they point at go
    labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}",
                             (HERE / "softwarex.aux").read_text()))
    body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], body)
    # no headings, tables or images; figures keep only their captions, in place
    body = re.sub(r"\\(sub)?section\*?\{[^}]*\}\s*", "", body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", r"\\begin{figure}[H]", body)
    body = re.sub(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", "", body, flags=re.S)
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", body)
    body = re.sub(r"(?<!\\)%[^\n]*", "", body)   # comments, not an escaped \\%
    bibliography = tex[tex.index("\\bibliographystyle"):tex.index("\\end{document}")]
    # a page wide enough for any paragraph on one line, and no hyphenation
    page = ("\\usepackage[paperwidth=560cm,paperheight=500cm,margin=1cm]{geometry}\n"
            "\\hyphenpenalty=10000 \\exhyphenpenalty=10000 \\pagestyle{empty}\n"
            "\\setlength{\\parskip}{1em}\\setlength{\\parindent}{0pt}\n")
    return (preamble + page + "\\begin{document}\n" + abstract.strip() + "\n\n" + body
            + f"\n\n{REFS}\n\n" + bibliography + "\\end{document}\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "softwarex"
        # compile beside the real files, so ../numbers.tex and the .bib resolve
        shutil.copytree(HERE, work, ignore=shutil.ignore_patterns("*.pdf", "*.aux", "._*"))
        shutil.copy(HERE.parent / "numbers.tex", Path(tmp) / "numbers.tex")
        for bib in HERE.parent.glob("*.bib"):
            shutil.copy(bib, Path(tmp) / bib.name)
        (work / "turnitin.tex").write_text(stripped_source())
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "turnitin.tex"],
                       cwd=work, capture_output=True)
        text = subprocess.run(["pdftotext", "-nopgbrk", str(work / "turnitin.pdf"), "-"],
                              capture_output=True, text=True, check=True).stdout
    text = text[:text.index(REFS)]
    # every paragraph is set as one line, so a line is a paragraph
    paragraphs = [" ".join(line.split()) for line in text.splitlines() if line.strip()]

    from docx import Document
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(OUT)
    words = sum(len(p.split()) for p in paragraphs)
    print(f"{OUT}: {len(paragraphs)} paragraphs, {words:,} words")


if __name__ == "__main__":
    main()
