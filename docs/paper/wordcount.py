"""
Word count of docs/paper/paper.tex as the Journal of Flood Risk Management
counts it: the abstract and the body from the Introduction to the end of the
Conclusions, excluding references, figures and tables (their captions too).
Each number macro such as \\RRLabelGap{} counts as one word.

    python docs/paper/wordcount.py          # total, then per section
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 6000


def words(tex: str) -> int:
    tex = re.sub(r"(?<!\\)%.*", "", tex)                           # comments
    tex = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{(table|figure)\*?\}",
                 " ", tex, flags=re.S)                             # floats
    tex = re.sub(r"\\(label|ref|eqref|cite[pt]?|citep|citet|url|href)\{[^}]*\}", " x ", tex)
    tex = re.sub(r"\\(sub)*section\*?\{[^}]*\}", " ", tex)          # headings
    tex = re.sub(r"\\[A-Za-z]+\{\}", " x ", tex)                   # number macros
    tex = re.sub(r"\\[A-Za-z]+", " ", tex)                         # other commands
    tex = re.sub(r"[{}$~\\]", " ", tex)
    return len([w for w in tex.split() if re.search(r"[A-Za-z0-9]", w)])


def main():
    tex = (HERE / "paper.tex").read_text()
    abstract = (HERE / "abstract.tex").read_text()
    body = tex[tex.index(r"\section{Introduction}"):tex.index(r"\section*{Data and code availability}")]
    parts = re.split(r"(\\section\{[^}]*\})", body)
    rows, name = [("Abstract", words(abstract))], None
    for part in parts:
        if part.startswith(r"\section{"):
            name = part[9:-1]
        elif name:
            rows.append((name, words(part)))
    total = sum(n for _, n in rows)
    print(f"{total} words against a limit of {LIMIT} "
          f"({'within' if total <= LIMIT else f'{total - LIMIT} over'})")
    for name, n in rows:
        print(f"  {n:5d}  {name}")
    return 0 if total <= LIMIT else 1


if __name__ == "__main__":
    sys.exit(main())
