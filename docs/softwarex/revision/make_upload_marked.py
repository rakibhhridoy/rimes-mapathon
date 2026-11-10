"""Flat, pdflatex-ready marked source for Editorial Manager.

make_marked.sh writes ../softwarex_MARKED.tex for lualatex (lua-ul) with
relative paths. Editorial Manager compiles with pdflatex in one flat folder,
so this copy flattens every path and swaps the lua-ul style for ulem.
Writes upload/softwarex_marked.tex; copy ../softwarex_MARKED.bbl to
upload/softwarex_marked.bbl beside it.
"""
from pathlib import Path

here = Path(__file__).resolve().parent
s = (here.parent / "softwarex_MARKED.tex").read_text(encoding="utf-8")
s = (s.replace(r"\input{../numbers.tex}", r"\input{numbers.tex}")
      .replace("{figures/", "{")
      .replace(r"\bibliography{../references,softwarex}", r"\bibliography{references,softwarex}")
      .replace("\\graphicspath{{../}}\n", ""))
a = s.index("%DIF CUSTOM STYLE")
b = s.index(r"\begin{document}", a)
style = r"""%DIF CUSTOM STYLE  (pdflatex; ulem draws highlight and strike rule in one mark)
\RequirePackage[normalem]{ulem}
\RequirePackage{xcolor}
\definecolor{delyellow}{RGB}{255,240,140}
\definecolor{insblue}{RGB}{0,64,192}
% deletions: yellow highlight with a strike rule, text grey
\DeclareRobustCommand{\DIFhlst}{\bgroup\markoverwith{\rlap{\textcolor{delyellow}{\rule[-0.55ex]{2pt}{2.6ex}}}\textcolor{black!70}{\rule[0.55ex]{2pt}{0.5pt}}}\ULon}
\renewcommand{\DIFdel}[1]{{\color{black!70}\DIFhlst{#1}}}
% the title cannot take ulem leaders; its short deletions use a yellow box
\DeclareRobustCommand{\DIFdelbox}[1]{{\setlength{\fboxsep}{0.5pt}\colorbox{delyellow}{\color{black!70}\sout{#1}}}}
% additions: blue
\renewcommand{\DIFadd}[1]{{\color{insblue}#1}}

"""
s = s[:a] + style + s[b:]
# route the title's deletions to the box style
t0 = s.index("\\title{")
t1 = s.index("\n", t0)
s = s[:t0] + s[t0:t1].replace("\\DIFdel{", "\\DIFdelbox{") + s[t1:]
body = "\n".join(line.split("%")[0] for line in s.split("\n"))
assert "../" not in body and "figures/" not in body, "relative path left in marked source"
out = here / "upload" / "softwarex_marked.tex"
out.write_text(s, encoding="utf-8")
print(f"wrote {out}")
