"""Write references.bib from verified.json, keyed as firstauthorYEAR."""
import json, re, unicodedata
from pathlib import Path

c = json.load(open(Path(__file__).with_name("verified.json")))

KEEP = {  # key -> DOI, only sources the review actually cites
 "rahman2019": "10.1007/s41748-019-00123-y", "islam2021": "10.1016/j.gsf.2020.09.006",
 "talukdar2020": "10.1007/s00477-020-01862-5", "hasan2023": "10.1016/j.ocecoaman.2023.106503",
 "islamr2024": "10.1016/j.envc.2023.100833", "adnan2023": "10.1016/j.jenvman.2022.116813",
 "islam2025review": "10.1007/s12145-025-01816-x", "uddin2019": "10.3390/rs11131581",
 "twele2016": "10.1080/01431161.2016.1192304", "tellman2021": "10.1038/s41586-021-03695-w",
 "amitrano2024": "10.3390/rs16040656", "pekel2016": "10.1038/nature20584",
 "roberts2017": "10.1111/ecog.02881", "ploton2020": "10.1038/s41467-020-18321-y",
 "meyer2021": "10.1111/2041-210x.13650", "wadoux2021": "10.1016/j.ecolmodel.2021.109692",
 "bentivoglio2022": "10.5194/hess-26-4345-2022", "wang2023": "10.3389/feart.2023.1132722",
 "ma2025": "10.1016/j.catena.2025.109029", "hossain2025": "10.1007/s43621-025-02084-x",
 "roy2025": "10.1007/s44288-025-00337-w", "hasanm2024": "10.1007/s11356-024-34949-5",
 "islama2025": "10.1016/j.geogeo.2025.100354", "rabby2019": "10.3390/data5010004",
 "juang2019": "10.1371/journal.pone.0218657", "reichenbach2018": "10.1016/j.earscirev.2018.03.001",
 "steger2016": "10.5194/nhess-16-2729-2016", "steger2021": "10.1016/j.scitotenv.2021.145935",
 "kirschbaum2018": "10.1002/2017ef000715", "barbetmassin2012": "10.1111/j.2041-210x.2011.00172.x",
 "herfort2023": "10.1038/s41467-023-39698-6", "koks2019": "10.1038/s41467-019-10442-3",
 "nobre2011": "10.1016/j.jhydrol.2011.03.051", "beven1979": "10.1080/02626667909491834",
 "getis1992": "10.1111/j.1538-4632.1992.tb00261.x", "niculescu2005": "10.1145/1102351.1102430",
 "zadrozny2002": "10.1145/775047.775151",
}

ODD_PUNCT = {"\u2024": ".", "\u2010": "-", "\u2011": "-", "\u2012": "-",
             "\u2013": "--", "\u2014": "---", "\u2018": "`", "\u2019": "'",
             "\u201c": "``", "\u201d": "''", "\u00a0": " ", "\u2009": " ",
             "\u202f": " ", "\u2002": " "}


def tex(s):
    s = unicodedata.normalize("NFC", str(s or ""))
    for bad, good in ODD_PUNCT.items():
        s = s.replace(bad, good)
    s = re.sub(r"<[^>]+>", "", s)
    for a, b in [("&amp;", "\\&"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]:
        s = s.replace(a, b) if a != "&" else re.sub(r"(?<!\\)&", r"\\&", s)
    return re.sub(r"\s+", " ", s).strip()

# Crossref occasionally stores a name in an order or case that misrepresents
# how the author publishes. Corrections are listed explicitly so they stay
# reviewable.
AUTHOR_FIX = {
    "Towfiqul Islam, Abu Reza Md": "Islam, Abu Reza Md. Towfiqul",
}


def fix_author(name: str) -> str:
    name = AUTHOR_FIX.get(name, name)
    family, _, given = name.partition(", ")
    if family.isupper() and len(family) > 2:        # "BEVEN" -> "Beven"
        family = family.title()
    return f"{family}, {given}" if given else family


out = []
problems = []
for key, doi in KEEP.items():
    r = c[doi]
    authors = " and ".join(tex(fix_author(a)) for a in r.get("authors", [])) or "{Anonymous}"
    kind = "inproceedings" if "Proceedings" in (r.get("journal") or "") else "article"
    venue = "booktitle" if kind == "inproceedings" else "journal"
    fields = {
        "author": authors, "title": "{" + tex(r["title"]) + "}", venue: tex(r.get("journal")),
        "year": r.get("year"), "volume": r.get("volume"), "number": r.get("issue"),
        "pages": (r.get("pages") or "").replace("-", "--") or None, "doi": doi,
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    out.append(f"@{kind}{{{key},\n{body}\n}}")

out.append("""@misc{hamilton2017,
  author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  title = {{Inductive Representation Learning on Large Graphs}},
  year = {2017},
  howpublished = {Advances in Neural Information Processing Systems 30; arXiv:1706.02216},
  eprint = {1706.02216}, archivePrefix = {arXiv}
}""")
out.append("""@techreport{inform2017,
  author = {{European Commission Joint Research Centre}},
  title = {{Index for Risk Management -- INFORM: Concept and Methodology, Version 2017}},
  institution = {Publications Office of the European Union},
  year = {2017}, doi = {10.2760/094023}
}""")
out.append("""@misc{unspider,
  author = {{UN-SPIDER}},
  title = {{Recommended Practice: Flood Mapping and Damage Assessment Using Sentinel-1 SAR Data in Google Earth Engine}},
  howpublished = {\\url{https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping}},
  year = {2019}
}""")
text = "\n\n".join(out) + "\n"
# Anything outside Latin-1 that survives is reported rather than shipped:
# pdflatex rejects it at compile time.
odd = sorted({ch for ch in text if ord(ch) > 0xFF})
if odd:
    print("WARNING, unmapped characters:", [f"U+{ord(ch):04X} {ch}" for ch in odd])
Path(__file__).parent.parent.joinpath("references.bib").write_text(text)
print(f"wrote {len(out)} entries")
